#!/usr/bin/env python3
"""
Run AMB3R SLAM on MP4 video files and write pose results as output MCAPs.

Extracts frames from MP4, runs AMB3R visual odometry, and writes /slam/tf
and /slam/health topics to output MCAPs compatible with evaluate_trajectory.py.

Usage:
    python run_slam_mp4.py \
        --input_dir ../data/eval_data_mp4 \
        --output_dir ../data/eval_data_mp4_output \
        --ckpt_path ./checkpoints/amb3r.pt
"""

import argparse
import sys
from pathlib import Path

import av
import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from mcap.writer import Writer as McapWriter

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SCHEMA_PY_DIR = REPO_ROOT / "microagi-schemas" / "python"
sys.path.insert(0, str(SCHEMA_PY_DIR))

from foxglove.FrameTransforms_pb2 import FrameTransforms  # noqa: E402
from microagi.health_pb2 import Health  # noqa: E402

sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "thirdparty"))


def get_args():
    parser = argparse.ArgumentParser(description="Run AMB3R SLAM on MP4 files")
    parser.add_argument("--input_dir", type=str, default="../data/eval_data_mp4")
    parser.add_argument("--output_dir", type=str, default="../data/eval_data_mp4_output")
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints/amb3r.pt")
    parser.add_argument("--model_name", type=str, default="amb3r", choices=["amb3r", "da3"])
    parser.add_argument("--resolution", type=int, nargs=2, default=[518, 392], help="W H")
    parser.add_argument("--max_frames", type=int, default=0, help="Max frames to process (0=all)")
    parser.add_argument("--frame_step", type=int, default=1, help="Use every Nth frame")
    parser.add_argument(
        "--example_mcap",
        type=str,
        default="../data/eval_data_half_output/output_example.mcap",
        help="Example MCAP to copy /slam/tf and /slam/health schema data from",
    )
    return parser.parse_args()


def preprocess_frame(rgb_array, resolution):
    """Center-crop and resize a (H,W,3) uint8 array to target resolution.

    Replicates DemoDataset._center_crop_and_resize from slam/datasets/demo.py.

    Returns:
        (3, target_H, target_W) float32 tensor in [-1, 1].
    """
    H, W = rgb_array.shape[:2]
    target_W, target_H = resolution

    # Portrait swap
    if target_W >= target_H and H > 1.1 * W:
        target_W, target_H = target_H, target_W

    target_aspect = target_W / target_H
    img_aspect = W / H

    # Center crop to target aspect ratio
    if img_aspect > target_aspect:
        new_W = int(round(H * target_aspect))
        left = (W - new_W) // 2
        rgb_array = rgb_array[:, left:left + new_W]
    elif img_aspect < target_aspect:
        new_H = int(round(W / target_aspect))
        top = (H - new_H) // 2
        rgb_array = rgb_array[top:top + new_H]

    # Resize
    rgb_array = cv2.resize(rgb_array, (target_W, target_H), interpolation=cv2.INTER_LANCZOS4)

    # To tensor [-1, 1]
    return torch.from_numpy(rgb_array.copy()).permute(2, 0, 1).float() / 255.0 * 2.0 - 1.0


def extract_frames_from_mp4(mp4_path, resolution, max_frames=0, frame_step=1):
    """Extract and preprocess RGB frames from an MP4 file.

    Returns:
        List of (timestamp_ns, tensor) where tensor is (3, H, W) in [-1, 1].
    """
    container = av.open(mp4_path)
    stream = container.streams.video[0]
    time_base = float(stream.time_base)

    frames = []
    saved = 0
    for idx, frame in enumerate(container.decode(video=0)):
        if idx % frame_step != 0:
            continue
        if max_frames > 0 and saved >= max_frames:
            break

        if frame.pts is not None:
            ts_ns = int(frame.pts * time_base * 1e9)
        else:
            ts_ns = int(idx / float(stream.average_rate) * 1e9)

        rgb = frame.to_ndarray(format="rgb24")
        tensor = preprocess_frame(rgb, resolution)
        frames.append((ts_ns, tensor))
        saved += 1

    container.close()
    print(f"  Extracted {len(frames)} frames from {Path(mp4_path).name}")
    return frames


def run_slam(model, frame_tensors):
    """Run AMB3R SLAM on preprocessed frame tensors.

    Args:
        model: loaded AMB3R model on GPU
        frame_tensors: list of (3, H, W) tensors in [-1, 1]

    Returns:
        poses: numpy array (T, 4, 4) camera-to-world matrices
    """
    from slam.pipeline import AMB3R_VO

    images = torch.stack(frame_tensors).unsqueeze(0)  # (1, T, 3, H, W)
    num_frames = images.shape[1]
    print(f"  Running SLAM on {num_frames} frames...")

    pipeline = AMB3R_VO(model, cfg_path=str(SCRIPT_DIR / "slam" / "slam_config.yaml"))
    memory = pipeline.run(images)
    poses = memory.poses.cpu().numpy()
    return poses


def pose_to_translation_quaternion(pose_4x4):
    """Convert 4x4 c2w matrix to (translation, quaternion_wxyz)."""
    translation = pose_4x4[:3, 3]
    rot = Rotation.from_matrix(pose_4x4[:3, :3])
    quat_xyzw = rot.as_quat()
    quat_wxyz = (quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2])
    return translation, quat_wxyz


def get_slam_schemas_from_example(example_path):
    """Extract schema data for /slam/tf and /slam/health from an example MCAP."""
    from mcap.reader import make_reader

    slam_tf_schema = None
    slam_health_schema = None
    with open(example_path, "rb") as f:
        reader = make_reader(f)
        summary = reader.get_summary()
        for cid, ch in summary.channels.items():
            schema = summary.schemas[ch.schema_id]
            if ch.topic == "/slam/tf":
                slam_tf_schema = schema
            elif ch.topic == "/slam/health":
                slam_health_schema = schema
    return slam_tf_schema, slam_health_schema


def write_output_mcap(output_path, timestamps_ns, poses, slam_tf_schema=None, slam_health_schema=None):
    """Write poses and health to an output MCAP file."""
    tf_schema_data = slam_tf_schema.data if slam_tf_schema else FrameTransforms.DESCRIPTOR.file.serialized_pb
    tf_schema_name = slam_tf_schema.name if slam_tf_schema else "foxglove.FrameTransforms"
    health_schema_data = slam_health_schema.data if slam_health_schema else Health.DESCRIPTOR.file.serialized_pb
    health_schema_name = slam_health_schema.name if slam_health_schema else "microagi.Health"

    with open(output_path, "wb") as f:
        writer = McapWriter(f)
        writer.start()

        tf_schema_id = writer.register_schema(
            name=tf_schema_name, encoding="protobuf", data=tf_schema_data,
        )
        health_schema_id = writer.register_schema(
            name=health_schema_name, encoding="protobuf", data=health_schema_data,
        )
        tf_channel_id = writer.register_channel(
            topic="/slam/tf", message_encoding="protobuf", schema_id=tf_schema_id,
        )
        health_channel_id = writer.register_channel(
            topic="/slam/health", message_encoding="protobuf", schema_id=health_schema_id,
        )

        for ts_ns, pose in zip(timestamps_ns, poses):
            translation, quat_wxyz = pose_to_translation_quaternion(pose)
            seconds = int(ts_ns // 1_000_000_000)
            nanos = int(ts_ns % 1_000_000_000)

            ft_msg = FrameTransforms()
            tf = ft_msg.transforms.add()
            tf.timestamp.seconds = seconds
            tf.timestamp.nanos = nanos
            tf.parent_frame_id = "world"
            tf.child_frame_id = "camera"
            tf.translation.x = float(translation[0])
            tf.translation.y = float(translation[1])
            tf.translation.z = float(translation[2])
            tf.rotation.w = float(quat_wxyz[0])
            tf.rotation.x = float(quat_wxyz[1])
            tf.rotation.y = float(quat_wxyz[2])
            tf.rotation.z = float(quat_wxyz[3])

            writer.add_message(
                channel_id=tf_channel_id, log_time=int(ts_ns),
                data=ft_msg.SerializeToString(), publish_time=int(ts_ns),
            )

            h_msg = Health()
            h_msg.timestamp.seconds = seconds
            h_msg.timestamp.nanos = nanos
            h_msg.valid = True

            writer.add_message(
                channel_id=health_channel_id, log_time=int(ts_ns),
                data=h_msg.SerializeToString(), publish_time=int(ts_ns),
            )

        writer.finish()

    print(f"  Wrote {len(timestamps_ns)} poses to {Path(output_path).name}")


def main():
    args = get_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load example schemas if available
    slam_tf_schema, slam_health_schema = None, None
    if args.example_mcap and Path(args.example_mcap).exists():
        slam_tf_schema, slam_health_schema = get_slam_schemas_from_example(args.example_mcap)
        print(f"Loaded SLAM schemas from {Path(args.example_mcap).name}")

    from amb3r.model_zoo import load_model

    print("Loading AMB3R model...")
    model = load_model(args.model_name, ckpt_path=args.ckpt_path)
    model.cuda()
    model.eval()

    # torch.compile the VGGT frontend aggregator for ~35% inference speedup
    model.front_end.model.aggregator = torch.compile(
        model.front_end.model.aggregator, mode="reduce-overhead", dynamic=False,
    )

    mp4_files = sorted(input_dir.glob("*.mp4"))
    if not mp4_files:
        print(f"No MP4 files found in {input_dir}")
        return

    for mp4_file in mp4_files:
        print(f"\nProcessing {mp4_file.name}...")

        frames = extract_frames_from_mp4(
            str(mp4_file), tuple(args.resolution),
            max_frames=args.max_frames, frame_step=args.frame_step,
        )
        if not frames:
            print("  No frames extracted, skipping.")
            continue

        timestamps_ns = [f[0] for f in frames]
        tensors = [f[1] for f in frames]

        poses = run_slam(model, tensors)

        n = min(len(poses), len(timestamps_ns))
        if len(poses) != len(timestamps_ns):
            print(f"  WARNING: {len(poses)} poses vs {len(timestamps_ns)} timestamps, using {n}.")
        poses = poses[:n]
        timestamps_ns = timestamps_ns[:n]

        output_name = f"{mp4_file.stem}.mcap"
        output_path = output_dir / output_name
        write_output_mcap(str(output_path), timestamps_ns, poses, slam_tf_schema, slam_health_schema)

    print("\nDone!")


if __name__ == "__main__":
    main()
