#!/usr/bin/env python3
"""
Run AMB3R SLAM on input MCAP files and write pose results as output MCAPs.

Reads camera frames from input MCAPs, runs AMB3R visual odometry,
and writes /slam/tf (FrameTransforms) and /slam/health (Health) topics
to output MCAPs compatible with evaluate_trajectory.py.

Usage:
    python run_slam_mcap.py \
        --input_dir ../data/eval_data_half \
        --output_dir ../data/eval_data_half_output \
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

# -- MCAP imports --
from mcap.reader import make_reader
from mcap.writer import Writer as McapWriter

# -- Protobuf schema imports --
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SCHEMA_PY_DIR = REPO_ROOT / "microagi-schemas" / "python"
sys.path.insert(0, str(SCHEMA_PY_DIR))

from foxglove.FrameTransforms_pb2 import FrameTransforms  # noqa: E402
from microagi.health_pb2 import Health  # noqa: E402

# -- AMB3R imports --
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "thirdparty"))

IMAGE_TOPIC = "/camera/color/0/image"


def get_args():
    parser = argparse.ArgumentParser(description="Run AMB3R SLAM on MCAP files")
    parser.add_argument("--input_dir", type=str, default="../data/eval_data_half")
    parser.add_argument("--output_dir", type=str, default="../data/eval_data_half_output")
    parser.add_argument("--ckpt_path", type=str, default="./checkpoints/amb3r.pt")
    parser.add_argument("--model_name", type=str, default="amb3r", choices=["amb3r", "da3"])
    parser.add_argument("--resolution", type=int, nargs=2, default=[448, 336], help="W H")
    parser.add_argument("--map_every", type=int, default=16, help="Run model every N frames")
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


def extract_frames_from_mcap(mcap_path, resolution):
    """Extract and preprocess RGB frames from an input MCAP file.

    Decodes H.265 CompressedVideo messages directly to tensors.

    Returns:
        List of (timestamp_ns, tensor) where tensor is (3, H, W) in [-1, 1].
    """
    from foxglove.CompressedVideo_pb2 import CompressedVideo

    frames = []
    codec = av.CodecContext.create("hevc", "r")

    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for _, channel, message in reader.iter_messages(topics=[IMAGE_TOPIC]):
            msg = CompressedVideo()
            msg.ParseFromString(message.data)

            pkt = av.Packet(msg.data)
            for frame in codec.decode(pkt):
                rgb = frame.to_ndarray(format="rgb24")
                tensor = preprocess_frame(rgb, resolution)
                frames.append((message.log_time, tensor))

    # Flush decoder
    for frame in codec.decode():
        rgb = frame.to_ndarray(format="rgb24")
        tensor = preprocess_frame(rgb, resolution)
        ts = frames[-1][0] + 1 if frames else 0
        frames.append((ts, tensor))

    print(f"  Extracted {len(frames)} frames from {Path(mcap_path).name}")
    return frames


def run_slam(model, frame_tensors, map_every=16):
    """Run AMB3R SLAM on preprocessed frame tensors.

    Args:
        model: loaded AMB3R model on GPU
        frame_tensors: list of (3, H, W) tensors in [-1, 1]
        map_every: run model every N frames

    Returns:
        poses: numpy array (T, 4, 4) camera-to-world matrices
    """
    from slam.pipeline import AMB3R_VO

    images = torch.stack(frame_tensors).unsqueeze(0)  # (1, T, 3, H, W)
    num_frames = images.shape[1]
    print(f"  Running SLAM on {num_frames} frames (map_every={map_every})...")

    pipeline = AMB3R_VO(model, cfg_path=str(SCRIPT_DIR / "slam" / "slam_config.yaml"))
    pipeline.cfg.skip_point_head = True
    pipeline.cfg.pts_by_unprojection = True
    pipeline.cfg.map_every = map_every
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


def write_output_mcap(output_path, timestamps_ns, poses):
    """Write poses and health to an output MCAP file."""
    ft_schema_data = FrameTransforms.DESCRIPTOR.file.serialized_pb
    health_schema_data = Health.DESCRIPTOR.file.serialized_pb

    with open(output_path, "wb") as f:
        writer = McapWriter(f)
        writer.start()

        tf_schema_id = writer.register_schema(
            name="foxglove.FrameTransforms", encoding="protobuf", data=ft_schema_data,
        )
        health_schema_id = writer.register_schema(
            name="microagi.Health", encoding="protobuf", data=health_schema_data,
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


def get_output_filename(input_filename):
    """Map input MCAP name to output MCAP name."""
    stem = Path(input_filename).stem
    short_id = stem.split("-")[0]
    return f"output_{short_id}.mcap"


def main():
    args = get_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    from amb3r.model_zoo import load_model

    print("Loading AMB3R model...")
    model = load_model(args.model_name, ckpt_path=args.ckpt_path)
    model.cuda()
    model.eval()

    # torch.compile the VGGT frontend aggregator for ~35% inference speedup
    model.front_end.model.aggregator = torch.compile(
        model.front_end.model.aggregator, mode="reduce-overhead", dynamic=False,
    )

    mcap_files = sorted(input_dir.glob("*.mcap"))
    if not mcap_files:
        print(f"No MCAP files found in {input_dir}")
        return

    for mcap_file in mcap_files:
        print(f"\nProcessing {mcap_file.name}...")

        frames = extract_frames_from_mcap(str(mcap_file), tuple(args.resolution))
        if not frames:
            print("  No frames extracted, skipping.")
            continue

        timestamps_ns = [f[0] for f in frames]
        tensors = [f[1] for f in frames]

        poses = run_slam(model, tensors, map_every=args.map_every)

        if len(poses) != len(timestamps_ns):
            print(
                f"  WARNING: {len(poses)} poses vs {len(timestamps_ns)} timestamps. "
                f"Using min({len(poses)}, {len(timestamps_ns)})."
            )
            n = min(len(poses), len(timestamps_ns))
            poses = poses[:n]
            timestamps_ns = timestamps_ns[:n]

        output_name = get_output_filename(mcap_file.name)
        output_path = output_dir / output_name
        write_output_mcap(str(output_path), timestamps_ns, poses)

    print("\nDone!")


if __name__ == "__main__":
    main()
