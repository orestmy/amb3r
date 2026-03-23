#!/usr/bin/env python3
"""
Merge input MCAP (with camera/IMU data) and output MCAP (with /slam/tf and
/slam/health) into a single MCAP that matches the format of output_example.mcap.

All original input topics are copied verbatim. The /slam/tf and /slam/health
channels use schema data from output_example.mcap for exact format compatibility.

Usage:
    python merge_mcap.py \
        --input_dir ../data/eval_data_half \
        --output_dir ../data/eval_data_half_output \
        --example ../data/eval_data_half_output/output_example.mcap
"""

import argparse
from pathlib import Path

from mcap.reader import make_reader
from mcap.writer import Writer as McapWriter


def get_args():
    parser = argparse.ArgumentParser(description="Merge input + SLAM output MCAPs")
    parser.add_argument("--input_dir", type=str, default="../data/eval_data_half")
    parser.add_argument("--output_dir", type=str, default="../data/eval_data_half_output")
    parser.add_argument(
        "--example",
        type=str,
        default="../data/eval_data_half_output/output_example.mcap",
    )
    return parser.parse_args()


def get_slam_schemas_from_example(example_path):
    """Extract schema data for /slam/tf and /slam/health from the example MCAP."""
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


def get_output_filename(input_filename):
    stem = Path(input_filename).stem
    short_id = stem.split("-")[0]
    return f"output_{short_id}.mcap"


def merge_mcaps(input_mcap, slam_mcap, output_path, slam_tf_schema, slam_health_schema):
    """Merge input MCAP topics with SLAM output topics into a single file."""

    with open(output_path, "wb") as out_f:
        writer = McapWriter(out_f)
        writer.start()

        # --- Pass 1: Register all schemas and channels from input MCAP ---
        input_schema_map = {}  # old_schema_id -> new_schema_id
        input_channel_map = {}  # old_channel_id -> new_channel_id

        with open(input_mcap, "rb") as f:
            reader = make_reader(f)
            summary = reader.get_summary()

            for old_sid, schema in summary.schemas.items():
                new_sid = writer.register_schema(
                    name=schema.name,
                    encoding=schema.encoding,
                    data=schema.data,
                )
                input_schema_map[old_sid] = new_sid

            for old_cid, ch in summary.channels.items():
                new_cid = writer.register_channel(
                    topic=ch.topic,
                    message_encoding=ch.message_encoding,
                    schema_id=input_schema_map[ch.schema_id],
                    metadata=ch.metadata,
                )
                input_channel_map[old_cid] = new_cid

        # --- Register /slam/tf and /slam/health using example schemas ---
        slam_tf_schema_id = writer.register_schema(
            name=slam_tf_schema.name,
            encoding=slam_tf_schema.encoding,
            data=slam_tf_schema.data,
        )
        slam_health_schema_id = writer.register_schema(
            name=slam_health_schema.name,
            encoding=slam_health_schema.encoding,
            data=slam_health_schema.data,
        )

        slam_tf_channel_id = writer.register_channel(
            topic="/slam/tf",
            message_encoding="protobuf",
            schema_id=slam_tf_schema_id,
        )
        slam_health_channel_id = writer.register_channel(
            topic="/slam/health",
            message_encoding="protobuf",
            schema_id=slam_health_schema_id,
        )

        # --- Collect all messages with timestamps for time-sorted writing ---
        all_messages = []

        # Read all input messages
        with open(input_mcap, "rb") as f:
            reader = make_reader(f)
            for _, channel, message in reader.iter_messages():
                new_cid = input_channel_map[channel.id]
                all_messages.append(
                    (message.log_time, new_cid, message.data, message.publish_time)
                )

        # Read SLAM output messages
        slam_channel_remap = {}
        with open(slam_mcap, "rb") as f:
            reader = make_reader(f)
            for _, channel, message in reader.iter_messages():
                if channel.topic == "/slam/tf":
                    new_cid = slam_tf_channel_id
                elif channel.topic == "/slam/health":
                    new_cid = slam_health_channel_id
                else:
                    continue
                all_messages.append(
                    (message.log_time, new_cid, message.data, message.publish_time)
                )

        # Sort by log_time for proper interleaving
        all_messages.sort(key=lambda x: x[0])

        # Write all messages
        for log_time, channel_id, data, publish_time in all_messages:
            writer.add_message(
                channel_id=channel_id,
                log_time=log_time,
                data=data,
                publish_time=publish_time,
            )

        writer.finish()

    print(f"  Wrote {len(all_messages)} messages to {Path(output_path).name}")


def main():
    args = get_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Get schemas from example
    slam_tf_schema, slam_health_schema = get_slam_schemas_from_example(args.example)
    print(f"Loaded SLAM schemas from {Path(args.example).name}")

    # Find matching pairs
    for input_mcap in sorted(input_dir.glob("*.mcap")):
        slam_output_name = get_output_filename(input_mcap.name)
        slam_mcap = output_dir / slam_output_name
        if not slam_mcap.exists():
            print(f"Skipping {input_mcap.name}: no matching {slam_output_name}")
            continue

        # Output with _upd suffix
        short_id = input_mcap.stem.split("-")[0]
        merged_name = f"output_{short_id}_upd.mcap"
        merged_path = output_dir / merged_name

        print(f"\nMerging {input_mcap.name} + {slam_output_name} -> {merged_name}")
        merge_mcaps(
            str(input_mcap),
            str(slam_mcap),
            str(merged_path),
            slam_tf_schema,
            slam_health_schema,
        )


if __name__ == "__main__":
    main()
