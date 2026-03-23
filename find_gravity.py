#!/usr/bin/env python3
"""
Find gravity direction in AMB3R world frame by correlating
AMB3R trajectory accelerations with IMU accelerometer data.

Then rotate AMB3R poses so gravity points -Y (Y-up world).

Usage:
    python find_gravity.py \
        --mcap ../data/eval_data_mp4_output/arkit_video\ 10.mcap \
        --imu ../data/eval_data_mp4_output/imu_10.csv \
        --arkit_poses ../data/eval_data_mp4_output/arkit_poses\ 10.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

SCHEMA_PY_DIR = Path(__file__).resolve().parent.parent / "microagi-schemas" / "python"
sys.path.insert(0, str(SCHEMA_PY_DIR))

from foxglove.FrameTransforms_pb2 import FrameTransforms
from mcap.reader import make_reader


def get_args():
    parser = argparse.ArgumentParser(description="Find gravity from IMU + AMB3R correlation")
    parser.add_argument("--mcap", type=str, required=True, help="AMB3R output MCAP")
    parser.add_argument("--imu", type=str, required=True, help="IMU CSV file")
    parser.add_argument("--arkit_poses", type=str, required=True, help="ARKit poses CSV (for timestamp offset)")
    parser.add_argument("--output_png", type=str, default=None, help="Output visualization PNG")
    return parser.parse_args()


def load_amb3r_poses(mcap_path):
    """Load poses and timestamps from AMB3R MCAP."""
    positions, quats, ts_ns = [], [], []
    with open(mcap_path, "rb") as f:
        reader = make_reader(f)
        for _, channel, message in reader.iter_messages(topics=["/slam/tf"]):
            msg = FrameTransforms()
            msg.ParseFromString(message.data)
            tf = msg.transforms[0]
            ts_ns.append(tf.timestamp.seconds * 1_000_000_000 + tf.timestamp.nanos)
            positions.append([tf.translation.x, tf.translation.y, tf.translation.z])
            quats.append([tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w])
    return np.array(positions), np.array(quats), np.array(ts_ns)


def load_imu(imu_path):
    """Load IMU data from CSV. Returns accel in m/s², timestamps in seconds."""
    accel, gyro, ts_us = [], [], []
    with open(imu_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_us.append(int(row["timestamp_us"]))
            accel.append([float(row["accel_x"]), float(row["accel_y"]), float(row["accel_z"])])
            gyro.append([float(row["gyro_x"]), float(row["gyro_y"]), float(row["gyro_z"])])
    accel = np.array(accel) * 9.81  # g-units → m/s²
    return accel, np.array(gyro), np.array(ts_us)


def load_arkit_first_timestamp(arkit_path):
    """Get the first ARKit timestamp in microseconds (epoch time)."""
    with open(arkit_path) as f:
        reader = csv.DictReader(filter(lambda r: not r.startswith("#"), f))
        row = next(reader)
        return int(row["timestamp_us"])


def get_gravity_from_rotations(amb3r_positions, amb3r_quats, amb3r_ts_s, imu_accel, imu_ts_s):
    """
    Alternative approach: use AMB3R camera rotations to rotate IMU accel to world frame.

    AMB3R gives R_world_cam for each frame. If we knew R_cam_imu we could compute
    g_world = R_world_cam @ R_cam_imu @ a_imu_stationary.

    But we don't know R_cam_imu. However, we can estimate it:
    - AMB3R gives angular velocity in world frame (from rotation differences)
    - IMU gyro gives angular velocity in IMU frame
    - Correlating these gives R_cam_imu

    Simpler: just use the fact that g_imu is constant in the IMU frame.
    For each AMB3R frame i: g_world = R_world_cam[i] @ R_cam_imu @ g_imu
    If R_cam_imu is close to identity (phone camera and IMU roughly aligned),
    then g_world ≈ R_world_cam[i] @ g_imu for all i.
    The correct R_cam_imu makes all these g_world estimates agree.
    """
    # Interpolate IMU to AMB3R timestamps
    imu_interp = np.column_stack([
        interp1d(imu_ts_s, imu_accel[:, i], bounds_error=False, fill_value="extrapolate")(amb3r_ts_s)
        for i in range(3)
    ])

    # AMB3R rotation matrices (world ← camera, c2w)
    R_world_cam = Rotation.from_quat(amb3r_quats).as_matrix()  # [N, 3, 3]

    # For each frame, rotate IMU accel to world frame assuming R_cam_imu ≈ I
    # g_world_estimates[i] = R_world_cam[i] @ imu_interp[i]
    g_world_estimates = np.array([
        R_world_cam[i] @ imu_interp[i] for i in range(len(amb3r_ts_s))
    ])  # [N, 3]

    # If R_cam_imu = I, all estimates should agree and point in gravity direction
    g_world_mean = g_world_estimates.mean(axis=0)
    g_world_std = g_world_estimates.std(axis=0)

    print(f"  R_cam_imu=I assumption:")
    print(f"    g_world mean: {g_world_mean} (mag: {np.linalg.norm(g_world_mean):.3f})")
    print(f"    g_world std:  {g_world_std}")
    consistency = np.linalg.norm(g_world_mean) / 9.81
    print(f"    Consistency:  {consistency:.3f} (1.0 = perfect)")

    # Try different R_cam_imu candidates (90° rotations)
    # Common phone orientations: camera and IMU axes may be permuted/flipped
    candidates = [
        ("I", np.eye(3)),
        ("flip_y", np.diag([1, -1, -1])),
        ("flip_yz", np.diag([1, -1, 1])),
        ("rot_x_90", Rotation.from_euler('x', 90, degrees=True).as_matrix()),
        ("rot_x_-90", Rotation.from_euler('x', -90, degrees=True).as_matrix()),
        ("rot_y_90", Rotation.from_euler('y', 90, degrees=True).as_matrix()),
        ("rot_z_90", Rotation.from_euler('z', 90, degrees=True).as_matrix()),
        ("rot_z_-90", Rotation.from_euler('z', -90, degrees=True).as_matrix()),
    ]

    best_consistency = 0
    best_g = None
    best_name = None

    for name, R_ci in candidates:
        g_ests = np.array([
            R_world_cam[i] @ R_ci @ imu_interp[i] for i in range(len(amb3r_ts_s))
        ])
        g_mean = g_ests.mean(axis=0)
        cons = np.linalg.norm(g_mean) / 9.81
        if cons > best_consistency:
            best_consistency = cons
            best_g = g_mean
            best_name = name

    print(f"\n  Best R_cam_imu candidate: {best_name} (consistency: {best_consistency:.3f})")
    print(f"    g_world: {best_g} (mag: {np.linalg.norm(best_g):.3f})")
    g_normalized = best_g / np.linalg.norm(best_g)
    print(f"    direction: {g_normalized}")

    return g_normalized


def get_gravity_direction(amb3r_positions, amb3r_ts_s, imu_accel, imu_ts_s):
    """
    Find gravity direction in AMB3R world frame.

    The IMU accelerometer measures specific force:
        a_imu = R_imu_world @ (accel_world - g_world)

    Where accel_world comes from double-differentiating AMB3R positions.
    Rearranging: R_world_imu @ a_imu = accel_world - g_world

    Taking the mean over time (mean accel_world ≈ 0 for bounded motion):
        g_world ≈ -R_world_imu @ mean(a_imu)

    And for the dynamic part (after removing means):
        R_world_imu @ (a_imu - mean(a_imu)) ≈ (accel_world - mean(accel_world))

    So we: (1) find R from mean-centered signals, (2) get gravity from R @ mean(a_imu).
    """
    # Step 1: trajectory acceleration from AMB3R positions
    # Use heavy smoothing — double differentiation amplifies noise
    wl = min(31, len(amb3r_positions) // 2 * 2 - 1)
    if wl < 5:
        wl = 5
    pos_smooth = np.column_stack([
        savgol_filter(amb3r_positions[:, i], window_length=wl, polyorder=3)
        for i in range(3)
    ])
    accel_world = np.gradient(np.gradient(pos_smooth, amb3r_ts_s, axis=0), amb3r_ts_s, axis=0)

    # Step 2: interpolate IMU to AMB3R timestamps
    imu_interp = np.column_stack([
        interp1d(imu_ts_s, imu_accel[:, i], bounds_error=False, fill_value="extrapolate")(amb3r_ts_s)
        for i in range(3)
    ])

    # Also smooth IMU to match trajectory bandwidth
    imu_smooth = np.column_stack([
        savgol_filter(imu_interp[:, i], window_length=wl, polyorder=3)
        for i in range(3)
    ])

    # Step 3: find R_world_imu from mean-centered dynamic signals
    accel_mean = accel_world.mean(axis=0)
    imu_mean = imu_smooth.mean(axis=0)
    accel_dyn = accel_world - accel_mean
    imu_dyn = imu_smooth - imu_mean

    # Use frames with significant motion for better SVD conditioning
    motion = np.linalg.norm(accel_dyn, axis=1)
    threshold = np.percentile(motion, 50)
    active = motion > threshold

    A = accel_dyn[active]
    B = imu_dyn[active]

    H = B.T @ A
    U, S, Vt = np.linalg.svd(H)
    d = np.linalg.det(Vt.T @ U.T)
    R_world_imu = Vt.T @ np.diag([1, 1, d]) @ U.T

    # Step 4: gravity = -R_world_imu @ mean(a_imu)
    # Because mean(accel_world) ≈ 0 and a_imu = R_imu_world @ (accel - g)
    # So mean(a_imu) ≈ R_imu_world @ (-g) → g = -R_world_imu @ mean(a_imu)
    g_world = -R_world_imu @ imu_mean
    g_mag = np.linalg.norm(g_world)
    g_world_normalized = g_world / g_mag

    # Also try: gravity from stationary IMU window for comparison
    window = 50
    variances = [
        np.var(imu_accel[i : i + window], axis=0).sum()
        for i in range(len(imu_accel) - window)
    ]
    best = np.argmin(variances)
    g_imu_stationary = np.mean(imu_accel[best : best + window], axis=0)
    g_world_stationary = -R_world_imu @ g_imu_stationary
    g_world_stationary_norm = g_world_stationary / np.linalg.norm(g_world_stationary)

    print(f"IMU mean accel:     {imu_mean} (mag: {np.linalg.norm(imu_mean):.3f} m/s²)")
    print(f"Gravity (from mean): {g_world} (mag: {g_mag:.3f} m/s²)")
    print(f"Gravity direction:   {g_world_normalized}")
    print(f"Gravity (stationary): {g_world_stationary_norm} (from lowest-variance window)")
    print(f"SVD singular values: {S}")

    # Use stationary-derived gravity if mean-derived seems off
    # (mean accel_world ≈ 0 assumption can fail with drift)
    if abs(g_mag - 9.81) > 2.0:
        print("WARNING: mean-derived gravity magnitude off, using stationary window instead")
        g_world_normalized = g_world_stationary_norm

    return g_world_normalized, R_world_imu, accel_world, imu_interp


def align_to_gravity(positions, quats, g_world):
    """Rotate AMB3R world so gravity points -Y (Y-up convention)."""
    g = g_world / np.linalg.norm(g_world)
    target = np.array([0, -1, 0])  # gravity = -Y

    v = np.cross(g, target)
    s = np.linalg.norm(v)
    c = np.dot(g, target)

    if s < 1e-6:
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1 - c) / (s**2)

    # Rotate positions
    aligned_positions = (R @ positions.T).T

    # Rotate orientations
    R_rot = Rotation.from_matrix(R)
    original_rots = Rotation.from_quat(quats)
    aligned_rots = R_rot * original_rots
    aligned_quats = aligned_rots.as_quat()

    return aligned_positions, aligned_quats, R


def load_arkit_gt(arkit_path):
    """Load ARKit GT positions."""
    positions = []
    with open(arkit_path) as f:
        reader = csv.DictReader(filter(lambda r: not r.startswith("#"), f))
        for row in reader:
            positions.append([float(row["tx"]), float(row["ty"]), float(row["tz"])])
    return np.array(positions)


def visualize(
    original_pos, aligned_pos, gt_pos, g_world, time_s, output_path
):
    """Generate comparison visualization."""
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))

    # 1. Original trajectory 3D
    ax = axes[0, 0]
    ax.plot(original_pos[:, 0], original_pos[:, 2], "r-", lw=1.5, label="AMB3R (original)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Original AMB3R (top-down)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 2. Gravity-aligned trajectory top-down
    ax = axes[0, 1]
    ax.plot(aligned_pos[:, 0], aligned_pos[:, 2], "r-", lw=1.5, label="AMB3R (gravity-aligned)")
    ax.plot(gt_pos[:, 0], gt_pos[:, 2], "b-", lw=1.5, alpha=0.5, label="ARKit GT")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Gravity-Aligned AMB3R vs ARKit (top-down)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 3. Gravity vector visualization
    ax = axes[0, 2]
    g = g_world / np.linalg.norm(g_world)
    ax.quiver(0, 0, g[0], g[1], angles="xy", scale_units="xy", scale=1, color="red", label=f"g_world = [{g[0]:.2f}, {g[1]:.2f}, {g[2]:.2f}]")
    ax.quiver(0, 0, 0, -1, angles="xy", scale_units="xy", scale=1, color="blue", alpha=0.5, label="Target -Y")
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Gravity Direction in AMB3R World (XY plane)")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # 4. Height (Y) over time — original
    ax = axes[1, 0]
    ax.plot(time_s, original_pos[:, 1], "r-", lw=1, label="AMB3R Y (original)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Height (Y) — Original AMB3R")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 5. Height (Y) over time — gravity-aligned vs ARKit
    N = min(len(aligned_pos), len(gt_pos))
    ax = axes[1, 1]
    # Offset aligned to match ARKit mean Y for visual comparison
    offset = gt_pos[:N, 1].mean() - aligned_pos[:N, 1].mean()
    ax.plot(time_s[:N], aligned_pos[:N, 1] + offset, "r-", lw=1, label="AMB3R Y (aligned + offset)")
    ax.plot(time_s[:N], gt_pos[:N, 1], "b-", lw=1, alpha=0.7, label="ARKit GT Y")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Y (m)")
    ax.set_title("Height (Y) — Aligned AMB3R vs ARKit GT")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 6. Y variance comparison
    ax = axes[1, 2]
    y_std_orig = original_pos[:, 1].std()
    y_std_aligned = aligned_pos[:, 1].std()
    y_std_gt = gt_pos[:, 1].std()
    bars = ax.bar(
        ["Original\nAMB3R Y", "Gravity-Aligned\nAMB3R Y", "ARKit GT\nY"],
        [y_std_orig, y_std_aligned, y_std_gt],
        color=["red", "orange", "blue"],
        alpha=0.7,
    )
    ax.set_ylabel("Std deviation (m)")
    ax.set_title("Height Variation (std) — Lower = Flatter Floor")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, [y_std_orig, y_std_aligned, y_std_gt]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}m", ha="center", fontsize=11)

    fig.suptitle(
        f"Gravity Alignment — Video 10\n"
        f"Gravity in world: [{g[0]:.3f}, {g[1]:.3f}, {g[2]:.3f}]",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    print(f"Saved visualization to {output_path}")


def main():
    args = get_args()

    # Load data
    print("Loading AMB3R poses...")
    positions, quats, ts_ns = load_amb3r_poses(args.mcap)

    print("Loading IMU data...")
    imu_accel, imu_gyro, imu_ts_us = load_imu(args.imu)

    print("Loading ARKit timestamp offset...")
    arkit_first_ts_us = load_arkit_first_timestamp(args.arkit_poses)

    # Align timestamps: AMB3R video PTS → epoch microseconds → seconds
    amb3r_ts_us = ts_ns / 1000.0 + arkit_first_ts_us
    amb3r_ts_s = amb3r_ts_us / 1e6
    imu_ts_s = imu_ts_us / 1e6

    print(f"AMB3R: {len(positions)} poses, {amb3r_ts_s[-1]-amb3r_ts_s[0]:.1f}s")
    print(f"IMU:   {len(imu_accel)} samples, {imu_ts_s[-1]-imu_ts_s[0]:.1f}s")
    print(f"Time overlap: {max(amb3r_ts_s[0], imu_ts_s[0]):.3f} to {min(amb3r_ts_s[-1], imu_ts_s[-1]):.3f}s")

    # Find gravity — try rotation-based approach first
    print("\nFinding gravity direction (rotation-based)...")
    g_world_rot = get_gravity_from_rotations(
        positions, quats, amb3r_ts_s, imu_accel, imu_ts_s
    )

    print("\nFinding gravity direction (accel correlation)...")
    g_world_corr, R_world_imu, accel_world, imu_interp = get_gravity_direction(
        positions, amb3r_ts_s, imu_accel, imu_ts_s
    )

    # Use rotation-based result (more reliable)
    g_world = g_world_rot
    print(f"\nUsing rotation-based gravity: {g_world}")

    # Align poses
    print("\nAligning to gravity...")
    aligned_pos, aligned_quats, R_align = align_to_gravity(positions, quats, g_world)

    # Check: after alignment, Y should be roughly constant for flat-ground walking
    print(f"\nOriginal Y range:  [{positions[:, 1].min():.3f}, {positions[:, 1].max():.3f}] (std: {positions[:, 1].std():.3f}m)")
    print(f"Aligned Y range:   [{aligned_pos[:, 1].min():.3f}, {aligned_pos[:, 1].max():.3f}] (std: {aligned_pos[:, 1].std():.3f}m)")

    # Load ARKit GT for comparison
    gt_pos = load_arkit_gt(args.arkit_poses)
    print(f"ARKit GT Y range:  [{gt_pos[:, 1].min():.3f}, {gt_pos[:, 1].max():.3f}] (std: {gt_pos[:, 1].std():.3f}m)")

    # Visualize
    output_png = args.output_png or str(Path(args.mcap).parent / "gravity_alignment.png")
    time_s = (amb3r_ts_s - amb3r_ts_s[0])
    visualize(positions, aligned_pos, gt_pos, g_world, time_s, output_png)

    # Save aligned poses
    output_npz = str(Path(args.mcap).parent / "aligned_poses_video10.npz")
    np.savez(
        output_npz,
        positions=aligned_pos,
        quats=aligned_quats,
        timestamps_s=amb3r_ts_s,
        gravity_world=g_world,
        R_align=R_align,
    )
    print(f"Saved aligned poses to {output_npz}")


if __name__ == "__main__":
    main()
