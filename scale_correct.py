#!/usr/bin/env python3
"""
Gravity-align AMB3R trajectory using IMU, then correct metric scale
by comparing IMU-derived velocity against AMB3R trajectory velocity.

Produces comparison plots against ARKit GT.

Usage:
    python scale_correct.py \
        --mcap ../data/eval_data_mp4_output/arkit_video\ 10.mcap \
        --imu ../data/eval_data_mp4_output/imu_10.csv \
        --arkit_poses ../data/eval_data_mp4_output/arkit_poses\ 10.csv \
        --video_num 10
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcap", type=str, required=True)
    parser.add_argument("--imu", type=str, required=True)
    parser.add_argument("--arkit_poses", type=str, required=True)
    parser.add_argument("--video_num", type=int, required=True)
    return parser.parse_args()


def load_mcap_poses(path):
    positions, quats, ts_ns = [], [], []
    with open(path, "rb") as f:
        reader = make_reader(f)
        for _, ch, msg in reader.iter_messages(topics=["/slam/tf"]):
            m = FrameTransforms()
            m.ParseFromString(msg.data)
            tf = m.transforms[0]
            ts_ns.append(tf.timestamp.seconds * 1_000_000_000 + tf.timestamp.nanos)
            positions.append([tf.translation.x, tf.translation.y, tf.translation.z])
            quats.append([tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w])
    return np.array(positions), np.array(quats), np.array(ts_ns)


def load_imu(path):
    accel, ts_us = [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts_us.append(int(row["timestamp_us"]))
            accel.append([float(row["accel_x"]), float(row["accel_y"]), float(row["accel_z"])])
    return np.array(accel) * 9.81, np.array(ts_us)


def load_arkit(path):
    positions, first_ts_us = [], None
    with open(path) as f:
        reader = csv.DictReader(filter(lambda r: not r.startswith("#"), f))
        for row in reader:
            if first_ts_us is None:
                first_ts_us = int(row["timestamp_us"])
            positions.append([float(row["tx"]), float(row["ty"]), float(row["tz"])])
    return np.array(positions), first_ts_us


def find_gravity(positions, quats, amb3r_ts_s, imu_accel, imu_ts_s):
    """Find gravity via rotation-based R_cam_imu search."""
    imu_interp = np.column_stack([
        interp1d(imu_ts_s, imu_accel[:, i], bounds_error=False, fill_value="extrapolate")(amb3r_ts_s)
        for i in range(3)
    ])
    R_world_cam = Rotation.from_quat(quats).as_matrix()

    candidates = [
        ("I", np.eye(3)),
        ("flip_y", np.diag([1, -1, -1])),
        ("rot_x_90", Rotation.from_euler("x", 90, degrees=True).as_matrix()),
        ("rot_x_-90", Rotation.from_euler("x", -90, degrees=True).as_matrix()),
        ("rot_y_90", Rotation.from_euler("y", 90, degrees=True).as_matrix()),
        ("rot_z_90", Rotation.from_euler("z", 90, degrees=True).as_matrix()),
        ("rot_z_-90", Rotation.from_euler("z", -90, degrees=True).as_matrix()),
    ]

    best_cons, best_g, best_R_ci, best_name = 0, None, None, None
    for name, R_ci in candidates:
        g_ests = np.array([R_world_cam[i] @ R_ci @ imu_interp[i] for i in range(len(amb3r_ts_s))])
        g_mean = g_ests.mean(axis=0)
        cons = np.linalg.norm(g_mean) / 9.81
        if cons > best_cons:
            best_cons, best_g, best_R_ci, best_name = cons, g_mean, R_ci, name

    g_norm = best_g / np.linalg.norm(best_g)
    print(f"  Gravity R_cam_imu={best_name}, consistency={best_cons:.3f}, dir={g_norm.round(3)}")
    return g_norm, best_R_ci


def align_to_gravity(positions, quats, g_world):
    """Rotate so gravity points -Y."""
    g = g_world / np.linalg.norm(g_world)
    target = np.array([0, -1, 0])
    v = np.cross(g, target)
    s = np.linalg.norm(v)
    c = np.dot(g, target)
    if s < 1e-6:
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * (1 - c) / (s**2)
    return (R @ positions.T).T, Rotation.from_matrix(R) * Rotation.from_quat(quats), R


def estimate_scale_from_imu_speed(positions, quats, amb3r_ts_s, imu_accel, imu_ts_s, g_world, R_cam_imu):
    """
    Estimate scale by comparing the *variance* of acceleration magnitudes.

    Key insight: the magnitude of (true linear accel) is scale-independent from
    the IMU side, but the trajectory-derived accel scales linearly with position scale.

    accel_traj = d²(s*pos)/dt² = s * d²pos/dt²

    So: s = |accel_imu_linear| / |accel_traj|  (comparing RMS or matched magnitudes)

    This avoids integration entirely.
    """
    R_world_cam = Rotation.from_quat(quats).as_matrix()
    g_vec = g_world * 9.81

    # IMU linear accel in world frame (gravity subtracted)
    imu_interp = np.column_stack([
        interp1d(imu_ts_s, imu_accel[:, i], bounds_error=False, fill_value="extrapolate")(amb3r_ts_s)
        for i in range(3)
    ])
    accel_imu_world = np.array([
        R_world_cam[i] @ R_cam_imu @ imu_interp[i] - g_vec
        for i in range(len(amb3r_ts_s))
    ])

    # Smooth IMU to match trajectory bandwidth (~30Hz, smoothed)
    wl = min(15, len(positions) // 2 * 2 - 1)
    if wl < 5:
        wl = 5
    accel_imu_smooth = np.column_stack([
        savgol_filter(accel_imu_world[:, i], window_length=wl, polyorder=3)
        for i in range(3)
    ])

    # Trajectory accel (from position double-differentiation)
    pos_smooth = np.column_stack([
        savgol_filter(positions[:, i], window_length=wl, polyorder=3)
        for i in range(3)
    ])
    accel_traj = np.gradient(np.gradient(pos_smooth, amb3r_ts_s, axis=0), amb3r_ts_s, axis=0)

    # Compare acceleration magnitudes in high-motion windows
    imu_mag = np.linalg.norm(accel_imu_smooth, axis=1)
    traj_mag = np.linalg.norm(accel_traj, axis=1)

    # Use windows where both signals have significant acceleration
    threshold_imu = np.percentile(imu_mag, 60)
    threshold_traj = np.percentile(traj_mag, 60)
    active = (imu_mag > threshold_imu) & (traj_mag > threshold_traj)

    if active.sum() < 10:
        print("  WARNING: Not enough high-motion frames, using 1.0")
        return 1.0

    # Scale = ratio of RMS accelerations
    rms_imu = np.sqrt(np.mean(imu_mag[active] ** 2))
    rms_traj = np.sqrt(np.mean(traj_mag[active] ** 2))
    scale = rms_imu / rms_traj

    # Also compute per-frame ratios and take median (more robust)
    ratios = imu_mag[active] / traj_mag[active]
    scale_median = np.median(ratios)

    print(f"  Scale (RMS ratio): {scale:.4f}, Scale (median ratio): {scale_median:.4f}")
    print(f"  RMS accel: IMU={rms_imu:.3f} m/s², traj={rms_traj:.3f} m/s²")
    print(f"  Active frames: {active.sum()}/{len(active)}")

    return scale_median


def estimate_scale_from_imu(positions, quats, amb3r_ts_s, imu_accel, imu_ts_s, g_world, R_cam_imu):
    """
    Estimate metric scale by double-integrating IMU acceleration over very short
    windows (0.2s) to get displacement, then comparing with AMB3R displacement.

    Uses zero-velocity-update-style approach: integrate accel twice over tiny windows
    to get displacement, compare magnitude with AMB3R position change.
    Short windows minimize drift.
    """
    R_world_cam = Rotation.from_quat(quats).as_matrix()
    g_vec = g_world * 9.81

    # Fine-grid IMU in world frame
    dt = 0.01
    t_start = max(amb3r_ts_s[0], imu_ts_s[0])
    t_end = min(amb3r_ts_s[-1], imu_ts_s[-1])
    t_fine = np.arange(t_start, t_end, dt)

    imu_fine = np.column_stack([
        interp1d(imu_ts_s, imu_accel[:, i], bounds_error=False, fill_value="extrapolate")(t_fine)
        for i in range(3)
    ])
    cam_idx = np.searchsorted(amb3r_ts_s, t_fine).clip(0, len(amb3r_ts_s) - 1)

    accel_world = np.zeros_like(imu_fine)
    for i in range(len(t_fine)):
        accel_world[i] = R_world_cam[cam_idx[i]] @ R_cam_imu @ imu_fine[i] - g_vec

    # Very short windows (0.2s) — double integrate for displacement
    window_s = 0.2
    window_n = max(int(window_s / dt), 3)
    stride_n = window_n // 2

    scale_ratios = []

    for start in range(0, len(t_fine) - window_n, stride_n):
        end = start + window_n
        t0, t1 = t_fine[start], t_fine[end - 1]
        dur = t1 - t0
        if dur < 0.05:
            continue

        # Double integrate: displacement = ∫∫ accel dt dt
        # First integral: velocity (starting from 0)
        vel = np.cumsum(accel_world[start:end], axis=0) * dt
        # Second integral: displacement
        disp_imu = np.sum(vel, axis=0) * dt
        disp_imu_mag = np.linalg.norm(disp_imu)

        # AMB3R displacement over same window
        idx0 = np.searchsorted(amb3r_ts_s, t0).clip(0, len(amb3r_ts_s) - 1)
        idx1 = np.searchsorted(amb3r_ts_s, t1).clip(0, len(amb3r_ts_s) - 1)
        if idx0 == idx1:
            continue
        disp_amb3r = positions[idx1] - positions[idx0]
        disp_amb3r_mag = np.linalg.norm(disp_amb3r)

        # Only use windows with meaningful motion
        if disp_imu_mag > 0.005 and disp_amb3r_mag > 0.002:
            ratio = disp_imu_mag / disp_amb3r_mag
            if 0.1 < ratio < 10.0:
                scale_ratios.append(ratio)

    if not scale_ratios:
        print("  WARNING: No valid windows for scale estimation, using 1.0")
        return 1.0

    scale_ratios = np.array(scale_ratios)
    # Use trimmed median — remove top/bottom 20%
    lo, hi = np.percentile(scale_ratios, [20, 80])
    trimmed = scale_ratios[(scale_ratios >= lo) & (scale_ratios <= hi)]
    scale = np.median(trimmed) if len(trimmed) > 5 else np.median(scale_ratios)

    print(f"  Scale estimation: median={scale:.4f}, trimmed_mean={trimmed.mean():.4f}, "
          f"std={trimmed.std():.4f}, N={len(trimmed)}/{len(scale_ratios)} windows")
    return scale


def umeyama(x, y, with_scale=True):
    n = len(x)
    mu_x, mu_y = x.mean(0), y.mean(0)
    x_c, y_c = x - mu_x, y - mu_y
    S_xy = y_c.T @ x_c / n
    U, D, Vt = np.linalg.svd(S_xy)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / (np.sum(x_c**2) / n) if with_scale else 1.0
    t = mu_y - s * R @ mu_x
    return s, R, t


def make_comparison_plot(video_num, pos_original, pos_gravity, pos_scaled, gt_pos, output_path):
    """Generate comparison: original vs gravity-aligned vs scale-corrected vs ARKit."""
    N = min(len(pos_original), len(pos_gravity), len(pos_scaled), len(gt_pos))
    pos_original, pos_gravity, pos_scaled, gt_pos = (
        pos_original[:N], pos_gravity[:N], pos_scaled[:N], gt_pos[:N]
    )

    # SE(3) align each to GT
    def align_se3(src, tgt):
        _, R, t = umeyama(src, tgt, with_scale=False)
        return src @ R.T + t

    def align_sim3(src, tgt):
        s, R, t = umeyama(src, tgt, with_scale=True)
        return s * (src @ R.T) + t, s

    orig_se3 = align_se3(pos_original, gt_pos)
    grav_se3 = align_se3(pos_gravity, gt_pos)
    scaled_se3 = align_se3(pos_scaled, gt_pos)
    scaled_sim3, sim3_scale = align_sim3(pos_scaled, gt_pos)

    ate_orig = np.linalg.norm(orig_se3 - gt_pos, axis=1)
    ate_grav = np.linalg.norm(grav_se3 - gt_pos, axis=1)
    ate_scaled = np.linalg.norm(scaled_se3 - gt_pos, axis=1)
    ate_sim3 = np.linalg.norm(scaled_sim3 - gt_pos, axis=1)

    gt_path = np.sum(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1))
    orig_path = np.sum(np.linalg.norm(np.diff(pos_original, axis=0), axis=1))
    scaled_path = np.sum(np.linalg.norm(np.diff(pos_scaled, axis=0), axis=1))

    time_s = np.linspace(0, N / 30.0, N)

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # 1. 3D comparison — scaled SE(3)
    ax = axes[0, 0]
    ax = fig.add_subplot(2, 2, 1, projection="3d")
    ax.plot(gt_pos[:, 0], gt_pos[:, 2], gt_pos[:, 1], "b-", lw=1.5, label="ARKit GT", alpha=0.8)
    ax.plot(scaled_se3[:, 0], scaled_se3[:, 2], scaled_se3[:, 1], "r-", lw=1.5, label="AMB3R (grav+scale, SE3)", alpha=0.8)
    ax.set_xlabel("X"); ax.set_ylabel("Z"); ax.set_zlabel("Y")
    ax.set_title("3D — Gravity + Scale Corrected (SE3)")
    ax.legend(fontsize=9)

    # 2. ATE comparison: all versions
    ax = axes[0, 1]
    rmse = lambda a: np.sqrt(np.mean(a**2))
    ax.plot(time_s, ate_orig * 100, "gray", lw=1, alpha=0.6, label=f"Original SE(3): {rmse(ate_orig)*100:.1f}cm")
    ax.plot(time_s, ate_grav * 100, "orange", lw=1, alpha=0.7, label=f"Gravity-aligned SE(3): {rmse(ate_grav)*100:.1f}cm")
    ax.plot(time_s, ate_scaled * 100, "r-", lw=1.2, label=f"Grav+Scale SE(3): {rmse(ate_scaled)*100:.1f}cm")
    ax.plot(time_s, ate_sim3 * 100, "b-", lw=1, alpha=0.5, label=f"Grav+Scale Sim(3): {rmse(ate_sim3)*100:.1f}cm")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("ATE (cm)")
    ax.set_title("ATE Comparison")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # 3. Top-down comparison
    ax = axes[1, 0]
    ax.plot(gt_pos[:, 0], gt_pos[:, 2], "b-", lw=1.5, label="ARKit GT", alpha=0.8)
    ax.plot(scaled_se3[:, 0], scaled_se3[:, 2], "r-", lw=1.5, label="AMB3R (grav+scale, SE3)", alpha=0.8)
    ax.plot(gt_pos[0, 0], gt_pos[0, 2], "go", ms=10)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)")
    ax.set_title("Top-Down — Gravity + Scale Corrected (SE3)")
    ax.set_aspect("equal"); ax.grid(True, alpha=0.3); ax.legend()

    # 4. Summary bars
    ax = axes[1, 1]
    labels = ["Original\nSE(3)", "Gravity\nSE(3)", "Grav+Scale\nSE(3)", "Grav+Scale\nSim(3)"]
    values = [rmse(ate_orig)*100, rmse(ate_grav)*100, rmse(ate_scaled)*100, rmse(ate_sim3)*100]
    colors = ["gray", "orange", "red", "blue"]
    bars = ax.bar(labels, values, color=colors, alpha=0.7)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.3, f"{val:.1f}cm", ha="center", fontsize=11)
    ax.set_ylabel("ATE RMSE (cm)")
    ax.set_title("Comparison: ATE RMSE")
    ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Video {video_num} — AMB3R with Gravity + IMU Scale Correction\n"
        f"Path: GT {gt_path:.1f}m, Original {orig_path:.1f}m, Scale-corrected {scaled_path:.1f}m | "
        f"Sim(3) residual scale: {sim3_scale:.3f}",
        fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  Saved to {output_path}")

    return {
        "rmse_orig_se3": rmse(ate_orig),
        "rmse_grav_se3": rmse(ate_grav),
        "rmse_scaled_se3": rmse(ate_scaled),
        "rmse_scaled_sim3": rmse(ate_sim3),
        "gt_path": gt_path,
        "orig_path": orig_path,
        "scaled_path": scaled_path,
        "sim3_residual_scale": sim3_scale,
    }


def main():
    args = get_args()
    vid = args.video_num
    out_dir = Path(args.mcap).parent

    print(f"\n=== Video {vid} ===")

    # Load data
    positions, quats, ts_ns = load_mcap_poses(args.mcap)
    imu_accel, imu_ts_us = load_imu(args.imu)
    gt_pos, arkit_first_ts_us = load_arkit(args.arkit_poses)

    # Align timestamps
    amb3r_ts_s = (ts_ns / 1000.0 + arkit_first_ts_us) / 1e6
    imu_ts_s = imu_ts_us / 1e6

    print(f"  AMB3R: {len(positions)} frames, IMU: {len(imu_accel)} samples")

    # Step 1: Find gravity
    print("  Finding gravity...")
    g_world, R_cam_imu = find_gravity(positions, quats, amb3r_ts_s, imu_accel, imu_ts_s)

    # Step 2: Gravity-align
    print("  Aligning to gravity...")
    grav_pos, grav_rots, R_grav = align_to_gravity(positions, quats, g_world)
    grav_quats = grav_rots.as_quat()

    print(f"  Y std: original={positions[:, 1].std():.3f}m, aligned={grav_pos[:, 1].std():.3f}m, GT={gt_pos[:, 1].std():.3f}m")

    # Step 3: Estimate scale from IMU (acceleration magnitude comparison)
    print("  Estimating scale from IMU...")
    scale = estimate_scale_from_imu_speed(
        grav_pos, grav_quats, amb3r_ts_s, imu_accel, imu_ts_s,
        np.array([0, -1, 0]),  # gravity in aligned frame
        R_cam_imu,
    )
    print(f"  IMU scale factor: {scale:.4f}")

    # Step 4: Apply scale
    scaled_pos = grav_pos * scale

    orig_path = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
    scaled_path = np.sum(np.linalg.norm(np.diff(scaled_pos, axis=0), axis=1))
    gt_path = np.sum(np.linalg.norm(np.diff(gt_pos, axis=0), axis=1))
    print(f"  Path: original={orig_path:.1f}m, scaled={scaled_path:.1f}m, GT={gt_path:.1f}m")

    # Step 5: Compare and plot
    output_png = out_dir / f"scale_corrected_v{vid}.png"
    metrics = make_comparison_plot(vid, positions, grav_pos, scaled_pos, gt_pos, str(output_png))

    print(f"\n  Results for video {vid}:")
    print(f"    Original SE(3) RMSE:     {metrics['rmse_orig_se3']*100:.1f} cm")
    print(f"    Gravity SE(3) RMSE:      {metrics['rmse_grav_se3']*100:.1f} cm")
    print(f"    Grav+Scale SE(3) RMSE:   {metrics['rmse_scaled_se3']*100:.1f} cm")
    print(f"    Grav+Scale Sim(3) RMSE:  {metrics['rmse_scaled_sim3']*100:.1f} cm")


if __name__ == "__main__":
    main()
