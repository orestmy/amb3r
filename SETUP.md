# AMB3R SLAM Pipeline — Setup & Run Guide

## Environment

Tested on: **A100-SXM4-80GB**, Ubuntu 22.04, Python 3.10, CUDA 12.8, PyTorch 2.7.1

The original AMB3R was built for PyTorch 2.5 + CUDA 11.8. Running on newer PyTorch/CUDA required several fixes documented below.

## Installation

```bash
# Core deps
pip install torch torchvision torchaudio  # already present if using NGC/deep learning VM
pip install mcap mcap-protobuf-support protobuf scipy pillow av opencv-python
pip install omegaconf einops timm==0.6.7 evo easydict addict huggingface-hub

# torch-scatter — install from source for your PyTorch version
pip install torch-scatter

# pytorch3d — must compile with CUDA support
FORCE_CUDA=1 pip install "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8" --no-build-isolation

# flash-attn — compile from source
pip install flash-attn --no-build-isolation

# spconv — use cu124 for CUDA 12.x (cu120 crashes with SIGFPE on CUDA 12.8)
pip install spconv-cu124

# utils3d — broken pyproject.toml, install manually
cd /tmp && git clone https://github.com/EasternJournalist/utils3d.git
cd utils3d && git checkout c5daf6f6
cp -r utils3d ~/.local/lib/python3.10/site-packages/utils3d
```

## Key Compatibility Fixes

### spconv SIGFPE crash
`spconv-cu118` and `spconv-cu120` crash with `Floating point exception` on CUDA 12.8. Solution: use `spconv-cu124` (2.3.8) which is forward-compatible.

### PTV3 flash_attn crash
PointTransformerV3 uses `flash_attn.flash_attn_varlen_qkvpacked_func` directly, which can SIGFPE depending on the flash-attn build. Fix in `amb3r/backend.py`:
```python
# Before:
self.point_transformer = PointTransformerV3()
# After:
self.point_transformer = PointTransformerV3(backbone_cfg={'enable_flash': False})
```
This uses the standard matmul attention fallback (lines 463-479 of `thirdparty/ptv3/point_transformer.py`), which is mathematically identical but uses PyTorch native ops.

### pytorch3d GPU support
If pytorch3d was installed without CUDA (e.g., from a CPU-only wheel), `knn_points` fails with `RuntimeError: Not compiled with GPU support`. Fix: reinstall with `FORCE_CUDA=1`.

## Protobuf Schemas

Clone the MicroAGI schemas repo (needed for MCAP reading/writing):
```bash
git clone https://github.com/MicroAGI-Labs/microagi-schemas.git /path/to/micro-agi/microagi-schemas
```

## Model Checkpoint

Download `amb3r.pt` (4.1GB) from [Google Drive](https://drive.google.com/file/d/14x0WW2rUE_he2hUEouP6ywSRnlJDeLel):
```bash
pip install gdown
gdown 14x0WW2rUE_he2hUEouP6ywSRnlJDeLel -O checkpoints/amb3r.pt
```

## Running

### SLAM on MP4 video
```bash
python run_slam_mp4.py \
    --input_dir ../data/eval_data_mp4 \
    --output_dir ../data/eval_data_mp4_output \
    --ckpt_path ./checkpoints/amb3r.pt \
    --max_frames 700          # optional: limit frames
    --frame_step 2            # optional: use every Nth frame
```

### SLAM on MCAP recordings
```bash
python run_slam_mcap.py \
    --input_dir ../data/eval_data_half \
    --output_dir ../data/eval_data_half_output \
    --ckpt_path ./checkpoints/amb3r.pt
```

### Merge input + SLAM output into single MCAP
```bash
python merge_mcap.py \
    --input_dir ../data/eval_data_half \
    --output_dir ../data/eval_data_half_output \
    --example ../data/eval_data_half_output/output_example.mcap
```

### Gravity alignment from IMU
```bash
python find_gravity.py \
    --mcap ../data/eval_data_mp4_output/arkit_video\ 10.mcap \
    --imu ../data/eval_data_mp4_output/imu_10.csv \
    --arkit_poses ../data/eval_data_mp4_output/arkit_poses\ 10.csv
```

### Evaluate trajectory
```bash
cd ../
PYTHONPATH="microagi-schemas/python:$PYTHONPATH" python evaluation/evaluate_trajectory.py \
    ../data/eval_data_mp4_output/arkit_video\ 10.mcap
```

## Performance

Benchmarked on A100-80GB, 1920x1440 input, 518x392 model resolution:

| Frames | Wall time | Notes |
|--------|-----------|-------|
| 100    | ~57s      | includes ~15s model load + torch.compile warmup |
| 700    | ~4 min    | ~30fps input, 23s of video |
| 1076   | ~6 min    | 35s of video |
| 2451   | ~14 min   | 82s of video |

Speed optimizations applied:
- Direct video decode → cv2 resize → tensor (no PNG round-trip, saves ~3-8 min)
- `torch.compile(model.front_end.model.aggregator, mode="reduce-overhead")` (~35% faster inference)

## Output Format

Output MCAPs contain:
- `/slam/tf` — `foxglove.FrameTransforms` (position xyz + quaternion wxyz per frame)
- `/slam/health` — `microagi.Health` (valid=true for all frames)

Compatible with `evaluation/evaluate_trajectory.py`.
