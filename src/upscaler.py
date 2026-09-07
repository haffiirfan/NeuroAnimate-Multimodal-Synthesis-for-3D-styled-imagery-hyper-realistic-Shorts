import os
import sys
import json
import glob
import shutil
import time
import threading
import subprocess
import torch
from .compositor import ffprobe_info

ESRGAN_WORKER_CODE = '''
import sys, os, json, cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F


class ResidualDenseBlock(nn.Module):
    """Single Residual Dense Block with 5 convolution layers."""
    def __init__(self, nf=64, gc=32):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        return self.conv5(torch.cat((x, x1, x2, x3, x4), 1)) * 0.2 + x


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block (3× RDB with residual scaling)."""
    def __init__(self, nf, gc=32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x


class RRDBNet(nn.Module):
    """
    RRDBNet architecture for Real-ESRGAN 2× super-resolution.
    23 RRDB blocks, pixel-unshuffle input, 2× upscaling via nearest interpolation.
    """
    def __init__(self, ic=3, oc=3, nf=64, nb=23, gc=32, scale=2):
        super().__init__()
        self.scale = scale
        self.conv_first = nn.Conv2d(ic * 4 if scale == 2 else ic, nf, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(nf, gc) for _ in range(nb)])
        self.conv_body = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_hr = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, oc, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, True)

    def forward(self, x):
        f = F.pixel_unshuffle(x, 2) if self.scale == 2 else x
        f = self.conv_first(f)
        f = f + self.conv_body(self.body(f))
        f = self.lrelu(self.conv_up1(F.interpolate(f, scale_factor=2, mode="nearest")))
        f = self.lrelu(self.conv_up2(F.interpolate(f, scale_factor=2, mode="nearest")))
        return self.conv_last(self.lrelu(self.conv_hr(f)))


# ── Worker Entry Point ──
gpu_id = int(sys.argv[1])
fps = json.loads(sys.argv[2])
fo = sys.argv[3]
bs = int(sys.argv[4])
wt = sys.argv[5]

dev = torch.device("cuda:0")
os.makedirs(fo, exist_ok=True)

m = RRDBNet()
ck = torch.load(wt, map_location="cpu")
m.load_state_dict(ck.get("params_ema", ck.get("params", ck)), strict=True)
m.eval().half().to(dev)


def preprocess(paths):
    imgs = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths]
    tensors = [
        (torch.from_numpy(i).float() / 255.0).permute(2, 0, 1).unsqueeze(0)
        for i in imgs
    ]
    return torch.cat(tensors, 0).half().to(dev)


def postprocess(tensor):
    o = tensor.float().clamp(0, 1).cpu().numpy()
    return [
        cv2.cvtColor((o[i].transpose(1, 2, 0) * 255).astype("uint8"), cv2.COLOR_RGB2BGR)
        for i in range(o.shape[0])
    ]


total = len(fps)
torch.cuda.reset_peak_memory_stats(dev)

with torch.no_grad():
    for s in range(0, total, bs):
        bp = fps[s : s + bs]
        out = postprocess(m(preprocess(bp)))
        for p, img in zip(bp, out):
            cv2.imwrite(f"{fo}/{os.path.basename(p)}", img)
        done = min(s + bs, total)
        pk = torch.cuda.max_memory_allocated(dev) / 1e9
        print(f"[GPU {gpu_id}] {done}/{total} | VRAM {pk:.1f}GB", flush=True)

print(
    f"[GPU {gpu_id}] Done. Peak: {torch.cuda.max_memory_allocated(dev) / 1e9:.2f}GB",
    flush=True,
)
'''


def auto_batch_size(final_video, target_gb=12.0, log=print):
    """
    Calculates optimal ESRGAN batch size based on frame resolution
    and available VRAM budget.

    Args:
        final_video (str): Path to the input video.
        target_gb (float): Target VRAM budget per GPU.
        log: Callable for progress output.

    Returns:
        int: Optimal batch size (1–48).
    """
    info = ffprobe_info(final_video)
    W = int(info.get("width", 480))
    H = int(info.get("height", 640))
    pw = int(W * 3 / 8) * 2
    ph = int(H * 3 / 8) * 2
    pixels = pw * ph

    VRAM_PER_PIXEL = 1.69e-6  # Empirical constant
    MODEL_GB = 0.5
    vram_per_frame = max(VRAM_PER_PIXEL * pixels, 0.05)
    batch = min(max(1, int((target_gb - MODEL_GB) / vram_per_frame)), 48)

    log(
        f"   Auto VRAM tune: {W}×{H} → {pw}×{ph} | batch={batch} "
        f"→ ~{MODEL_GB + batch * vram_per_frame:.1f}GB/GPU"
    )
    return batch


def run_esrgan(final_video, esrgan_weights="/kaggle/working/RealESRGAN_x2plus.pth",
               use_dual_gpu=True, work_dir="/kaggle/working", log=print):
    """
    Upscales a video by 1.5× using Real-ESRGAN across one or two GPUs.

    Workflow:
        1. Extract frames at 75% resolution (scale=3/8 × 2 for even dims)
        2. Split frames between GPU 0 and GPU 1
        3. Run ESRGAN workers in parallel subprocesses
        4. Reassemble upscaled frames into H.264 video

    Args:
        final_video (str): Path to the composited video.
        esrgan_weights (str): Path to RealESRGAN_x2plus.pth weights.
        use_dual_gpu (bool): Whether to split across two GPUs.
        work_dir (str): Working directory.
        log: Callable for progress output.

    Returns:
        tuple: (output_video_path: str, elapsed_seconds: float)
    """
    log("🔍 Starting Real-ESRGAN 1.5× upscaling ...")

    FI = os.path.join(work_dir, "frames_in")
    FO = os.path.join(work_dir, "frames_esr")
    EV = os.path.join(work_dir, "enhanced_output.mp4")

    for d in [FI, FO]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d)

    # ── Extract frames at reduced resolution ───
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", final_video,
            "-vf", "scale=trunc(iw*3/8)*2:trunc(ih*3/8)*2",
            os.path.join(FI, "frame_%05d.png"),
        ],
        check=True,
        stderr=subprocess.DEVNULL,
    )

    all_frames = sorted(glob.glob(os.path.join(FI, "frame_*.png")))
    total = len(all_frames)
    batch_size = auto_batch_size(final_video, target_gb=12.0, log=log)

    n_gpus = min(torch.cuda.device_count(), 2) if use_dual_gpu else 1
    log(f"   {total} frames | {n_gpus} GPU(s) | batch={batch_size} | scale=1.5×")

    # ── Write worker script to temp file ──
    worker_path = "/tmp/esrgan_worker.py"
    with open(worker_path, "w") as f:
        f.write(ESRGAN_WORKER_CODE)

    # ── Split frames across GPUs ──
    mid = total // 2
    splits = [all_frames[:mid], all_frames[mid:]] if n_gpus == 2 else [all_frames]
    errors = []

    def worker(gpu_id, frame_list):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        result = subprocess.run(
            [
                "python", worker_path, str(gpu_id),
                json.dumps(frame_list), FO, str(batch_size), esrgan_weights,
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        for line in result.stdout.strip().split("\n"):
            log(f"   {line}")
        if result.returncode != 0:
            errors.append(result.stderr[-600:])

    # ── Launch parallel workers ───
    threads = [
        threading.Thread(target=worker, args=(i, splits[i]))
        for i in range(n_gpus)
    ]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        raise RuntimeError("ESRGAN failed:\n" + errors[0])

    elapsed = time.time() - t0
    log(f"✅ Upscaling done in {elapsed:.1f}s")

    # ── Reassemble into video ───
    all_out = sorted(glob.glob(os.path.join(FO, "frame_*.png")))
    for i, fp in enumerate(all_out):
        os.rename(fp, os.path.join(FO, f"seq_{i:05d}.png"))

    trimmed = os.path.join(work_dir, "driving_trimmed.mp4")
    subprocess.run(
        [
            "ffmpeg", "-y", "-framerate", "25",
            "-i", os.path.join(FO, "seq_%05d.png"),
            "-i", trimmed,
            "-map", "0:v", "-map", "1:a?",
            "-c:v", "libx264", "-crf", "16", "-preset", "slow",
            "-pix_fmt", "yuv420p", "-shortest", EV,
        ],
        check=True,
        stderr=subprocess.DEVNULL,
    )

    log(f"✅ Enhanced → {os.path.getsize(EV) / 1e6:.1f} MB")
    return EV, elapsed
