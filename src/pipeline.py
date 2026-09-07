import os
import glob
import time
import threading
import traceback

from .memory_orchestrator import clear_gpu, clear_gpu_memory
from .prompt_enhancer import PromptEnhancer
from .image_generator import ImageGenerator
from .liveportrait_runner import normalize_video, run_liveportrait
from .body_animator import run_body_animation
from .compositor import composite_videos
from .upscaler import run_esrgan


class NeuroAnimatePipeline:
    """
    End-to-end pipeline: Text → Enhanced Prompt → Portrait Image → Animated Video.

    This class wraps all six models and manages the DMO lifecycle. Each stage
    loads its model into VRAM, runs inference, and explicitly tears down before
    the next stage begins.

    Attributes:
        enhancer (PromptEnhancer): Mistral 7B prompt enhancement module.
        generator (ImageGenerator): SDXL Base + Refiner image generation module.
        timings (dict): Per-stage timing measurements.
        enhanced_prompt (str): Most recently enhanced prompt.
        images (list): Most recently generated PIL images.
    """

    def __init__(self):
        self.enhancer = PromptEnhancer()
        self.generator = ImageGenerator()
        self.timings = {}
        self.enhanced_prompt = None
        self.images = None

    def enhance(self, user_prompt, mode="Photorealism", bypass=False):
        """
        Stage 1: Enhance user prompt using Mistral 7B on GPU 1.

        Args:
            user_prompt (str): Raw text description.
            mode (str): Creative mode — 'Photorealism', 'Graphic Designer', or 'Gaming'.
            bypass (bool): If True, skip enhancement and use raw prompt.

        Returns:
            str: The enhanced (or original) prompt.
        """
        enhanced, elapsed = self.enhancer.enhance(user_prompt, mode, bypass)
        self.enhanced_prompt = enhanced
        if elapsed > 0:
            self.timings["mistral"] = elapsed
        return enhanced

    def generate_images(self, prompt, num_images=2, seed=42):
        """
        Stage 2: Generate portrait images using SDXL Base (GPU 0) + Refiner (GPU 1).

        Args:
            prompt (str): The enhanced text prompt.
            num_images (int): Number of images (1–4).
            seed (int): Random seed.

        Returns:
            list[PIL.Image]: Generated and refined images.
        """
        images, timings = self.generator.generate(prompt, num_images, seed)
        self.images = images
        self.timings.update(timings)

        if len(images) > 1:
            grid = ImageGenerator.create_grid(images)
            return [grid] + images
        return images

    def generate_video(self, driving_video_path=None,
                       motion_multiplier=0.55, enable_upscale=True,
                       use_dual_gpu=True, log=print):
        """
        Stages 3–6: Animate the most recent portrait into a video.

        Workflow:
            normalize → LivePortrait (GPU 0) ∥ body animation (CPU)
            → VRAM clear → composite → Real-ESRGAN upscale (dual GPU)

        Args:
            driving_video_path (str): Path to the driving video. If None,
                falls back to the default Kaggle input path.
            motion_multiplier (float): LivePortrait motion intensity.
            enable_upscale (bool): Whether to run Real-ESRGAN.
            use_dual_gpu (bool): Whether ESRGAN uses both GPUs.
            log: Callable for progress output.

        Returns:
            tuple: (video_path: str or None, status_message: str)
        """
        try:
            # ── Resolve source portrait ──────────────────────────────────
            portrait_files = glob.glob("/kaggle/working/refined_images/*.png")
            if not portrait_files:
                return None, "❌ No generated images found. Generate images first."
            source_image = max(portrait_files, key=os.path.getmtime)

            # ── Resolve driving video ──
            if driving_video_path and os.path.exists(driving_video_path):
                driving_video = driving_video_path
            else:
                driving_video = "/kaggle/input/sharukh/shahrukh.mp4.mp4"
                if not os.path.exists(driving_video):
                    return None, "❌ No driving video found. Upload a driving video."

            clear_gpu()

            # ── Stage 3: Normalise inputs ──
            log("━" * 52)
            log("🚀 NeuroAnimate Pipeline Starting  [2D Body Mode]")
            log("━" * 52)

            source, trimmed = normalize_video(source_image, driving_video, log=log)

            # ── Stage 4: Parallel face + body animation ──
            log("⚡ Stage 4: LivePortrait (GPU 0) ∥ Body Animation (CPU) — parallel ...")
            lp_result = [None]
            body_result = [None]
            errors = []

            def _run_liveportrait():
                try:
                    lp_result[0] = run_liveportrait(
                        source, trimmed, motion_multiplier, log=log
                    )
                except Exception as e:
                    errors.append(f"LivePortrait: {e}\n{traceback.format_exc()}")

            def _run_body():
                try:
                    body_result[0] = run_body_animation(source, trimmed, log=log)
                except Exception as e:
                    errors.append(f"Body: {e}\n{traceback.format_exc()}")

            t_lp = threading.Thread(target=_run_liveportrait)
            t_bd = threading.Thread(target=_run_body)

            t0 = time.time()
            t_lp.start()
            t_bd.start()
            t_lp.join()
            t_bd.join()

            self.timings["stage4_liveportrait_body"] = time.time() - t0
            log(f"⏱ Stage 4 parallel: {self.timings['stage4_liveportrait_body']:.1f}s")

            if errors:
                raise RuntimeError("\n".join(errors))
            log("✅ Stage 4 complete")

            # ── DMO: Clear VRAM before compositing ───
            clear_gpu_memory(log)

            # ── Stage 5: Composite face + body ───
            final = composite_videos(
                lp_result[0], body_result[0], source, trimmed, log=log
            )

            # ── Stage 6: Super-resolution ───
            if enable_upscale:
                final, esrgan_time = run_esrgan(
                    final, use_dual_gpu=use_dual_gpu, log=log
                )
                self.timings["esrgan"] = esrgan_time

            # ── Timing report ───
            log("━" * 52)
            log("📊 TIMING REPORT")
            log("━" * 52)
            for key, value in self.timings.items():
                log(f"  {key:<25} {value:.1f}s")
            total = sum(self.timings.values())
            log(f"  {'TOTAL':<25} {total:.1f}s")
            log("━" * 52)

            if final and os.path.exists(final):
                size_mb = os.path.getsize(final) / 1e6
                return final, f"✅ Video generated! ({size_mb:.1f} MB)"
            else:
                return None, "❌ Pipeline returned no output file."

        except Exception as e:
            clear_gpu()
            return None, f"❌ Video generation failed:\n{traceback.format_exc()}"
