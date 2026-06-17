# NeuroAnimate: Multimodal Synthesis for 3D-styled imagery & hyper-realistic Shorts

This repository contains the official implementation of **NeuroAnimate**, a memory-orchestrated pipeline that transforms textual prompts into fully animated portrait videos.

Due to the heavy hardware requirements of running 60.1 GB of heterogeneous models, the codebase is provided as a Kaggle Notebook (`.ipynb`), which is configured to run on a dual-T4 GPU environment (32 GB total VRAM).

## Overview

Deploying complex, multi-model AI pipelines usually requires high-end, server-grade GPUs. OrchestraGen solves this by implementing **Dynamic Memory Orchestration (DMO)**, a scheduling discipline that allows multiple heavy models to execute sequentially within a memory-constrained address space. 

By utilizing strict VRAM reclamation and heterogeneous CPU-GPU scheduling, the pipeline achieves a 1.88× memory scaling factor, maintaining peak single-GPU occupancy at just 13.8 GB.

### Pipeline Stages
1. **Prompt Enhancement:** Mistral-7B-Instruct
2. **Base Portrait Generation:** SDXL Base + Refiner
3. **Facial Animation:** LivePortrait & InsightFace
4. **Body Motion Synthesis:** Custom retargeting
5. **Super-Resolution:** Real-ESRGAN

## Quick Start

The entire pipeline is self-contained in a single Kaggle notebook. 

1. Create a [Kaggle](https://www.kaggle.com/) account if you don't have one.
2. Upload `NeuroAnimate.ipynb` as a new Notebook in Kaggle.
3. In the right-hand panel, set the **Accelerator** to **GPU T4 x2**.
4. Run the cells sequentially. The notebook will automatically download the required model weights (approx. 60.1 GB) from HuggingFace directly into the Kaggle environment.

*Note: Do not run this on a local machine unless you have at least 32 GB of VRAM and are comfortable manually configuring environment variables for dual-GPU execution.*

