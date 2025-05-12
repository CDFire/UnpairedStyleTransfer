# Unpaired Style Transfer from 3D Renders to Anime using Temporal-Aware GANs

This project implements a deep learning pipeline that transforms 3D-rendered animation frames into anime-style 2D frames while maintaining **temporal consistency** and **stylistic coherence**. The method uses an **8-channel input representation** (RGB, depth map, edge map, prior blurred frame) and a two-stage training strategy involving **perceptual pretraining** and **CycleGAN-based unpaired adaptation**.

> 📄 For a detailed explanation of the approach and motivations, please see:
> https://github.com/CDFire/UnpairedStyleTransfer/blob/main/Unpaired_Style_Transfer_Overview.pdf

## Sample Output

https://github.com/CDFire/UnpairedStyleTransfer/blob/main/ExampleOutputs/render_output.mp4

---

## Project Structure

| File               | Description                                                                 |
|--------------------|-----------------------------------------------------------------------------|
| `ColorMaps.py`     | Handles generation of RGB maps from 3D renders.                             |
| `DepthMaps.py`     | Extracts and normalizes depth information from render frames.               |
| `EdgeMaps.py`      | Produces edge representations using Canny filters or learned methods.       |
| `FrameExtract.py`  | Splits video sequences into individual frames for preprocessing.            |
| `dataset.py`       | Custom PyTorch dataset class for managing multi-channel input tensors.      |
| `CycleGAN.py`      | Cycle-consistent GAN architecture for unpaired adaptation.                  |
| `RefinedPipeline.py` | End-to-end integration of frame extraction, preprocessing, and stylization. |
| `RefinedVideo.py`  | Recombines stylized frames into a final video output.                       |
| `render_output.mp4`| Example of stylized output (see above).                                     |

---

## Method Overview

### Stage 1: Perceptual Pretraining  
- Uses paired anime frames to train a U-Net generator.
- Losses: Pixel (L1), Perceptual (VGG), and Style (Gram Matrix).
- Inputs: 8-channel tensors capturing RGB, edges, depth, and temporal context.

### Stage 2: CycleGAN Adaptation  
- Uses unpaired 3D and anime frames to fine-tune the pretrained model.
- Losses: Cycle-consistency, Identity, and Adversarial.
- Enhances style realism while preserving structure and temporal smoothness.
