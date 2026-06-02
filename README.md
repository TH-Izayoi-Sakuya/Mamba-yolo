# GhostConv-Mamba-YOLO

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20507899.svg)](https://doi.org/10.5281/zenodo.20507899)

##  Environment Setup
The code is built on top of the `ultralytics` framework with customized State-Space Models (Mamba).

**Dependencies:**
- Python >= 3.8
- PyTorch >= 1.13.0
- [Ultralytics](https://github.com/ultralytics/ultralytics) >= 8.0.0
- `mamba_ssm` and `causal_conv1d` (for MambaBlock2D)
- Additional evaluation tools: `thop`, `scikit-learn`, `opencv-python`

**Installation:**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics opencv-python scikit-learn thop
pip install causal-conv1d>=1.0.0
pip install mamba-ssm
```

##  Repository Structure
- `mamba_yolo_scca_ghost_seg.yaml`: The core network architecture combining GhostConv, C2f, MambaBlock2D, and SPPF.
- `custom_modules.py`: Implementation of our spatial-channel modules (GhostConv, SCCA, ADDR, and MambaBlock2D).
- `bdc_trainer.py`: Custom segmentation trainer that integrates the Boundary-Driven Constraint (BDC) during the training phase.
- `train_seg.py`: The main entry script for model training.
- `eval_dice_miou.py`: Comprehensive evaluation script for computing standard metrics (Dice, mIoU) and boundary-specific metrics (HD95, ASSD, Boundary-F), as well as hardware efficiency (Params, FLOPs, FPS).
- `pred.py`: Script for qualitative visualization and inference.


##  Dataset Preparation
We evaluate our model on polyp segmentation datasets (e.g., Kvasir-SEG, CVC-ClinicDB, PolypDB). 
Since YOLOv8 requires polygon-based label formats (TXT) rather than binary masks, we provide a robust conversion script.

1. Place your raw images and binary masks in a structured directory.
2. Run the preparation script to generate YOLO-format `.txt` labels:
```bash
python prepare_masks_to_yoloseg.py
```
3. Update the data paths in `data.yaml`.

##  Training
To train the GhostConv-Mamba-YOLO model from scratch, execute:
``bash
python train_seg.py
```
*Note: The `train_seg.py` automatically registers the custom modules and overrides the default Ultralytics loss function with our `SegTrainerWithBDC`.*

##  Evaluation
To reproduce the quantitative results reported in the manuscript (including Dice, miOU, HD95, ASSD, Params, FLOPs, and FPS), run the comprehensive evaluation script:
```bash
python eval_dice_miou.py
```
This script will output the global confusion metrics and boundary metrics, saving the per-image results to `eval_out/per_image_metrics.csv`.

##  Inference and Visualization
To generate segmentation masks and boundary visualizations on unseen images:
```bash
python pred.py
```
Ensure you have updated `MODEL_PATH`, `IMAGE_DIR`, and `OUT_DIR` within the script to match your local environment before running.

