# GhostConv-Mamba-YOLO

[![DOI](https://zenodo.org/badge/1256952707.svg)](https://doi.org/10.5281/zenodo.20507899)

##  Environment Setup
The code is built on top of the `ultralytics` framework with customized State-Space Models (Mamba). Due to the strict compilation requirements of Mamba, we provide specific installation instructions.

**Dependencies:**
- Python == 3.10
- PyTorch == 2.1.1
- [Ultralytics](https://github.com/ultralytics/ultralytics) >= 8.0.0
- Additional tools: `thop`, `scikit-learn`, `opencv-python`

**Installation (Windows User Guide):**
Since compiling `mamba_ssm` and `causal_conv1d` from scratch on Windows often causes errors, we highly recommend using pre-compiled `.whl` files.
**🚨 Important:** All required pre-compiled `.whl` files (Triton, causal_conv1d, mamba_ssm) are explicitly provided in the [GitHub Releases](https://github.com/TH-Izayoi-Sakuya/Mamba-yolo/releases) page of this repository. Please download them to your local directory before running the commands below.

```bash
# 1. Create and activate a Conda environment
conda create -n mamba python=3.10
conda activate mamba

# 2. Install PyTorch and CUDA toolkit
conda install cudatoolkit==11.8
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118

# 3. Handle Setuptools and Packaging dependencies
pip install setuptools==68.2.2
conda install packaging

# 4. Navigate to the directory where you downloaded the .whl files from our Release page
cd path/to/your/downloaded/whl_files

# 5. Install the pre-compiled wheels
pip install triton-2.0.0-cp310-cp310-win_amd64.whl
pip install causal_conv1d-1.1.1-cp310-cp310-win_amd64.whl
pip install mamba_ssm-1.1.3-cp310-cp310-win_amd64.whl

# 6. Prevent Numpy version conflict
pip install "numpy<2.0"

# 7. Install remaining dependencies
pip install ultralytics opencv-python scikit-learn thop
```
*Verification:* Run `python -c "import mamba_ssm; print('Import successful!')"` to ensure Mamba is installed correctly.

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
```bash
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

##  Citation
If you find this code useful for your research, please consider citing our paper once published:

```bibtex
@article{zhang2026boundary,
  title={Boundary-Aware Lightweight Global-Context Segmentation for Small Colorectal Polyps in Colonoscopy Images},
  author={Zhang, Wentao and Tao, Maohu and Qin, Tao and Zhang, Yanduo},
  journal={Pattern Analysis and Applications},
  year={2026},
  publisher={Springer}
}
```
*(The citation details will be updated upon publication.)*
