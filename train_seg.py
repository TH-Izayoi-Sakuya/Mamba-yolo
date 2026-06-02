# train_seg.py
from __future__ import annotations

import os
from pathlib import Path

from ultralytics import YOLO
from bdc_trainer import SegTrainerWithBDC, patch_all_conv_classes_forward_fuse


def register_custom_modules():
    import ultralytics.nn.tasks as tasks
    import custom_modules as cm

    for name in dir(cm):
        obj = getattr(cm, name)
        if isinstance(obj, type):
            tasks.__dict__[name] = obj


def resolve_path(p: str) -> str:
    return str(Path(p).resolve())


def main():
    project_dir = Path(__file__).resolve().parent
    os.chdir(project_dir)

    patch_all_conv_classes_forward_fuse(verbose=True)

    register_custom_modules()

    model_yaml = "mamba_yolo_scca_ghost_seg.yaml"
    data_yaml = "data.yaml"

    model = YOLO(model_yaml, task="segment")

    model.train(
        data=resolve_path(data_yaml),
        epochs=300,
        imgsz=640,
        batch=8,
        device=0,
        workers=8,
        seed=0,
        amp=False,
        deterministic=False,
        project=resolve_path("runs/segment"),
        name="mamba_yolo_scca_ghost_bdc",
        trainer=SegTrainerWithBDC,
    )


if __name__ == "__main__":
    main()
