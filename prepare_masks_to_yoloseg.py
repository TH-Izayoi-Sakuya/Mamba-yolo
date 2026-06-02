import random
import shutil
from pathlib import Path

import cv2
import numpy as np

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_mask_for_image(img_path: Path, masks_dir: Path) -> Path | None:
    stem = img_path.stem
    candidates = [
        masks_dir / f"{stem}.png",
        masks_dir / f"{stem}.jpg",
        masks_dir / f"{stem}.jpeg",
        masks_dir / f"{stem}_mask.png",
        masks_dir / f"{stem}_mask.jpg",
        masks_dir / f"{stem}_mask.jpeg",
    ]
    for p in candidates:
        if p.exists():
            return p
    for p in masks_dir.glob(stem + ".*"):
        return p
    for p in masks_dir.glob(stem + "_mask.*"):
        return p
    return None


def mask_to_polygons(mask: np.ndarray, min_area: int = 30):
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    _, binm = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        eps = 0.002 * peri
        approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2)
        if approx.shape[0] >= 3:
            polys.append(approx)
    return polys


def write_yolo_seg_label(txt_path: Path, polys, w: int, h: int, class_id: int = 0):
    lines = []
    for poly in polys:
        pts = poly.astype(np.float32)
        pts[:, 0] = np.clip(pts[:, 0] / w, 0, 1)
        pts[:, 1] = np.clip(pts[:, 1] / h, 0, 1)
        flat = " ".join([f"{v:.6f}" for v in pts.reshape(-1)])
        lines.append(f"{class_id} {flat}")
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main():
    base = Path(".").resolve()
    src_root = base / "data" / "kvasir-seg"
    images_dir = src_root / "images"
    masks_dir = src_root / "masks"
    assert images_dir.exists(), f"not found: {images_dir}"
    assert masks_dir.exists(), f"not found: {masks_dir}"

    out_root = base / "data" / "kvasir_yolo_seg"
    img_train = out_root / "images" / "train"
    img_val = out_root / "images" / "val"
    lab_train = out_root / "labels" / "train"
    lab_val = out_root / "labels" / "val"

    imgs = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    imgs.sort()
    assert len(imgs) > 0, f"no images in {images_dir}"

    random.seed(42)
    random.shuffle(imgs)
    val_ratio = 0.1
    n_val = max(1, int(len(imgs) * val_ratio))
    val_set = set(imgs[:n_val])
    train_set = set(imgs[n_val:])

    img_train.mkdir(parents=True, exist_ok=True)
    img_val.mkdir(parents=True, exist_ok=True)
    lab_train.mkdir(parents=True, exist_ok=True)
    lab_val.mkdir(parents=True, exist_ok=True)

    def process_one(img_path: Path, is_val: bool):
        mpath = find_mask_for_image(img_path, masks_dir)
        if mpath is None:
            print(f"[SKIP] no mask for {img_path.name}")
            return

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[SKIP] cannot read image {img_path}")
            return
        h, w = img.shape[:2]

        mask = cv2.imread(str(mpath), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[SKIP] cannot read mask {mpath}")
            return
        if mask.shape[:2] != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        polys = mask_to_polygons(mask, min_area=30)

        dst_img = (img_val if is_val else img_train) / img_path.name
        shutil.copy2(img_path, dst_img)

        dst_txt = (lab_val if is_val else lab_train) / (img_path.stem + ".txt")
        write_yolo_seg_label(dst_txt, polys, w=w, h=h, class_id=0)

    for p in train_set:
        process_one(p, is_val=False)
    for p in val_set:
        process_one(p, is_val=True)

    print(f"[OK] YOLO-seg dataset created at: {out_root}")
    print(f"     train images: {len(list(img_train.iterdir()))}")
    print(f"     val images  : {len(list(img_val.iterdir()))}")


if __name__ == "__main__":
    main()
