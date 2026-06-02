from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# =========================
# Kvasir-seg dataset inference path configuration
# =========================
MODEL_PATH = r"\runs\segment\mamba_yolo_scca_ghost_bdc\weights\best.pt"
IMAGE_DIR = r"\data\test"
OUT_DIR = r"\data\val"
# =========================

IMGSZ = 640
CONF = 0.25
MASK_THR = 0.5

# Blue (OpenCV uses BGR)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)

BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.8
FONT_THICKNESS = 2
ALPHA = 0.45  # mask transparency

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def draw_mask(image: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha=0.45) -> np.ndarray:
    """
    Overlay a single binary mask on the original image.
    image: BGR uint8
    mask: HxW, bool/0-1
    """
    out = image.copy()
    mask = mask.astype(bool)

    overlay = np.zeros_like(out, dtype=np.uint8)
    overlay[mask] = color

    out = np.where(mask[..., None], cv2.addWeighted(out, 1 - alpha, overlay, alpha, 0), out)
    return out


def draw_box_and_label(image: np.ndarray, box, label_text: str, color=(255, 0, 0)) -> np.ndarray:
    """
    Draw bounding box and label.
    box: [x1, y1, x2, y2]
    """
    out = image.copy()
    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(out, (x1, y1), (x2, y2), color, BOX_THICKNESS)

    (tw, th), baseline = cv2.getTextSize(label_text, FONT, FONT_SCALE, FONT_THICKNESS)

    # Label box position: prefer above the box, fallback to top-inside
    text_y1 = y1 - th - baseline - 4
    text_y2 = y1
    if text_y1 < 0:
        text_y1 = y1
        text_y2 = y1 + th + baseline + 4

    text_x1 = x1
    text_x2 = x1 + tw + 6

    cv2.rectangle(out, (text_x1, text_y1), (text_x2, text_y2), color, -1)

    text_org = (text_x1 + 3, text_y2 - baseline - 2)
    cv2.putText(out, label_text, text_org, FONT, FONT_SCALE, WHITE, FONT_THICKNESS, cv2.LINE_AA)

    return out


def main():
    image_dir = Path(IMAGE_DIR)
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not image_dir.exists():
        raise FileNotFoundError(f"IMAGE_DIR does not exist: {image_dir}")

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"MODEL_PATH does not exist: {MODEL_PATH}")

    model = YOLO(MODEL_PATH)

    img_paths = sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    if not img_paths:
        raise RuntimeError(f"No images found in directory: {image_dir}")

    for img_path in img_paths:
        # Original image: keep raw background, no filters
        image_bgr = cv2.imread(str(img_path))
        if image_bgr is None:
            print(f"[WARN] Failed to read: {img_path}")
            continue

        results = model.predict(
            source=str(img_path),
            imgsz=IMGSZ,
            conf=CONF,
            save=False,
            verbose=False,
            retina_masks=True
        )

        r = results[0]
        out_img = image_bgr.copy()

        # Draw masks first
        if r.masks is not None:
            masks = r.masks.data.detach().cpu().numpy()  # (n,h,w)
            h, w = out_img.shape[:2]

            resized_masks = []
            for m in masks:
                m_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
                m_bin = (m_resized >= MASK_THR).astype(np.uint8)
                resized_masks.append(m_bin)

            for m_bin in resized_masks:
                out_img = draw_mask(out_img, m_bin, color=BLUE, alpha=ALPHA)

        # Then draw boxes and labels
        if r.boxes is not None and len(r.boxes) > 0:
            boxes_xyxy = r.boxes.xyxy.detach().cpu().numpy()
            confs = r.boxes.conf.detach().cpu().numpy()
            
            # Directly skip reading class names, hardcode as kvasir-seg
            for box, conf in zip(boxes_xyxy, confs):
                label_text = f"kvasir-seg {conf:.2f}"
                out_img = draw_box_and_label(out_img, box, label_text, color=BLUE)

        save_path = out_dir / img_path.name
        cv2.imwrite(str(save_path), out_img)
        print(f"[OK] saved: {save_path}")

    print(f"\nAll done. Saved {len(img_paths)} result images to: {out_dir}")


if __name__ == "__main__":
    main()