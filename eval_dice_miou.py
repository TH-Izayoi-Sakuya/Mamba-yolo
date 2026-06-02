# eval_dice_miou.py
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from ultralytics import YOLO

# ===================== You only need to confirm/modify these paths (Kvasir-seg) =====================
DATA_YAML = Path("data.yaml")
GT_MASKS_DIR = Path("data/kvasir-seg/masks")
RUNS_DIR = Path("runs/segment")
IMGSZ = 640
CONF = 0.25
MASK_THR = 0.5
OUT_DIR = Path("eval_out")
MAX_AUC_SAMPLES = 200_000
SEED = 0
BENCH_BATCH = 1
WARMUP = 20
ITERS = 200
# ===============================================================================

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Optional deps
try:
    import cv2
except Exception:
    cv2 = None

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception:
    roc_auc_score = None
    average_precision_score = None

try:
    from thop import profile
except ImportError:
    profile = None
    print("[WARNING] 'thop' is not installed. FLOPs calculation will be skipped. Run 'pip install thop'.")


def find_latest_best_pt(runs_dir: Path) -> Path:
    pts = list(runs_dir.rglob("weights/best.pt"))
    if not pts:
        raise FileNotFoundError(f"No best.pt found under: {runs_dir.resolve()}")
    pts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return pts[0]


def sizeof_mb(path: Path) -> float:
    return float(path.stat().st_size / (1024 * 1024))


def load_mask(p: Path) -> np.ndarray:
    arr = np.array(Image.open(p).convert("L"))
    return (arr > 0).astype(np.uint8)


def safe_div(a: float, b: float, eps: float = 1e-7) -> float:
    return float((a + eps) / (b + eps))


def resize_like(arr: np.ndarray, ref_hw: Tuple[int, int], is_prob: bool) -> np.ndarray:
    h, w = ref_hw
    if arr.shape[:2] == (h, w):
        return arr
    pil = Image.fromarray(arr)
    pil = pil.resize((w, h), resample=Image.BILINEAR if is_prob else Image.NEAREST)
    return np.array(pil)


def count_params(pt_model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in pt_model.parameters())
    trainable = sum(p.numel() for p in pt_model.parameters() if p.requires_grad)
    return total, trainable


def get_flops(pt_model: torch.nn.Module, x: torch.Tensor) -> float:
    if profile is None:
        return float("nan")
    import contextlib
    import os
    with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
        macs, _ = profile(pt_model, inputs=(x,), verbose=False)
    gflops = (macs * 2) / 1e9
    return gflops


@torch.inference_mode()
def benchmark_forward(pt_model: torch.nn.Module, x: torch.Tensor, warmup: int, iters: int) -> Dict[str, float]:
    for _ in range(warmup):
        _ = pt_model(x)
    if x.is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        _ = pt_model(x)
        if x.is_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    if x.is_cuda:
        peak_mem_mb = float(torch.cuda.max_memory_allocated() / (1024 * 1024))
    else:
        peak_mem_mb = float("nan")

    arr = np.array(times, dtype=np.float64)
    mean_ms = float(arr.mean())
    return {
        "mean_ms": mean_ms,
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "fps": float(1000.0 / mean_ms) if mean_ms > 0 else float("nan"),
        "peak_memory_mb": peak_mem_mb
    }


def confusion_from_binary(pred: np.ndarray, gt: np.ndarray) -> Tuple[int, int, int, int]:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    tp = int(np.logical_and(pred_b, gt_b).sum())
    fp = int(np.logical_and(pred_b, np.logical_not(gt_b)).sum())
    fn = int(np.logical_and(np.logical_not(pred_b), gt_b).sum())
    tn = int(np.logical_and(np.logical_not(pred_b), np.logical_not(gt_b)).sum())
    return tp, fp, fn, tn


def metrics_from_conf(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    dice = safe_div(2 * tp, 2 * tp + fp + fn)
    iou = safe_div(tp, tp + fp + fn)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    accuracy = safe_div(tp + tn, tp + tn + fp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    f2 = safe_div(5 * precision * recall, 4 * precision + recall)
    bacc = 0.5 * (recall + specificity)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) + 1e-7)
    mcc = float(((tp * tn) - (fp * fn)) / denom) if denom > 0 else 0.0

    return {
        "dice": dice, "iou": iou, "precision": precision, "recall": recall,
        "specificity": specificity, "accuracy": accuracy, "f1": f1, "f2": f2,
        "balanced_acc": float(bacc), "mcc": mcc,
    }


def _boundary(mask: np.ndarray) -> np.ndarray:
    m = (mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return np.zeros_like(m, dtype=np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    er = cv2.erode(m, kernel, iterations=1)
    return (m ^ er).astype(np.uint8)


def boundary_metrics(pred: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    if cv2 is None:
        return {"assd": float("nan"), "hd95": float("nan"), "hd_max": float("nan"), "boundary_f": float("nan")}

    pb = _boundary(pred)
    gb = _boundary(gt)

    if pb.sum() == 0 and gb.sum() == 0:
        return {"assd": 0.0, "hd95": 0.0, "hd_max": 0.0, "boundary_f": 1.0}
    if pb.sum() == 0 or gb.sum() == 0:
        return {"assd": float("nan"), "hd95": float("nan"), "hd_max": float("nan"), "boundary_f": 0.0}

    dt_g = cv2.distanceTransform((1 - gb).astype(np.uint8), cv2.DIST_L2, 3)
    dt_p = cv2.distanceTransform((1 - pb).astype(np.uint8), cv2.DIST_L2, 3)
    d_p2g = dt_g[pb.astype(bool)]
    d_g2p = dt_p[gb.astype(bool)]
    d_all = np.concatenate([d_p2g, d_g2p], axis=0)

    assd = float(np.mean(d_all)) if d_all.size else float("nan")
    hd95 = float(np.percentile(d_all, 95)) if d_all.size else float("nan")
    hd_max = float(np.max(d_all)) if d_all.size else float("nan")

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    pb_dilated = cv2.dilate(pb, kernel)
    gb_dilated = cv2.dilate(gb, kernel)
    tp_p = (pb & gb_dilated).sum()
    tp_g = (gb & pb_dilated).sum()
    precision_b = tp_p / (pb.sum() + 1e-7)
    recall_b = tp_g / (gb.sum() + 1e-7)
    boundary_f = 2 * precision_b * recall_b / (precision_b + recall_b + 1e-7)

    return {"assd": assd, "hd95": hd95, "hd_max": hd_max, "boundary_f": float(boundary_f)}


def analyze_image_attributes(img_path: Path, gt: np.ndarray) -> Tuple[str, str, str]:
    if cv2 is None:
        return "unknown", "unknown", "unknown"
    img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        return "unknown", "unknown", "unknown"

    ratio = gt.sum() / max(gt.size, 1)
    if ratio < 0.05: size_grp = "small"
    elif ratio < 0.15: size_grp = "medium"
    else: size_grp = "large"

    fg_mask = gt > 0
    bg_mask = gt == 0
    mean_fg = img_gray[fg_mask].mean() if fg_mask.sum() > 0 else 0
    mean_bg = img_gray[bg_mask].mean() if bg_mask.sum() > 0 else 0
    contrast = abs(mean_fg - mean_bg)
    contrast_grp = "low" if contrast < 50 else "high"

    gb = _boundary(gt)
    if gb.sum() > 0:
        gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(gx, gy)
        clarity = grad_mag[gb > 0].mean()
        clarity_grp = "blurred" if clarity < 30 else "clear"
    else:
        clarity_grp = "unknown"

    return size_grp, contrast_grp, clarity_grp


def auc_metrics(pred_score: np.ndarray, gt: np.ndarray, max_samples: int, seed: int) -> Dict[str, float]:
    if roc_auc_score is None or average_precision_score is None:
        return {"auroc": float("nan"), "auprc": float("nan")}
    y = gt.reshape(-1).astype(np.uint8)
    s = pred_score.reshape(-1).astype(np.float32)
    if y.min() == y.max():
        return {"auroc": float("nan"), "auprc": float("nan")}
    n = y.size
    if n > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_samples, replace=False)
        y = y[idx]
        s = s[idx]
    return {
        "auroc": float(roc_auc_score(y, s)),
        "auprc": float(average_precision_score(y, s)),
    }


@dataclass
class Row:
    image: str
    size_group: str
    contrast_group: str
    clarity_group: str
    tp: int; fp: int; fn: int; tn: int
    dice: float; iou: float; precision: float; recall: float; specificity: float; accuracy: float
    f1: float; f2: float; balanced_acc: float; mcc: float
    assd: float; hd95: float; hd_max: float; boundary_f: float
    auroc: float; auprc: float


def nanmean(xs: List[float]) -> float:
    return float(np.nanmean(np.array(xs, dtype=np.float64)))


def nanstd(xs: List[float]) -> float:
    return float(np.nanstd(np.array(xs, dtype=np.float64)))


def write_csv(rows: List[Row], out_csv: Path) -> None:
    import csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def main():
    torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ul_device = "0" if torch_device.type == "cuda" else "cpu"
    print(f"[OK] Torch device: {torch_device} | Ultralytics device: {ul_device}")

    assert RUNS_DIR.exists(), f"RUNS_DIR not found: {RUNS_DIR.resolve()}"
    best_pt = find_latest_best_pt(RUNS_DIR)
    print(f"[OK] Using weights: {best_pt}  ({sizeof_mb(best_pt):.2f} MB)")

    model = YOLO(str(best_pt))

    pt_model = model.model
    pt_model.eval()
    pt_model.to(torch_device)

    total_params, trainable_params = count_params(pt_model)
    print(f"[EFF] Params: total={total_params/1e6:.2f} M | trainable={trainable_params/1e6:.2f} M")

    x = torch.zeros((BENCH_BATCH, 3, IMGSZ, IMGSZ), device=torch_device)
    gflops = get_flops(pt_model, x)
    print(f"[EFF] Compute: {gflops:.2f} GFLOPs" if not math.isnan(gflops) else "[EFF] Compute: thop missing, skipping GFLOPs")

    lat = benchmark_forward(pt_model, x, warmup=WARMUP, iters=ITERS)
    print(f"[EFF] Latency(ms): mean={lat['mean_ms']:.2f} | median={lat['median_ms']:.2f} | p95={lat['p95_ms']:.2f}")
    print(f"[EFF] FPS: {lat['fps']:.2f}  (batch={BENCH_BATCH}, imgsz={IMGSZ})")
    if not math.isnan(lat['peak_memory_mb']):
        print(f"[EFF] Peak CUDA Memory: {lat['peak_memory_mb']:.2f} MB")

    assert DATA_YAML.exists(), f"DATA_YAML not found: {DATA_YAML.resolve()}"
    assert GT_MASKS_DIR.exists(), f"GT_MASKS_DIR not found: {GT_MASKS_DIR.resolve()}"

    d = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    root = Path(d["path"])
    val_images_dir = (root / d["val"]).resolve()
    assert val_images_dir.exists(), f"val images dir not found: {val_images_dir}"

    img_paths = sorted([p for p in val_images_dir.iterdir() if p.suffix.lower() in IMG_EXTS])
    if not img_paths:
        raise RuntimeError(f"No images found in {val_images_dir}")

    rows: List[Row] = []
    missing_gt = 0
    TP = FP = FN = TN = 0
    
    # Track per-stratum confusion matrices
    strat_conf = {
        "size": {"small": [0]*4, "medium": [0]*4, "large": [0]*4, "unknown": [0]*4},
        "contrast": {"low": [0]*4, "high": [0]*4, "unknown": [0]*4},
        "clarity": {"blurred": [0]*4, "clear": [0]*4, "unknown": [0]*4}
    }
    strat_boundary = {
        "size": {"small": [], "medium": [], "large": [], "unknown": []},
        "contrast": {"low": [], "high": [], "unknown": []},
        "clarity": {"blurred": [], "clear": [], "unknown": []}
    }

    for img_p in img_paths:
        gt_p = GT_MASKS_DIR / f"{img_p.stem}.png"
        if not gt_p.exists():
            gt_p2 = GT_MASKS_DIR / f"{img_p.stem}{img_p.suffix}"
            if gt_p2.exists():
                gt_p = gt_p2
            else:
                missing_gt += 1
                continue

        gt = load_mask(gt_p)
        size_grp, contrast_grp, clarity_grp = analyze_image_attributes(img_p, gt)

        r = model.predict(
            source=str(img_p),
            imgsz=IMGSZ,
            device=ul_device,
            conf=CONF,
            verbose=False,
            retina_masks=True,
        )[0]

        if r.masks is None:
            pred_score = np.zeros_like(gt, dtype=np.float32)
        else:
            m = r.masks.data.detach().cpu().numpy().astype(np.float32)
            pred_score = m.max(axis=0)

        pred_score = resize_like(pred_score, gt.shape, is_prob=True).astype(np.float32)
        pred_bin = (pred_score >= MASK_THR).astype(np.uint8)

        tp, fp, fn, tn = confusion_from_binary(pred_bin, gt)
        TP += tp; FP += fp; FN += fn; TN += tn
        
        for i, val in enumerate([tp, fp, fn, tn]):
            strat_conf["size"][size_grp][i] += val
            strat_conf["contrast"][contrast_grp][i] += val
            strat_conf["clarity"][clarity_grp][i] += val

        m1 = metrics_from_conf(tp, fp, fn, tn)
        m2 = boundary_metrics(pred_bin, gt)
        for cat, grp in [("size", size_grp), ("contrast", contrast_grp), ("clarity", clarity_grp)]:
            strat_boundary[cat][grp].append(m2)
        m3 = auc_metrics(pred_score, gt, max_samples=MAX_AUC_SAMPLES, seed=SEED)

        rows.append(
            Row(
                image=img_p.name,
                size_group=size_grp, contrast_group=contrast_grp, clarity_group=clarity_grp,
                tp=tp, fp=fp, fn=fn, tn=tn,
                dice=m1["dice"], iou=m1["iou"], precision=m1["precision"],
                recall=m1["recall"], specificity=m1["specificity"], accuracy=m1["accuracy"],
                f1=m1["f1"], f2=m1["f2"], balanced_acc=m1["balanced_acc"], mcc=m1["mcc"],
                assd=m2["assd"], hd95=m2["hd95"], hd_max=m2["hd_max"], boundary_f=m2["boundary_f"],
                auroc=m3["auroc"], auprc=m3["auprc"],
            )
        )

    if not rows:
        raise RuntimeError("No samples evaluated.")

    keys = [
        "dice", "iou", "precision", "recall", "specificity", "accuracy",
        "f1", "f2", "balanced_acc", "mcc", "assd", "hd95", "hd_max", "boundary_f", "auroc", "auprc"
    ]
    macro_mean = {k: nanmean([getattr(r, k) for r in rows]) for k in keys}

    # ================= Aggregate all Micro and Macro metrics into one dict =================
    final_metrics = metrics_from_conf(TP, FP, FN, TN)
    
    # Supplement macro-average metrics (these cannot be derived from the global confusion matrix)
    final_metrics["assd"] = macro_mean["assd"]
    final_metrics["hd95"] = macro_mean["hd95"]
    final_metrics["hd_max"] = macro_mean["hd_max"]
    final_metrics["boundary_f"] = macro_mean["boundary_f"]
    final_metrics["auroc"] = macro_mean["auroc"]
    final_metrics["auprc"] = macro_mean["auprc"]

    # Flatten and append stratified subgroup micro results
    for cat, groups in strat_conf.items():
        for grp, conf_vals in groups.items():
            t_p, f_p, f_n, t_n = conf_vals
            if (t_p + f_p + f_n + t_n) == 0 or grp == "unknown":
                continue 
            grp_metrics = metrics_from_conf(t_p, f_p, f_n, t_n)
            final_metrics[f"{grp}_dice"] = grp_metrics["dice"]
            final_metrics[f"{grp}_iou"] = grp_metrics["iou"]

    # Append stratified subgroup boundary metrics (ASSD/HD95/HD_max/Boundary-F)
    for cat, groups in strat_boundary.items():
        for grp, bm_list in groups.items():
            if grp == "unknown" or len(bm_list) == 0:
                continue
            final_metrics[f"{grp}_assd"] = nanmean([b["assd"] for b in bm_list])
            final_metrics[f"{grp}_hd95"] = nanmean([b["hd95"] for b in bm_list])
            final_metrics[f"{grp}_hd_max"] = nanmean([b["hd_max"] for b in bm_list])
            final_metrics[f"{grp}_boundary_f"] = nanmean([b["boundary_f"] for b in bm_list])

    print(f"\n[OK] Evaluated images: {len(rows)}  (missing GT: {missing_gt})")
    print("---- Overall Micro (global confusion) & Extended Metrics ----")
    # Adjust alignment spacing for neat key-value output
    for k, v in final_metrics.items():
        print(f"{k:>22}: {v:.6f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "per_image_metrics.csv"
    write_csv(rows, out_csv)

    summary = {
        "dataset": "Kvasir-seg",
        "weights": str(best_pt),
        "imgsz": IMGSZ,
        "efficiency": {
            "params_total": total_params,
            "flops_g": gflops,
            "fps": lat["fps"],
            "peak_memory_mb": lat["peak_memory_mb"]
        },
        "final_metrics": final_metrics
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[OK] Saved Details: {out_csv}")
    print(f"[OK] Saved Summary: {OUT_DIR / 'summary.json'}")

if __name__ == "__main__":
    main()