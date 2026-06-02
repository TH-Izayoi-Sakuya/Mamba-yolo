# bdc_trainer.py
from __future__ import annotations

import os
import sys
import types
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any


import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.models.yolo.segment.train import SegmentationTrainer


# =============================
# BDC hyperparameters (bypass Ultralytics param validation)
# =============================
BDC_DEFAULTS = {
    "lambda_bd": 0.05,
    "pool_size": 16,
    "use_boundary_weight": True,
    "base_weight": 0.10,
    "boundary_kernel": 3,
    "hook_layers": (20,),  # After inserting ADDR, Segment head inputs [20,23,26]; layer 20 preferred
}

def _env_float(key: str, default: float) -> float:
    v = os.getenv(key, "").strip()
    return default if not v else float(v)

def _env_int(key: str, default: int) -> int:
    v = os.getenv(key, "").strip()
    return default if not v else int(v)

def _env_bool(key: str, default: bool) -> bool:
    v = os.getenv(key, "").strip().lower()
    if not v:
        return default
    return v in ("1", "true", "yes", "y", "t", "on")

def _env_layers(key: str, default: Tuple[int, ...]) -> Tuple[int, ...]:
    v = os.getenv(key, "").strip()
    if not v:
        return default
    out = []
    for s in v.split(","):
        s = s.strip()
        if not s:
            continue
        try:
            out.append(int(s))
        except Exception:
            pass
    return tuple(out) if out else default


@dataclass
class BDCConfig:
    lambda_bd: float
    pool_size: int
    use_boundary_weight: bool
    base_weight: float
    boundary_kernel: int
    hook_layers: Tuple[int, ...]


def load_bdc_config() -> BDCConfig:
    return BDCConfig(
        lambda_bd=_env_float("BD_LAMBDA", float(BDC_DEFAULTS["lambda_bd"])),
        pool_size=_env_int("BD_POOL", int(BDC_DEFAULTS["pool_size"])),
        use_boundary_weight=_env_bool("BD_USE_W", bool(BDC_DEFAULTS["use_boundary_weight"])),
        base_weight=_env_float("BD_BASE_W", float(BDC_DEFAULTS["base_weight"])),
        boundary_kernel=_env_int("BD_K", int(BDC_DEFAULTS["boundary_kernel"])),
        hook_layers=_env_layers("BD_LAYERS", tuple(BDC_DEFAULTS["hook_layers"])),
    )


# =============================
# Global forward_fuse patch (class-level)
# =============================
def _forward_fuse_impl(self, x):
    # After fuse, BN is merged into conv, so use act(conv(x))
    return self.act(self.conv(x))

def patch_all_conv_classes_forward_fuse(verbose: bool = False) -> int:
    '''
    Patch forward_fuse onto all classes named “Conv” in the current Python process,
    so model.fuse() won't fail/skip due to missing forward_fuse.
    '''
    checked = 0
    patched = 0
    for m in list(sys.modules.values()):
        if m is None:
            continue
        for name in dir(m):
            try:
                obj = getattr(m, name)
            except Exception:
                continue
            if not isinstance(obj, type):
                continue
            if obj.__name__ != "Conv":
                continue
            checked += 1
            if hasattr(obj, "forward_fuse"):
                continue
            try:
                obj.forward_fuse = _forward_fuse_impl
                patched += 1
            except Exception:
                pass
    if verbose:
        print(f"[PATCH] Conv-classes checked: {checked}, class patched: {patched}")
    return patched


def patch_all_conv_instances_forward_fuse(model: nn.Module, verbose: bool = False) -> Tuple[int, int]:
    """
    Patch forward_fuse on all conv-like instances in a model (instance-level).
    """
    checked = 0
    patched = 0
    for m in model.modules():
        # only patch modules that have conv+bn+act
        if hasattr(m, "conv") and hasattr(m, "bn") and hasattr(m, "act"):
            checked += 1
            if not hasattr(m, "forward_fuse"):
                try:
                    m.forward_fuse = types.MethodType(_forward_fuse_impl, m)
                    patched += 1
                except Exception:
                    pass
    if verbose:
        print(f"[PATCH] conv-like checked: {checked}, instance patched: {patched}")
    return checked, patched


# =============================
# Ultralytics layer container retrieval
# =============================
def _unwrap_model_layers(yolo_model: Any) -> List[nn.Module]:
    """
    Compatible with Ultralytics model wrappers as much as possible.
    """
    m = yolo_model
    if hasattr(m, "model"):
        m = m.model
    if hasattr(m, "model"):
        m = m.model
    if isinstance(m, (nn.Sequential, nn.ModuleList)):
        return list(m)
    if hasattr(m, "children"):
        return list(m.children())
    return []


# =============================
# Helper: boundary map
# =============================
def mask_to_boundary(mask: torch.Tensor, k: int = 3) -> torch.Tensor:
    """
    mask: (B,1,H,W) in {0,1}
    return boundary map (B,1,H,W)
    """
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    mask = mask.float()
    pad = k // 2
    dil = F.max_pool2d(mask, kernel_size=k, stride=1, padding=pad)
    ero = 1.0 - F.max_pool2d(1.0 - mask, kernel_size=k, stride=1, padding=pad)
    b = (dil - ero).clamp_(0.0, 1.0)
    return b


# =============================
#  BDC：hook + loss
# =============================
class BoundaryDecouplingConstraint(nn.Module):
    def __init__(self, cfg: BDCConfig):
        super().__init__()
        self.cfg = cfg
        self.hooks: List[torch.utils.hooks.RemovableHandle] = []
        self.features: Dict[int, torch.Tensor] = {}

    def clear(self):
        self.features.clear()

    def detach(self):
        for h in self.hooks:
            try:
                h.remove()
            except Exception:
                pass
        self.hooks.clear()

    def attach_to(self, yolo_model):
        self.detach()
        layers = _unwrap_model_layers(yolo_model)
        layer_ids = set(self.cfg.hook_layers)

        def _make_hook(idx: int):
            def _hook(module, inp, out):
                if torch.is_tensor(out):
                    self.features[idx] = out
                elif isinstance(out, (list, tuple)) and len(out) > 0 and torch.is_tensor(out[0]):
                    self.features[idx] = out[0]
            return _hook

        for i, layer in enumerate(layers):
            if i in layer_ids:
                self.hooks.append(layer.register_forward_hook(_make_hook(i)))

        if self.hooks:
            print(f"[BDC] enabled, hook_layers={self.cfg.hook_layers}")
        else:
            print("[BDC] no hooks attached (check hook_layers index)")

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        pred: (B,1,H,W) sigmoid prob
        gt:   (B,1,H,W) binary
        """
        cfg = self.cfg
        if pred.ndim == 3:
            pred = pred.unsqueeze(1)
        if gt.ndim == 3:
            gt = gt.unsqueeze(1)

        gt = (gt > 0.5).float()
        pred = pred.float().clamp(0.0, 1.0)

        # boundary target
        bd = mask_to_boundary(gt, k=cfg.boundary_kernel)

        # downsample
        if cfg.pool_size > 1:
            pred_s = F.avg_pool2d(pred, cfg.pool_size, cfg.pool_size)
            gt_s = F.avg_pool2d(gt, cfg.pool_size, cfg.pool_size)
            bd_s = F.avg_pool2d(bd, cfg.pool_size, cfg.pool_size)
        else:
            pred_s, gt_s, bd_s = pred, gt, bd

        # weighted BCE (boundary upweight)
        if cfg.use_boundary_weight:
            w = cfg.base_weight + bd_s
            loss_bd = (F.binary_cross_entropy(pred_s, gt_s, reduction="none") * w).mean()
        else:
            loss_bd = F.binary_cross_entropy(pred_s, gt_s)

        # feature constraint (simplified: feature gradients are larger at boundaries)
        loss_feat = 0.0
        n = 0
        for idx in cfg.hook_layers:
            feat = self.features.get(idx, None)
            if feat is None or feat.ndim != 4:
                continue
            f = feat.mean(dim=1, keepdim=True)
            gx = f[:, :, :, 1:] - f[:, :, :, :-1]
            gy = f[:, :, 1:, :] - f[:, :, :-1, :]
            gx = F.pad(gx, (0, 1, 0, 0))
            gy = F.pad(gy, (0, 0, 0, 1))
            gmag = torch.sqrt(gx * gx + gy * gy + 1e-7)

            bd_r = F.interpolate(bd, size=gmag.shape[-2:], mode="nearest")
            loss_feat = loss_feat + (gmag * bd_r).mean()
            n += 1
        if n > 0:
            loss_feat = loss_feat / n

        return cfg.lambda_bd * (loss_bd + loss_feat)


# =============================
# Trainer with BDC
# =============================
class SegTrainerWithBDC(SegmentationTrainer):
    """
    Extends Ultralytics SegmentationTrainer, adding BDC loss during training.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bdc_cfg = load_bdc_config()
        self.bdc = BoundaryDecouplingConstraint(self.bdc_cfg)

    def _setup_train(self, *args, **kwargs):
        super()._setup_train(*args, **kwargs)
        # attach hooks after model built
        try:
            self.bdc.attach_to(self.model)
        except Exception as e:
            print(f"[BDC] attach failed: {e}")

    def loss(self, batch, preds=None):
        # original loss (including seg loss)
        loss, loss_items = super().loss(batch, preds)

        # get GT mask
        # Ultralytics seg batch common key: 'masks' (B,H,W) or (B,1,H,W)
        gt = batch.get("masks", None)
        if gt is None:
            return loss, loss_items

        if gt.ndim == 3:
            gt = gt.unsqueeze(1)

        # get pred mask (find from preds)
        # Ultralytics segment preds usually contain proto and mask coeff etc.; pred mask during training may not be directly given
        # Safe strategy: use 'pred_masks' from batch if available; otherwise skip BDC
        pred = batch.get("pred_masks", None)
        if pred is None:
            return loss, loss_items

        if pred.ndim == 3:
            pred = pred.unsqueeze(1)

        # BDC loss
        try:
            bdc_loss = self.bdc(pred, gt)
            loss = loss + bdc_loss
            if isinstance(loss_items, dict):
                loss_items["bdc"] = float(bdc_loss.detach().cpu())
        except Exception as e:
            # Prevent BDC from interrupting the main training flow
            if os.getenv("BDC_DEBUG", "").strip():
                print(f"[BDC] loss failed: {e}")

        return loss, loss_items
