#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multitask YOLO11-S (Shared Backbone: Detection + Wrist Pose)
============================================================

Stage 0 → Pretrained YOLO11-s-pose.pt (NO TRAINING)
Stage 1 → Copy pose backbone → YOLO11-s detector
Stage 2 → Train ONLY detection head on unify_safety
Inference → ONE forward pass → detection + wrist keypoints
Optimized for AI-COMS multi-camera real-time streaming.
"""

from __future__ import annotations
import os, time, yaml
from pathlib import Path
from typing import Any, Dict
from datetime import datetime

import torch
import torch.nn as nn
import cv2
from ultralytics import YOLO


# =============================================================================
# ============================== CONFIG =======================================
# =============================================================================

# Use YOLO11-S for both pose & detection
POSE_PT = r"C:\models\yolo11s-pose.pt"          # Pretrained YOLO11-s pose model
DET_PT  = r"C:\models\yolo11s.pt"               # Pretrained YOLO11-s detector

DET_DATA_YAML = r"C:\data\unify_safety.yaml"    # PPE dataset (helmet, vest, head, fire)

SAVE_DIR = Path("#unified_yolo11s_detpose")
DEVICE = "0"

# YOLO11 backbone split point
BACKBONE_END_IDX = 11
IMG_SIZE = 640
CONF_THRES = 0.25
FPS_TEST_IMGS = 50


# =============================================================================
# ============================== HELPERS ======================================
# =============================================================================

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def write_yaml(obj, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, allow_unicode=True)


def transfer_backbone(det_model, pose_wts):
    print("\n Transferring Pose → Detection backbone ...")

    ckpt = torch.load(pose_wts, map_location="cpu")
    pose_sd = ckpt["model"].state_dict() if "model" in ckpt else ckpt
    det_sd = det_model.model.state_dict()

    tr = sk = 0
    for k, v in pose_sd.items():
        # skip pose/kpt head layers
        if "pose" in k or "kpt" in k or "head" in k:
            sk += 1
            continue

        if k in det_sd and det_sd[k].shape == v.shape:
            det_sd[k] = v
            tr += 1
        else:
            sk += 1

    det_model.model.load_state_dict(det_sd, strict=False)
    print(f"   ✓ transferred={tr}, skipped={sk}")


def freeze_backbone(model, keep=3):
    print("🔒 Freezing backbone (train only last layers)...")
    layers = list(model.model.model)
    for lyr in layers[:-keep]:
        for p in lyr.parameters():
            p.requires_grad = False


def measure_fps(model, dataset_yaml, n=50, imgsz=640):
    try:
        ds = read_yaml(dataset_yaml)
        base = Path(ds.get("path", "."))
        rel = ds.get("test", ds.get("val", "images/val"))
        img_dir = base / rel

        imgs = []
        for e in ["*.jpg", "*.png", "*.jpeg"]:
            imgs.extend(img_dir.glob(e))

        imgs = imgs[:min(n, len(imgs))]
        if not imgs:
            return None

        dev = next(model.parameters()).device

        # warmup
        w = cv2.imread(str(imgs[0]))
        t = torch.from_numpy(cv2.cvtColor(w, cv2.COLOR_BGR2RGB)).permute(2,0,1).float().unsqueeze(0)/255
        t = torch.nn.functional.interpolate(t, (imgsz, imgsz)).to(dev)
        for _ in range(3): model(t)

        # measure
        total = ok = 0
        for im_path in imgs:
            im = cv2.imread(str(im_path))
            if im is None: continue

            t = torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)).permute(2,0,1).float().unsqueeze(0)/255
            t = torch.nn.functional.interpolate(t, (imgsz, imgsz)).to(dev)

            torch.cuda.synchronize()
            s = time.time()
            model(t)
            torch.cuda.synchronize()

            total += (time.time() - s)
            ok += 1

        return ok / total if ok > 0 else None

    except Exception as e:
        print("FPS error:", e)
        return None


# =============================================================================
# ========================= SHARED BACKBONE MODEL =============================
# =============================================================================

class DualHeadDetPose(nn.Module):
    """Shared YOLO11-S backbone → Detection + Pose Heads"""

    def __init__(self, det_model: YOLO, pose_model: YOLO, backbone_end_idx=11):
        super().__init__()
        det_raw = det_model.model
        pose_raw = pose_model.model

        self.backbone = nn.Sequential(*list(pose_raw.model)[:backbone_end_idx])

        self.det_head  = nn.Sequential(*list(det_raw.model)[backbone_end_idx:])
        self.pose_head = nn.Sequential(*list(pose_raw.model)[backbone_end_idx:])

        self.device = next(pose_raw.parameters()).device

    @torch.no_grad()
    def forward(self, x):
        feats = self.backbone(x)
        return self.det_head(feats), self.pose_head(feats)


# =============================================================================
# ============================= PIPELINE ======================================
# =============================================================================

class MultiTaskDetPose:

    @staticmethod
    def train_pipeline():
        ensure_dir(SAVE_DIR)
        det_dir = SAVE_DIR / "det_unify"

        print("\n=== Stage 0: Load Pretrained YOLO11-s-Pose (NO TRAIN) ===")
        print(f"Using: {POSE_PT}")

        print("\n=== Stage 1: Load YOLO11-s Detector ===")
        det_model = YOLO(DET_PT)

        print("\n=== Backbone Transfer S → S ===")
        transfer_backbone(det_model, POSE_PT)

        print("\n=== Freeze Backbone ===")
        freeze_backbone(det_model, keep=3)

        print("\n=== Stage 2: Train Detection Head (unify_safety) ===")
        det_model.train(
            data=DET_DATA_YAML,
            epochs=80,
            batch=8,
            device=DEVICE,
            project=str(SAVE_DIR),
            name="det_unify",
            exist_ok=True,
            imgsz=640,
            lr0=0.001,
            warmup_epochs=3
        )

        det_best = det_dir / "weights" / "best.pt"

        info = {
            "pose_backbone": str(POSE_PT),
            "det_best": str(det_best),
            "save_dir": str(SAVE_DIR),
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        write_yaml(info, SAVE_DIR / "training_info.yaml")

        print("\n YOLO11-s unified detection training complete.")
        return info


    @staticmethod
    def unified_test(det_w, pose_w, imgsz=640, conf=0.25, fps_imgs=50):
        print("\n Building Unified YOLO11-s (Det + Pose) Model...")

        det_model = YOLO(det_w)
        pose_model = YOLO(pose_w)

        unified = DualHeadDetPose(det_model, pose_model,
                                  backbone_end_idx=BACKBONE_END_IDX)

        dev = f"cuda:{DEVICE}" if torch.cuda.is_available() else "cpu"
        unified = unified.to(dev).eval()

        print("   ✓ Unified model ready on", dev)

        print("\n⚡ FPS Test...")
        fps = measure_fps(unified, DET_DATA_YAML, n=fps_imgs, imgsz=imgsz)
        if fps:
            print(f"   ✓ Unified FPS ≈ {fps:.2f}")
        else:
            print("   ⚠ FPS skipped")

        return unified


# =============================================================================
# ================================= MAIN ======================================
# =============================================================================

if __name__ == "__main__":
    results = MultiTaskDetPose.train_pipeline()

    MultiTaskDetPose.unified_test(
        det_w=results["det_best"],
        pose_w=results["pose_backbone"],
        imgsz=IMG_SIZE,
        conf=CONF_THRES,
        fps_imgs=FPS_TEST_IMGS
    )
