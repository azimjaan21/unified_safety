"""
train_dualhead.py
-----------------
Train a YOLOv11m shared-backbone dual-head model:
Head A → PPE (helmet, vest, head)
Head B → Fire/Smoke
"""

import argparse
import yaml
from ultralytics import YOLO
from ultralytics_ext.dual_task_trainer import DualTaskTrainer


def main(opt):
    # 1️⃣ Load dual-task configuration
    with open(opt.dual_data, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 2️⃣ Load YOLO model definition (shared backbone + DualDetect)
    print(f"🔹 Loading model from: {opt.model}")
    model = YOLO(opt.model)  # e.g., configs/yolo11m_dualhead.yaml

    # 3️⃣ Build trainer
    print("🔹 Initializing DualTaskTrainer...")
    trainer = DualTaskTrainer(overrides={
        "imgsz": cfg.get("imgsz", 640),
        "batch": cfg.get("batch", 32),
        "epochs": cfg.get("epochs", 100),
        "patience": cfg.get("patience", 20),
        "optimizer": cfg.get("optimizer", "auto"),
        "lr0": cfg.get("lr0", 0.01),
        "weight_decay": cfg.get("weight_decay", 0.0005),
        "warmup_epochs": cfg.get("warmup_epochs", 3),
        "project": opt.project,
        "name": opt.name,
        "workers": opt.workers,
        "device": opt.device
    })

    # 4️⃣ Map head configs
    head_cfgs = {
        h["name"]: {
            "nc": h["nc"],
            "data": h["data"],
            "loss_weight": h.get("loss_weight", 1.0)
        } for h in cfg["heads"]
    }

    # 5️⃣ Attach model & criterion
    trainer.model = model.model   # underlying nn.Module
    trainer.criterion = model.trainer.build_criterion(trainer.model)
    trainer.set_head_configs(head_cfgs)

    # 6️⃣ Start training loop
    print("🚀 Starting training...")
    trainer.train()

    print("✅ Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dual-head YOLOv11m Training Script")
    parser.add_argument("--model", default="configs/yolo11m_dualhead.yaml",
                        help="Path to dual-head model YAML")
    parser.add_argument("--dual_data", default="configs/dual_data.yaml",
                        help="Path to dual data config YAML")
    parser.add_argument("--project", default="runs/dualhead",
                        help="Training output directory")
    parser.add_argument("--name", default="yolo11m_dualhead",
                        help="Experiment name")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of dataloader workers")
    parser.add_argument("--device", default="0",
                        help="CUDA device ID (e.g. 0, 0,1,2,3 or 'cpu')")
    args = parser.parse_args()

    main(args)
