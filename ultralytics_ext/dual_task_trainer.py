"""
DualTaskTrainer: trains two detection heads (PPE + Fire) on shared backbone.
Alternates mini-batches between tasks to balance learning.
"""

import itertools
import torch
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils import LOGGER
from ultralytics.data import build_dataloader
from ultralytics.data import build_yolo_dataset


class DualTaskTrainer(DetectionTrainer):
    """
    Extension of Ultralytics DetectionTrainer that alternates between two datasets.
    """

    def set_head_configs(self, head_cfgs):
        """
        Store dataset paths and class counts for each head.
        Example:
        {
          'ppe': {'nc': 3, 'data': 'data/ppe.yaml', 'loss_weight': 1.0},
          'fire': {'nc': 2, 'data': 'data/fire.yaml', 'loss_weight': 1.0}
        }
        """
        self.head_cfgs = head_cfgs
        self.loss_weights = {k: v.get("loss_weight", 1.0) for k, v in head_cfgs.items()}

    def build_dataset(self, task_name, mode="train"):
        """Build YOLO dataset for a given task."""
        data_yaml = self.head_cfgs[task_name]["data"]
        overrides = self.args.copy()
        overrides["data"] = data_yaml
        dataset = build_yolo_dataset(self.model.args, overrides, img_path=None, mode=mode, rect=(mode == "val"))
        return dataset

    def get_dataloader_for_task(self, task_name, mode="train"):
        """Return dataloader for the specific task."""
        dataset = self.build_dataset(task_name, mode)
        shuffle = (mode == "train")
        batch_size = self.args.batch if mode == "train" else max(1, self.args.batch // 2)
        return build_dataloader(dataset, batch_size=batch_size,
                                workers=self.args.workers, shuffle=shuffle,
                                rank=self.args.rank, mode=mode)

    def train(self):
        """Train alternating between PPE and FIRE datasets each iteration."""
        device = self.device
        model = self.model
        model.to(device)

        dl_ppe = self.get_dataloader_for_task("ppe", "train")
        dl_fire = self.get_dataloader_for_task("fire", "train")

        it_ppe = itertools.cycle(iter(dl_ppe))
        it_fire = itertools.cycle(iter(dl_fire))

        n_batches = max(len(dl_ppe), len(dl_fire))
        n_epochs = self.epochs

        LOGGER.info(f"Starting dual-head training for {n_epochs} epochs ({len(dl_ppe)} PPE + {len(dl_fire)} FIRE batches/epoch).")

        for epoch in range(n_epochs):
            model.train()
            for i in range(n_batches):
                if i % 2 == 0:
                    batch = next(it_ppe)
                    head = "ppe"
                else:
                    batch = next(it_fire)
                    head = "fire"

                imgs = batch["img"].to(device, non_blocking=True).float() / 255.0
                preds = model(imgs)
                preds = preds[head]  # pick outputs for that head

                self.optimizer.zero_grad(set_to_none=True)
                loss, loss_items = self.criterion(preds, batch)
                (loss * self.loss_weights[head]).backward()
                self.optimizer.step()

            LOGGER.info(f"Epoch {epoch+1}/{n_epochs} finished ✓")

        LOGGER.info("Training completed.")
