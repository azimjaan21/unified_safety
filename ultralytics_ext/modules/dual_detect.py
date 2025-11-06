"""
DualDetect module — adds two YOLO heads (PPE and Fire) sharing one backbone.
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules import Detect


class DualDetect(nn.Module):
    """
    DualDetect: wraps two Detect heads using the same feature maps.
    Each head produces detections for its own set of classes.
    Forward returns a dictionary: {"ppe": preds1, "fire": preds2}
    """

    def __init__(self, nc_list):
        super().__init__()
        # Expect [[3], [2]] = 3 PPE classes, 2 Fire classes
        assert isinstance(nc_list, (list, tuple)) and len(nc_list) == 2, \
            "nc_list must be [[ppe_nc], [fire_nc]]"
        self.ppe_nc = int(nc_list[0][0])
        self.fire_nc = int(nc_list[1][0])

        # YOLO Detect layers (Ultralytics auto-fills ch at build time)
        self.ppe_head = Detect(nc=self.ppe_nc, ch=[])
        self.fire_head = Detect(nc=self.fire_nc, ch=[])

    def forward(self, x):
        # x is a list of [P3, P4, P5] feature maps
        ppe_pred = self.ppe_head(x)
        fire_pred = self.fire_head(x)
        return {"ppe": ppe_pred, "fire": fire_pred}
