"""
DualDetect module — adds two YOLO heads (PPE and Fire) sharing one backbone.
"""

import torch
import torch.nn as nn
from ultralytics.nn.modules import Detect


from ultralytics.nn.modules import Detect
import torch.nn as nn

class DualDetect(nn.Module):
    def __init__(self, ppe_nc, fire_nc, ch):
        super().__init__()
        self.ppe_nc = ppe_nc
        self.fire_nc = fire_nc
        self.ppe_head = Detect(nc=ppe_nc, ch=ch)
        self.fire_head = Detect(nc=fire_nc, ch=ch)

    def forward(self, x):
        if not isinstance(x, (list, tuple)):
            x = [x]
        return {
            "ppe": self.ppe_head(x),
            "fire": self.fire_head(x)
        }


    def forward(self, x):
        # After Concat, x is a single tensor → wrap it in list for Detect
        if not isinstance(x, (list, tuple)):
            x = [x]
        return {
            "ppe": self.ppe_head(x),
            "fire": self.fire_head(x),
        }


