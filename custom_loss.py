from typing import Optional

from segmentation_models_pytorch import losses
from segmentation_models_pytorch.losses.constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE
from segmentation_models_pytorch.utils import base
import torch

class CustomLoss(base.Loss):
    def __init__(self, num_class=1, **kwargs) -> None:
        super().__init__(**kwargs)
        mode="binary" if num_class==1 else "multiclass"
        # if num_class>1:
        self.loss_funcs = [losses.SoftBCEWithLogitsLoss() ]
        # else:
            # self.loss_funcs = [losses.DiceLoss(mode), losses.JaccardLoss(mode)]
        self._name = "".join([str(loss_func)+" + " for loss_func in self.loss_funcs])
        
    def forward(self, y_pr, y_gt):
        loss = 0
        y_gt = y_gt.long()
        for loss_func in self.loss_funcs:
            loss += loss_func.forward(y_pr, y_gt)
        return loss
    