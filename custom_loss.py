from typing import Optional

from segmentation_models_pytorch import losses
from segmentation_models_pytorch.losses.constants import BINARY_MODE, MULTICLASS_MODE, MULTILABEL_MODE
from segmentation_models_pytorch.utils import base
import torch

class FocalLoss(losses.FocalLoss):
    def __init__(
        self,
        mode: str,
        alpha: Optional[float] = None,
        gamma: Optional[float] = 2.0,
        ignore_index: Optional[int] = None,
        reduction: Optional[str] = "mean",
        normalized: bool = False,
        reduced_threshold: Optional[float] = None,
    ):
        super().__init__(mode, alpha, gamma, ignore_index, reduction, normalized, reduced_threshold)

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:

        if self.mode in {BINARY_MODE, MULTILABEL_MODE}:
            y_true = y_true.view(-1)
            y_pred = y_pred.view(-1)

            if self.ignore_index is not None:
                # Filter predictions with ignore label from loss computation
                not_ignored = y_true != self.ignore_index
                y_pred = y_pred[not_ignored]
                y_true = y_true[not_ignored]

            loss = self.focal_loss_fn(y_pred, y_true)

        elif self.mode == MULTICLASS_MODE:

            num_classes = y_pred.size(1)
            loss = 0

            # Filter anchors with -1 label from loss computation
            if self.ignore_index is not None:
                not_ignored = y_true != self.ignore_index

            for cls in range(num_classes):
                cls_y_true = (y_true == cls).long()
                cls_y_pred = y_pred[:, cls, ...]
                if len(cls_y_true.shape) != len(cls_y_pred.shape):
                    cls_y_true = cls_y_true.view_as(cls_y_pred)
                if self.ignore_index is not None:
                    cls_y_true = cls_y_true[not_ignored]
                    cls_y_pred = cls_y_pred[not_ignored]
                loss += self.focal_loss_fn(cls_y_pred, cls_y_true)

        return loss
class CustomLoss(base.Loss):
    def __init__(self, num_class=1, loss_combination="focal_dice", **kwargs) -> None:
        super().__init__(**kwargs)
        mode="binary" if num_class==1 else "multiclass"
        if num_class>1:
            if loss_combination == "focal_dice":            
                self.loss_funcs = [FocalLoss(mode), losses.DiceLoss(mode)]
            elif loss_combination == "focal_tversky":
                self.loss_funcs = [FocalLoss(mode), losses.TverskyLoss(mode)]
            elif loss_combination == "tversky":
                self.loss_funcs = [losses.TverskyLoss(mode)]
            elif loss_combination == "dice":
                self.loss_funcs = [losses.DiceLoss(mode)]
            elif loss_combination == "focal":
                self.loss_funcs = [FocalLoss(mode)]
        else:
            self.loss_funcs = [losses.DiceLoss(mode), losses.JaccardLoss(mode)]
        self._name = "".join([str(loss_func)+" + " for loss_func in self.loss_funcs])
        
    def forward(self, y_pr, y_gt):
        loss = 0
        y_gt = y_gt.long()
        for loss_func in self.loss_funcs:
            loss += loss_func.forward(y_pr, y_gt)/len(self.loss_funcs)
        return loss
    