import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def kd_loss_normal(logits_student, logits_teacher,t_norm):
    tstd=logits_teacher.std(dim=1,keepdim=True)
    sstd=logits_student.std(dim=1,keepdim=True)

    dywt=tstd*t_norm
    dyws=sstd*t_norm

    rt=(logits_teacher)/dywt
    rs=(logits_student)/dyws

    log_pred_student = F.log_softmax(rs, dim=1)
    pred_teacher = F.softmax(rt, dim=1)
    
    loss_kd = (F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1,keepdim=True)*(dywt**2)).mean()

    return loss_kd

class NormKD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(NormKD, self).__init__(student, teacher)
        self.temperature_norm = cfg.NormKD.TEMPERATURE_NORM
        self.ce_loss_weight = cfg.NormKD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.NormKD.LOSS.KD_WEIGHT

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_kd = self.kd_loss_weight * kd_loss_normal(
            logits_student, logits_teacher, self.temperature_norm 
        )
        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_kd,
        }
        return logits_student, losses_dict
