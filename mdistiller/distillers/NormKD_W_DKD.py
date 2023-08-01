import torch
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

def dkd_loss_norm(logits_student, logits_teacher, target, alpha, beta, t_norm):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)

    #norm
    tstd=logits_teacher.std(dim=1,keepdim=True)
    sstd=logits_student.std(dim=1,keepdim=True)
    dywt=tstd*t_norm
    dyws=sstd*t_norm
    rt=(logits_teacher)/dywt
    rs=(logits_student)/dyws

    pred_student = F.softmax(rs, dim=1)
    pred_teacher = F.softmax(rt, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask)

    log_pred_student = torch.log(pred_student)

    tckd_loss = (
       (F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1,keepdim=True)*(dywt**2)).mean()
    )
    pred_teacher_part2 = F.softmax(
        rt - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        rs - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        (F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction="none").sum(1,keepdim=True)*(dywt**2)).mean()
    )

    return alpha*tckd_loss + beta*nckd_loss  



class NormKD_W_DKD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(NormKD_W_DKD, self).__init__(student, teacher)
        self.ce_loss_weight = cfg.NormKD_W_DKD.CE_WEIGHT
        self.alpha = cfg.NormKD_W_DKD.ALPHA
        self.beta = cfg.NormKD_W_DKD.BETA
        self.t_norm = cfg.NormKD_W_DKD.TEMPERATURE_NORM
        self.warmup = cfg.NormKD_W_DKD.WARMUP

    def forward_train(self, image, target, **kwargs):
        logits_student, _ = self.student(image)
        with torch.no_grad():
            logits_teacher, _ = self.teacher(image)

        # losses
        loss_ce = self.ce_loss_weight * F.cross_entropy(logits_student, target)
        loss_dkd = min(kwargs["epoch"] / self.warmup, 1.0) * dkd_loss_norm(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.t_norm,
        )

        losses_dict = {
            "loss_ce": loss_ce,
            "loss_kd": loss_dkd,
        }
        return logits_student, losses_dict
