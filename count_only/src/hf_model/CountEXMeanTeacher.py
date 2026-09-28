"""
CountEXMeanTeacher: Mean Teacher wrapper for CountEXStage2.

Wraps two CountEXStage2 models (student + teacher):
- Student: trainable, receives gradient updates
- Teacher: frozen (no grad), updated via EMA from student weights

EMA update rule:
    θ_teacher = α × θ_teacher + (1 - α) × θ_student
    where α = 0.999

Reference: Mean teachers are better role models (Tarvainen & Valpola, 2017)
"""

import copy
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, List, Tuple

from transformers.utils import ModelOutput
from .CountEXStage2 import CountEXStage2
from .CountEXWithDetectionPrior import DensityWithPriorOutput


@dataclass
class MeanTeacherOutput(ModelOutput):
    """Output containing both student and teacher predictions."""
    # Student outputs
    student_density_map: torch.FloatTensor = None
    student_pred_count: torch.FloatTensor = None
    student_detection_prior: torch.FloatTensor = None
    student_raw_density: torch.FloatTensor = None
    student_pred_boxes: torch.FloatTensor = None
    student_pred_scores: torch.FloatTensor = None
    student_logits: Optional[torch.FloatTensor] = None
    student_pos_features: Optional[List[torch.FloatTensor]] = None
    student_neg_features: Optional[List[torch.FloatTensor]] = None
    # Teacher outputs (detached, no grad)
    teacher_density_map: torch.FloatTensor = None
    teacher_pred_count: torch.FloatTensor = None
    teacher_detection_prior: torch.FloatTensor = None
    teacher_raw_density: torch.FloatTensor = None
    teacher_pred_boxes: torch.FloatTensor = None
    teacher_pred_scores: torch.FloatTensor = None
    teacher_logits: Optional[torch.FloatTensor] = None


class CountEXMeanTeacher(nn.Module):
    """
    Mean Teacher wrapper around CountEXStage2.

    Contains a student and a teacher model. The student is trained normally,
    and the teacher is updated via exponential moving average (EMA) of the
    student's parameters. The teacher is completely frozen (no gradients).
    """

    def __init__(self, student: CountEXStage2, ema_decay: float = 0.999):
        super().__init__()
        self.student = student
        self.teacher = copy.deepcopy(student)
        self.ema_decay = ema_decay

        # Freeze teacher completely
        for param in self.teacher.parameters():
            param.requires_grad = False

        self._print_info()

    def _print_info(self):
        student_params = sum(p.numel() for p in self.student.parameters())
        student_trainable = sum(p.numel() for p in self.student.parameters() if p.requires_grad)
        teacher_params = sum(p.numel() for p in self.teacher.parameters())
        teacher_trainable = sum(p.numel() for p in self.teacher.parameters() if p.requires_grad)
        print("=" * 60)
        print("CountEXMeanTeacher")
        print("=" * 60)
        print(f"  Student: {student_trainable:,} / {student_params:,} trainable")
        print(f"  Teacher: {teacher_trainable:,} / {teacher_params:,} trainable (should be 0)")
        print(f"  EMA decay: {self.ema_decay}")
        print("=" * 60)

    @torch.no_grad()
    def ema_update(self):
        """Update teacher parameters via EMA from student parameters."""
        for s_param, t_param in zip(self.student.parameters(), self.teacher.parameters()):
            t_param.data.mul_(self.ema_decay).add_(s_param.data, alpha=1.0 - self.ema_decay)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor,
        token_type_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        pixel_mask: Optional[torch.LongTensor] = None,
        encoder_outputs: Optional[Tuple] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> MeanTeacherOutput:
        # Student forward (with grad)
        student_out = self.student(
            pixel_values=pixel_values,
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            pixel_mask=pixel_mask,
            encoder_outputs=encoder_outputs,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            **kwargs,
        )

        # Teacher forward (no grad)
        with torch.no_grad():
            teacher_out = self.teacher(
                pixel_values=pixel_values,
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                pixel_mask=pixel_mask,
                encoder_outputs=encoder_outputs,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                **kwargs,
            )

        return MeanTeacherOutput(
            # Student
            student_density_map=student_out.density_map,
            student_pred_count=student_out.pred_count,
            student_detection_prior=student_out.detection_prior,
            student_raw_density=student_out.raw_density,
            student_pred_boxes=student_out.pred_boxes,
            student_pred_scores=student_out.pred_scores,
            student_logits=student_out.logits,
            student_pos_features=student_out.pos_features,
            student_neg_features=student_out.neg_features,
            # Teacher
            teacher_density_map=teacher_out.density_map,
            teacher_pred_count=teacher_out.pred_count,
            teacher_detection_prior=teacher_out.detection_prior,
            teacher_raw_density=teacher_out.raw_density,
            teacher_pred_boxes=teacher_out.pred_boxes,
            teacher_pred_scores=teacher_out.pred_scores,
            teacher_logits=teacher_out.logits,
        )
