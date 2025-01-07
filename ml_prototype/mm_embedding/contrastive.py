import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for self-supervised learning.
    """
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor):
        """
        Compute contrastive loss, plus diagonal/off-diagonal *cosine* similarity.

        Returns:
            total_loss: Contrastive loss (CrossEntropy).
            diag_cos_sim: Average diagonal cosine similarity (z_i with z_j of same index).
            off_diag_cos_sim: Average off-diagonal cosine similarity.
        """
        # 1) Normalize embeddings for both the loss and for the cos-sim measurement
        z_i_normalized = F.normalize(z_i, dim=1)  # shape: [B, D]
        z_j_normalized = F.normalize(z_j, dim=1)  # shape: [B, D]

        # 2) Contrastive logits (scaled by temperature)
        logits = torch.matmul(z_i_normalized, z_j_normalized.T) / self.temperature
        batch_size = z_i.size(0)

        # 3) Create ground-truth labels
        labels = torch.arange(batch_size, device=z_i.device)  # e.g., [0, 1, 2, ..., B-1]

        # 4) Symmetric cross-entropy loss
        loss_i = F.cross_entropy(logits, labels, reduction='mean')
        loss_j = F.cross_entropy(logits.T, labels, reduction='mean')
        total_loss = 0.5 * (loss_i + loss_j)

        # 5) For logging: raw cosine similarity (unscaled)
        #    We'll compute it on the same normalized embeddings, but *without* dividing by temperature
        cos_sim_matrix = torch.matmul(z_i_normalized, z_j_normalized.T)  # shape: [B, B]
        diag_indices = torch.eye(batch_size, dtype=torch.bool, device=z_i.device)

        diag_cos_sim = cos_sim_matrix[diag_indices].mean()       # mean of diagonal
        off_diag_cos_sim = cos_sim_matrix[~diag_indices].mean()  # mean of off-diagonal

        return total_loss, diag_cos_sim.detach(), off_diag_cos_sim.detach()
