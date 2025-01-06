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

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute the contrastive loss.

        Args:
            z_i (torch.Tensor): Embeddings of the original data. Shape: [batch_size, embedding_dim]
            z_j (torch.Tensor): Embeddings of the augmented data. Shape: [batch_size, embedding_dim]

        Returns:
            torch.Tensor: Contrastive loss.
        """
        # Normalize the embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Compute cosine similarity matrix
        similarity_matrix = torch.matmul(z_i, z_j.T) / self.temperature

        # Create labels for contrastive loss
        batch_size = z_i.size(0)
        labels = torch.arange(batch_size, device=z_i.device)

        # Use 'mean' reduction so each batch is scaled fairly
        loss_i = F.cross_entropy(similarity_matrix, labels, reduction='mean')
        loss_j = F.cross_entropy(similarity_matrix.T, labels, reduction='mean')

        # Average the two losses
        total_loss = (loss_i + loss_j) / 2.0
        return total_loss
