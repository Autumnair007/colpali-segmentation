"""Small residual adapter for frozen multi-vector document embeddings."""
from __future__ import annotations

import torch
import torch.nn.functional as F


class ResidualEmbeddingAdapter(torch.nn.Module):
    """Token-wise residual MLP adapter that preserves the ColQwen2 embedding dim."""

    def __init__(self, dim: int = 128, hidden_dim: int = 256, init_scale: float = 0.1):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.norm = torch.nn.LayerNorm(dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_dim, dim),
        )
        self.scale = torch.nn.Parameter(torch.tensor(float(init_scale)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        first, _, second = self.mlp
        torch.nn.init.xavier_uniform_(first.weight)
        torch.nn.init.zeros_(first.bias)
        torch.nn.init.zeros_(second.weight)
        torch.nn.init.zeros_(second.bias)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.shape[-1] != self.dim:
            raise ValueError(f"Expected embedding dim {self.dim}, got {embeddings.shape[-1]}.")
        residual_scale = torch.clamp(self.scale, min=0.0, max=0.5)
        adapted = embeddings.float() + residual_scale * self.mlp(self.norm(embeddings.float()))
        return F.normalize(adapted, p=2, dim=-1)


def apply_adapter_to_embeddings(
    adapter: ResidualEmbeddingAdapter,
    embeddings: list[torch.Tensor],
    device: str | torch.device,
) -> list[torch.Tensor]:
    return [adapter(embedding.to(device)) for embedding in embeddings]
