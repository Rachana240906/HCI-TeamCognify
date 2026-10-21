"""
model.py
========
GRU-based EEG classifier for CD-B2M.

Architecture
------------
Input  : (batch, n_channels, seq_len)  — EEG epoch, channels as features
GRU    : bidirectional GRU, n_layers
Output : (batch, n_classes)            — logits

Projection head (for brain-guided training only)
-------------------------------------------------
An additional linear layer maps the final GRU hidden state to the
brain-representation space so that L_brain = MSE(proj(h), b) can be
computed without altering the classification head.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class EEGEncoder(nn.Module):
    """
    Bidirectional GRU encoder.

    Parameters
    ----------
    input_size   : number of EEG channels (features per timestep)
    hidden_size  : GRU hidden units per direction
    num_layers   : number of stacked GRU layers
    dropout      : dropout between GRU layers (0 if num_layers==1)
    bidirectional: whether to use bidirectional GRU
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
    ):
        super().__init__()
        self.hidden_size  = hidden_size
        self.num_layers   = num_layers
        self.bidirectional = bidirectional
        self.n_directions  = 2 if bidirectional else 1

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

    @property
    def output_dim(self) -> int:
        return self.hidden_size * self.n_directions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, channels, time)

        Returns
        -------
        h : (batch, hidden_size * n_directions)  — final hidden state
        """
        # GRU expects (batch, seq_len, input_size)
        # We treat time as seq_len and channels as features
        x = x.permute(0, 2, 1)            # → (batch, time, channels)
        _, hidden = self.gru(x)            # hidden: (layers*dirs, batch, H)
        # Take the last layer's hidden state, concatenate forward + backward
        if self.bidirectional:
            h_fwd = hidden[-2]             # (batch, H) — last layer, forward
            h_bwd = hidden[-1]             # (batch, H) — last layer, backward
            h = torch.cat([h_fwd, h_bwd], dim=-1)   # (batch, 2H)
        else:
            h = hidden[-1]                 # (batch, H)
        return h                           # (batch, output_dim)


class EEGClassifier(nn.Module):
    """
    Full classification model: EEGEncoder + classification head.

    Also exposes a projection head for brain-guided training.

    Parameters
    ----------
    input_size   : EEG channels
    n_classes    : number of output classes
    hidden_size  : GRU hidden units per direction
    num_layers   : GRU layers
    dropout      : GRU + classifier dropout
    bidirectional: bidirectional GRU
    proj_dim     : dimension of the brain-rep projection head
                   (set to brain_dim when using brain guidance)
    brain_dim    : actual brain representation dimension
                   (only needed when proj_dim is set)
    """

    def __init__(
        self,
        input_size: int,
        n_classes: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        proj_dim: int | None = None,
        brain_dim: int | None = None,
    ):
        super().__init__()
        self.encoder = EEGEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        enc_dim = self.encoder.output_dim

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(enc_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, n_classes),
        )

        # Optional brain-projection head (used only during brain-guided training)
        self.proj_head: nn.Module | None = None
        if proj_dim is not None and brain_dim is not None:
            self.proj_head = nn.Sequential(
                nn.Linear(enc_dim, proj_dim),
                nn.ReLU(),
                nn.Linear(proj_dim, brain_dim),
            )

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Parameters
        ----------
        x : (batch, channels, time)

        Returns
        -------
        logits : (batch, n_classes)
        proj   : (batch, brain_dim) or None if no projection head
        """
        h = self.encoder(x)
        logits = self.classifier(h)
        proj = self.proj_head(h) if self.proj_head is not None else None
        return logits, proj

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class predictions (no gradient)."""
        with torch.no_grad():
            logits, _ = self.forward(x)
        return logits.argmax(dim=-1)


def build_model(cfg: dict, n_classes: int, brain_dim: int | None = None) -> EEGClassifier:
    """
    Construct an EEGClassifier from the config dict.

    Parameters
    ----------
    cfg       : full config dict (uses cfg['model'] section)
    n_classes : number of output classes
    brain_dim : brain representation dimension; if provided, adds proj_head
    """
    m = cfg["model"]
    proj_dim = m.get("proj_hidden") if brain_dim is not None else None
    return EEGClassifier(
        input_size=m["input_size"],
        n_classes=n_classes,
        hidden_size=m["hidden_size"],
        num_layers=m["num_layers"],
        dropout=m["dropout"],
        bidirectional=m["bidirectional"],
        proj_dim=proj_dim,
        brain_dim=brain_dim,
    )
