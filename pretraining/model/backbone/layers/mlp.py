from typing import Optional

from torch import nn


def mlp(
    in_size: int,
    hidden_size: int,
    out_size: Optional[int] = None,
    num_layers: int = 2,
    norm: Optional[str] = None,
    activation: Optional[str] = "gelu",
    bias: bool = True,
    dropout: float = 0.0,
) -> nn.Module:
    assert norm is None or norm in ["layer_norm", "batch_norm"]
    assert activation is None or activation in ["gelu", "relu"]

    norm_layer = (
        nn.LayerNorm
        if norm == "layer_norm"
        else nn.BatchNorm1d
        if norm == "batch_norm"
        else nn.Identity
    )
    activation_layer = (
        nn.GELU
        if activation == "gelu"
        else nn.ReLU
        if activation == "relu"
        else nn.Identity
    )

    layers = [
        nn.Linear(in_size, hidden_size, bias=bias),
        norm_layer(hidden_size),
        activation_layer(),
    ]

    if num_layers > 1:
        num_hidden_layers = num_layers - 2 if out_size is not None else num_layers - 1
        for _ in range(num_hidden_layers):
            layers += [
                nn.Linear(hidden_size, hidden_size, bias=bias),
                norm_layer(hidden_size),
                activation_layer(),
            ]
        if out_size is not None:
            layers.append(nn.Linear(hidden_size, out_size, bias=bias))

    if dropout > 0:
        layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)
