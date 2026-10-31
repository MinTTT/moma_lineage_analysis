"""PyTorch LSTM, CNN, and residual CNN trajectory labelers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Any

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass(frozen=True)
class ModelConfig:
    model_type: str = "resnet"
    # Round R002: the extended feature layout has five channels (see
    # ``trajectory_cleaning.data.FEATURE_COLUMNS``). Old two-channel checkpoints
    # restore ``input_size=2`` from their saved config.
    input_size: int = 5
    output_size: int = 2
    hidden_size: int = 64
    num_layers: int = 4
    kernel_size: int = 5
    dropout: float = 0.15
    bidirectional: bool = True
    conv_dilations: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.model_type not in {"cnn", "resnet", "lstm"}:
            raise ValueError("model_type must be one of: cnn, resnet, lstm")
        if self.input_size <= 0 or self.output_size <= 0 or self.hidden_size <= 0:
            raise ValueError("Model sizes must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.conv_dilations is not None:
            if self.model_type == "lstm":
                raise ValueError("conv_dilations is only supported for cnn and resnet")
            if not isinstance(self.conv_dilations, (tuple, list)):
                raise ValueError("conv_dilations must be a tuple or list of positive integers")
            if len(self.conv_dilations) != self.num_layers:
                raise ValueError("conv_dilations must contain exactly num_layers values")
            if any(
                isinstance(dilation, bool) or not isinstance(dilation, Integral) or dilation <= 0
                for dilation in self.conv_dilations
            ):
                raise ValueError("conv_dilations values must be positive integers (not bool)")
            # JSON checkpoints decode tuples as lists; keep one canonical config type.
            object.__setattr__(self, "conv_dilations", tuple(int(d) for d in self.conv_dilations))

    @property
    def effective_conv_dilations(self) -> tuple[int, ...] | None:
        """Resolve the historical defaults without changing serialized configs."""
        if self.model_type == "lstm":
            return None
        if self.conv_dilations is not None:
            return self.conv_dilations
        if self.model_type == "cnn":
            return (1,) * self.num_layers
        return tuple(2 ** (index % 4) for index in range(self.num_layers))

    @property
    def receptive_field(self) -> int | None:
        """Width in normalized feature frames; LSTM context is sequence-wide."""
        dilations = self.effective_conv_dilations
        if dilations is None:
            return None
        convolutions_per_layer = 2 if self.model_type == "resnet" else 1
        return 1 + convolutions_per_layer * (self.kernel_size - 1) * sum(dilations)

    @property
    def temporal_offsets(self) -> tuple[int, ...] | None:
        """Exact structural offsets via kernel-support sums, independent of weights.

        Residual identities add no new offsets: odd kernels already include zero.
        This describes interior network support, not trajectory normalization.
        """
        dilations = self.effective_conv_dilations
        if dilations is None:
            return None
        offsets = {0}
        radius = self.kernel_size // 2
        convolutions_per_layer = 2 if self.model_type == "resnet" else 1
        for dilation in dilations:
            kernel_offsets = range(-radius * dilation, radius * dilation + 1, dilation)
            for _ in range(convolutions_per_layer):
                offsets = {left + right for left in offsets for right in kernel_offsets}
        return tuple(sorted(offsets))

    @property
    def has_dense_temporal_support(self) -> bool | None:
        offsets = self.temporal_offsets
        return None if offsets is None else len(offsets) == self.receptive_field

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ModelConfig":
        return cls(**values)


class ChannelLayerNorm(nn.Module):
    """Layer normalization over channels independently at each time point."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.normalization(values.transpose(1, 2)).transpose(1, 2)


class CNNLabeler(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.conv_dilations = config.effective_conv_dilations
        self.receptive_field = config.receptive_field
        self.temporal_offsets = config.temporal_offsets
        self.has_dense_temporal_support = config.has_dense_temporal_support
        layers: list[nn.Module] = []
        in_channels = config.input_size
        for dilation in self.conv_dilations:
            padding = dilation * (config.kernel_size - 1) // 2
            layers.append(
                nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        config.hidden_size,
                        kernel_size=config.kernel_size,
                        padding=padding,
                        dilation=dilation,
                    ),
                    ChannelLayerNorm(config.hidden_size),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                )
            )
            in_channels = config.hidden_size
        self.encoder = nn.ModuleList(layers)
        self.head = nn.Conv1d(config.hidden_size, config.output_size, kernel_size=1)

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        values = features.transpose(1, 2)
        channel_mask = mask.unsqueeze(1) if mask is not None else None
        if channel_mask is not None:
            values = values * channel_mask
        for layer in self.encoder:
            values = layer(values)
            if channel_mask is not None:
                values = values * channel_mask
        return self.head(values).transpose(1, 2)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.convolution_1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.normalization_1 = ChannelLayerNorm(channels)
        self.convolution_2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.normalization_2 = ChannelLayerNorm(channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        values: torch.Tensor,
        channel_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = values
        output = self.convolution_1(values)
        output = self.dropout(self.activation(self.normalization_1(output)))
        if channel_mask is not None:
            output = output * channel_mask
        output = self.normalization_2(self.convolution_2(output))
        output = self.activation(residual + output)
        if channel_mask is not None:
            output = output * channel_mask
        return output


class ResNetLabeler(nn.Module):
    """A compact non-causal 1D residual network with dilated context."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.conv_dilations = config.effective_conv_dilations
        self.receptive_field = config.receptive_field
        self.temporal_offsets = config.temporal_offsets
        self.has_dense_temporal_support = config.has_dense_temporal_support
        self.input_projection = nn.Conv1d(config.input_size, config.hidden_size, kernel_size=1)
        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    config.hidden_size,
                    config.kernel_size,
                    dilation=dilation,
                    dropout=config.dropout,
                )
                for dilation in self.conv_dilations
            ]
        )
        self.head = nn.Conv1d(config.hidden_size, config.output_size, kernel_size=1)

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        values = self.input_projection(features.transpose(1, 2))
        channel_mask = mask.unsqueeze(1) if mask is not None else None
        if channel_mask is not None:
            values = values * channel_mask
        for block in self.blocks:
            values = block(values, channel_mask)
        return self.head(values).transpose(1, 2)


class LSTMLabeler(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            bidirectional=config.bidirectional,
        )
        output_size = config.hidden_size * (2 if config.bidirectional else 1)
        self.head = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(output_size, config.output_size),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            output, _ = self.lstm(features)
        else:
            lengths = mask.sum(dim=1).to(dtype=torch.long, device="cpu")
            packed = pack_padded_sequence(
                features,
                lengths,
                batch_first=True,
                enforce_sorted=False,
            )
            packed_output, _ = self.lstm(packed)
            output, _ = pad_packed_sequence(
                packed_output,
                batch_first=True,
                total_length=features.shape[1],
            )
        return self.head(output)


def build_model(config: ModelConfig | dict[str, Any]) -> nn.Module:
    if isinstance(config, dict):
        config = ModelConfig.from_dict(config)
    if config.model_type == "cnn":
        return CNNLabeler(config)
    if config.model_type == "resnet":
        return ResNetLabeler(config)
    return LSTMLabeler(config)
