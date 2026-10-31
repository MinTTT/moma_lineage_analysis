"""CPU regressions for dilation, structural context, padding, and old checkpoints."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from moma_lineage_analysis.trajectory_cleaning import cli
from moma_lineage_analysis.trajectory_cleaning.data import TrajectoryNormalizer
from moma_lineage_analysis.trajectory_cleaning.models import (
    ChannelLayerNorm,
    ModelConfig,
    build_model,
)
from moma_lineage_analysis.trajectory_cleaning.training import load_checkpoint, save_checkpoint


SCHEDULES = [
    (17, (1, 1, 1, 1)),
    (33, (1, 1, 2, 4)),
    (49, (1, 2, 3, 6)),
    (65, (1, 2, 5, 8)),
    (97, (1, 3, 7, 13)),
    (121, (1, 3, 8, 18)),
    (161, (1, 4, 11, 24)),
    (197, (1, 4, 12, 32)),
]


def _legacy_model(config: ModelConfig) -> nn.Module:
    """Construct the previous architecture explicitly, preserving its parameter order."""
    model = nn.Module()
    if config.model_type == "cnn":
        layers = []
        channels = config.input_size
        for _ in range(config.num_layers):
            layers.append(nn.Sequential(
                nn.Conv1d(channels, config.hidden_size, config.kernel_size,
                          padding=config.kernel_size // 2),
                ChannelLayerNorm(config.hidden_size), nn.GELU(), nn.Dropout(config.dropout),
            ))
            channels = config.hidden_size
        model.encoder = nn.ModuleList(layers)
    else:
        model.input_projection = nn.Conv1d(config.input_size, config.hidden_size, 1)
        blocks = []
        for index in range(config.num_layers):
            block = nn.Module()
            dilation = 2 ** (index % 4)
            padding = dilation * (config.kernel_size - 1) // 2
            block.convolution_1 = nn.Conv1d(
                config.hidden_size, config.hidden_size, config.kernel_size,
                padding=padding, dilation=dilation,
            )
            block.normalization_1 = ChannelLayerNorm(config.hidden_size)
            block.convolution_2 = nn.Conv1d(
                config.hidden_size, config.hidden_size, config.kernel_size,
                padding=padding, dilation=dilation,
            )
            block.normalization_2 = ChannelLayerNorm(config.hidden_size)
            block.activation = nn.GELU()
            block.dropout = nn.Dropout(config.dropout)
            blocks.append(block)
        model.blocks = nn.ModuleList(blocks)
    model.head = nn.Conv1d(config.hidden_size, config.output_size, 1)
    return model.eval()


def _legacy_forward(model, model_type, features, mask):
    values = features.transpose(1, 2)
    channel_mask = None if mask is None else mask.unsqueeze(1)
    if model_type == "resnet":
        values = model.input_projection(values)
    if channel_mask is not None:
        values = values * channel_mask
    if model_type == "cnn":
        for layer in model.encoder:
            values = layer(values)
            if channel_mask is not None:
                values = values * channel_mask
    else:
        for block in model.blocks:
            output = block.convolution_1(values)
            output = block.dropout(block.activation(block.normalization_1(output)))
            if channel_mask is not None:
                output = output * channel_mask
            output = block.normalization_2(block.convolution_2(output))
            values = block.activation(values + output)
            if channel_mask is not None:
                values = values * channel_mask
    return model.head(values).transpose(1, 2)


@pytest.mark.parametrize("model_type", ["cnn", "resnet"])
@pytest.mark.parametrize("num_layers", [4, 6])
def test_legacy_default_and_explicit_schedule_outputs_are_bytewise_equal(model_type, num_layers):
    config = ModelConfig(model_type=model_type, hidden_size=8, num_layers=num_layers)
    torch.manual_seed(11)
    legacy = _legacy_model(config)
    torch.manual_seed(11)
    default = build_model(config).eval()
    torch.manual_seed(11)
    explicit = build_model(replace(
        config, conv_dilations=config.effective_conv_dilations,
    )).eval()
    assert list(default.state_dict()) == list(legacy.state_dict())
    for name, tensor in legacy.state_dict().items():
        assert torch.equal(default.state_dict()[name], tensor)
        assert torch.equal(explicit.state_dict()[name], tensor)
    features = torch.randn(2, 41, 5)
    mask = torch.arange(41).unsqueeze(0) < torch.tensor([[41], [23]])
    with torch.no_grad():
        for valid in (None, mask):
            expected = _legacy_forward(legacy, model_type, features, valid)
            assert torch.equal(default(features, valid), expected)
            assert torch.equal(explicit(features, valid), expected)


@pytest.mark.parametrize("model_type,kernel_size", [("cnn", 5), ("resnet", 3)])
@pytest.mark.parametrize("width,dilations", SCHEDULES)
def test_sweep_receptive_fields_have_dense_support(model_type, kernel_size, width, dilations):
    config = ModelConfig(
        model_type=model_type, kernel_size=kernel_size, conv_dilations=dilations,
    )
    assert config.hidden_size == 64
    assert config.receptive_field == width == 1 + 4 * sum(dilations)
    radius = width // 2
    assert config.temporal_offsets == tuple(range(-radius, radius + 1))
    assert config.has_dense_temporal_support is True
    model = build_model(config)
    assert model.receptive_field == width
    assert model.temporal_offsets == config.temporal_offsets
    assert model.has_dense_temporal_support is True
    assert model.conv_dilations == dilations
    convolutions = (
        [layer[0] for layer in model.encoder] if model_type == "cnn" else
        [conv for block in model.blocks for conv in (block.convolution_1, block.convolution_2)]
    )
    expected_dilations = (
        dilations if model_type == "cnn" else tuple(d for d in dilations for _ in range(2))
    )
    assert tuple(conv.dilation[0] for conv in convolutions) == expected_dilations
    assert all(conv.padding[0] == conv.dilation[0] * (kernel_size // 2) for conv in convolutions)


def test_legacy_contexts_and_holey_schedules_are_reported_honestly():
    assert ModelConfig(model_type="cnn").receptive_field == 17
    control = ModelConfig(model_type="resnet")
    assert control.receptive_field == 121
    assert control.effective_conv_dilations == (1, 2, 4, 8)
    assert control.temporal_offsets == tuple(range(-60, 61))
    holey = ModelConfig(model_type="cnn", kernel_size=3, num_layers=1, conv_dilations=(3,))
    assert holey.receptive_field == 7
    assert holey.temporal_offsets == (-3, 0, 3)
    assert holey.has_dense_temporal_support is False
    unit = ModelConfig(model_type="resnet", kernel_size=1, conv_dilations=(1, 3, 8, 18))
    assert unit.receptive_field == 1
    assert unit.temporal_offsets == (0,)
    assert unit.has_dense_temporal_support is True
    recurrent = ModelConfig(model_type="lstm")
    assert recurrent.receptive_field is None
    assert recurrent.temporal_offsets is None
    assert recurrent.has_dense_temporal_support is None
    assert recurrent.effective_conv_dilations is None


@pytest.mark.parametrize("model_type,kernel_size", [("cnn", 5), ("resnet", 3)])
@pytest.mark.parametrize("width,dilations", SCHEDULES)
def test_features_outside_receptive_field_cannot_change_center_logit(
    model_type, kernel_size, width, dilations,
):
    # Perturb already-normalized tensors: preprocessing can use trajectory-wide context.
    config = ModelConfig(
        model_type=model_type, hidden_size=8, kernel_size=kernel_size,
        conv_dilations=dilations, dropout=0,
    )
    torch.manual_seed(19)
    model = build_model(config).eval()
    radius = width // 2
    center = radius + 4
    features = torch.randn(1, width + 8, 5)
    perturbed = features.clone()
    outside = (torch.arange(features.shape[1]) - center).abs() > radius
    perturbed[:, outside] = 100 * torch.randn_like(perturbed[:, outside])
    assert outside.sum() == 8
    with torch.no_grad():
        original = model(features)
        altered = model(perturbed)
    assert torch.equal(original[:, center], altered[:, center])
    # A nontrivial model reacts to perturbations at locations that can see them.
    assert not torch.equal(original[:, 0], altered[:, 0])


@pytest.mark.parametrize("model_type,kernel_size", [("cnn", 5), ("resnet", 3)])
@pytest.mark.parametrize("dilations", [(1, 1, 1, 1), (1, 4, 12, 32)])
def test_masked_padding_cannot_feed_back_into_short_trajectory_boundaries(
    model_type, kernel_size, dilations,
):
    torch.manual_seed(31)
    model = build_model(ModelConfig(
        model_type=model_type, hidden_size=8, kernel_size=kernel_size,
        conv_dilations=dilations, dropout=0,
    )).double().eval()
    short_length, padded_length = 11, 75
    short = torch.randn(1, short_length, 5, dtype=torch.float64)
    batch = torch.randn(2, padded_length, 5, dtype=torch.float64)
    batch[0, :short_length] = short[0]
    mask = torch.arange(padded_length).unsqueeze(0) < torch.tensor([[short_length], [padded_length]])
    zero_padding = batch.clone()
    zero_padding[0, short_length:] = 0
    batch[0, short_length:] *= 100
    with torch.no_grad():
        unpadded = model(short, torch.ones(1, short_length, dtype=torch.bool))
        padded = model(batch, mask)
        clean = model(zero_padding, mask)
    assert torch.equal(padded[0, :short_length], clean[0, :short_length])
    torch.testing.assert_close(padded[0, :short_length], unpadded[0], rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("dilations", [
    (), (1, 2, 3), (1, 2, 3, 4, 5), (1, 2, 0, 4), (1, -2, 3, 4),
    (True, 2, 3, 4), (1, 2, 3, False), (1.0, 2, 3, 4), (1, 2, 3, "4"),
    (1, 2, 3, None), "1,2,3,4", 4, {1, 2, 3, 4},
])
def test_invalid_dilation_configs_are_rejected(dilations):
    with pytest.raises(ValueError, match="conv_dilations"):
        ModelConfig(model_type="cnn", conv_dilations=dilations)


def test_config_json_roundtrip_and_legacy_missing_field():
    values = [np.int64(1), 2, 3, 6]
    config = ModelConfig(model_type="cnn", conv_dilations=values)
    assert config.conv_dilations == (1, 2, 3, 6)
    assert all(type(d) is int for d in config.conv_dilations)
    values[0] = 99
    assert config.conv_dilations[0] == 1
    encoded = json.loads(json.dumps(config.to_dict()))
    assert encoded["conv_dilations"] == [1, 2, 3, 6]
    assert ModelConfig.from_dict(encoded) == config
    encoded.pop("conv_dilations")
    assert ModelConfig.from_dict(encoded).conv_dilations is None
    with pytest.raises(ValueError, match="only supported"):
        ModelConfig(model_type="lstm", conv_dilations=(1, 2, 3, 6))
    for bad in ([1, 2, True, 6], [1, 2, 3.0, 6], [1, 2]):
        with pytest.raises(ValueError, match="conv_dilations"):
            ModelConfig.from_dict({"model_type": "cnn", "conv_dilations": bad})


@pytest.mark.parametrize("model_type", ["cnn", "resnet"])
@pytest.mark.parametrize("explicit", [False, True])
def test_checkpoint_load_preserves_outputs_and_explicit_schedule(tmp_path, model_type, explicit):
    config = ModelConfig(
        model_type=model_type, hidden_size=8,
        conv_dilations=(1, 2, 3, 6) if explicit else None,
    )
    torch.manual_seed(47)
    model = build_model(config).eval()
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path, model, model_config=config, normalizer=TrajectoryNormalizer(),
        thresholds=(0.3, 0.7),
    )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if explicit:
        # Exercise the JSON-style list representation, too.
        payload["model_config"]["conv_dilations"] = list(config.conv_dilations)
    else:
        # Legacy checkpoints do not contain this field at all.
        payload["model_config"].pop("conv_dilations")
    torch.save(payload, path)
    loaded, _, thresholds, metadata = load_checkpoint(path, device="cpu")
    assert tuple(loaded.state_dict()) == tuple(model.state_dict())
    assert loaded.conv_dilations == config.effective_conv_dilations
    assert loaded.receptive_field == config.receptive_field
    assert metadata["model_config"]["conv_dilations"] == config.conv_dilations
    np.testing.assert_array_equal(thresholds, np.array([0.3, 0.7], dtype=np.float32))
    features = torch.randn(1, 27, 5)
    with torch.no_grad():
        assert torch.equal(loaded(features), model(features))


def test_released_legacy_checkpoint_loads_with_original_context():
    root = Path(__file__).resolve().parents[2]
    model, _, _, metadata = load_checkpoint(
        root / "src/moma_lineage_analysis/models/trajectory_cleaning_run/best_model.pt",
        device="cpu",
    )
    assert metadata["model_config"]["model_type"] == "cnn"
    assert metadata["model_config"]["conv_dilations"] is None
    assert model.receptive_field == 17
    with torch.no_grad():
        assert torch.isfinite(model(torch.zeros(1, 21, 5))).all()


@pytest.mark.parametrize("value", ["", "1,,2,3", "1,2,3.0,4", "1,2,true,4", "1,2,0,4", "1,-2,3,4"])
def test_cli_rejects_invalid_dilation_tokens(value):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([
            "train", "--input", "unused.csv", "--output-dir", "unused",
            "--conv-dilations", value,
        ])


def test_cli_explicit_config_reaches_model_checkpoint_and_metrics(tmp_path, monkeypatch):
    frame = pd.DataFrame([
        {
            "trajectory_key": f"T{trajectory}", "fov_cell_index": trajectory * 10 + step,
            "time_h": step * 0.1, "spine_length": 3.0 + step * 0.2,
            "division_point": step == 2, "excluded_point": step == 4,
        }
        for trajectory in range(6) for step in range(5)
    ])
    source = tmp_path / "synthetic.csv"
    frame.to_csv(source, index=False)
    output = tmp_path / "run"
    args = cli.build_parser().parse_args([
        "train", "--input", str(source), "--output-dir", str(output),
        "--model", "resnet", "--kernel-size", "3", "--hidden-size", "8",
        "--conv-dilations", "1, 2,3,6", "--device", "cpu",
    ])
    assert args.conv_dilations == (1, 2, 3, 6)

    def no_training(model, *datasets, **kwargs):
        assert model.conv_dilations == args.conv_dilations
        assert model.receptive_field == 49
        return model.eval(), [{"epoch": 1}], np.array([0.4, 0.6], dtype=np.float32)

    monkeypatch.setattr(cli, "train_model", no_training)
    monkeypatch.setattr(cli, "evaluate_model", lambda *args, **kwargs: {"synthetic": True})
    cli.train_command(args)
    metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["model_config"]["conv_dilations"] == [1, 2, 3, 6]
    assert metrics["model_config"]["kernel_size"] == 3
    context = metrics["temporal_context"]
    assert context["effective_conv_dilations"] == [1, 2, 3, 6]
    assert context["receptive_field"] == 49
    assert context["temporal_offsets"] == list(range(-24, 25))
    assert context["has_dense_temporal_support"] is True
    loaded, _, _, metadata = load_checkpoint(output / "best_model.pt", device="cpu")
    assert metadata["temporal_context"]["receptive_field"] == loaded.receptive_field == 49
    assert loaded.conv_dilations == (1, 2, 3, 6)
