"""PyTorch 1.3/CUDA worker for PianoFlow v4.3.1 inference.

This file intentionally avoids importing the main PianoFlow modules: the
GTX-TITAN-compatible environment is Python 3.7 / PyTorch 1.3, while the main
application uses a newer Python/PyTorch stack.
"""
from __future__ import print_function

import argparse
import os
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def gelu_compat(value):
    return F.gelu(value)


class TemporalBlock(nn.Module):
    def __init__(self, channels, dilation, dropout):
        super(TemporalBlock, self).__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size=5, padding=2 * dilation,
            dilation=dilation, groups=channels
        )
        self.pointwise = nn.Sequential(
            nn.Conv1d(channels, channels * 3, 1),
            GeluCompat(),
            nn.Dropout(dropout),
            nn.Conv1d(channels * 3, channels, 1),
        )

    def forward(self, value):
        return value + self.pointwise(self.depthwise(self.norm(value)))


class GeluCompat(nn.Module):
    def forward(self, value):
        return gelu_compat(value)


class HeavyAcousticModel(nn.Module):
    def __init__(self, feature_count, output_count, channels, transformer_layers):
        super(HeavyAcousticModel, self).__init__()
        self.input = nn.Sequential(
            nn.Linear(feature_count, channels),
            nn.LayerNorm(channels),
            GeluCompat(),
        )
        self.temporal = nn.Sequential(*[
            TemporalBlock(channels, dilation, 0.18)
            for dilation in (1, 2, 4, 8, 16, 32, 1, 2)
        ])
        # PyTorch 1.3 lacks batch_first and the "gelu" activation option.
        # Transposing around the encoder preserves the trained tensor layout;
        # assigning F.gelu changes no state_dict keys or parameter values.
        layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=8,
            dim_feedforward=channels * 4,
            dropout=0.16,
            activation="relu",
        )
        layer.activation = gelu_compat
        self.transformer = nn.TransformerEncoder(layer, transformer_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels),
            GeluCompat(),
            nn.Dropout(0.15),
            nn.Linear(channels, output_count),
        )

    def forward(self, features):
        encoded = self.input(features)
        encoded = self.temporal(encoded.transpose(1, 2)).transpose(1, 2)
        encoded = self.transformer(encoded.transpose(0, 1)).transpose(0, 1)
        return self.head(encoded)


class PriorResidualAcousticModel(nn.Module):
    def __init__(self, task, feature_count, output_count, channels, transformer_layers):
        super(PriorResidualAcousticModel, self).__init__()
        self.task = task
        self.residual = HeavyAcousticModel(
            feature_count, output_count, channels, transformer_layers
        )
        self.gate_logit = nn.Parameter(torch.tensor(-2.20))
        self.maximum_logit_correction = 8.0

    def prior(self, features):
        if self.task == "pitch":
            return features[..., 134:220]
        if self.task == "ornament":
            return features[..., 220:223]
        return features[..., 223:227]

    def forward(self, features):
        probability = self.prior(features).clamp(1e-5, 1 - 1e-5)
        if self.task == "boundary":
            base_logits = torch.log(probability) - torch.log1p(-probability)
        else:
            probability = probability / probability.sum(-1, keepdim=True).clamp_min(1e-7)
            base_logits = torch.log(probability)
        residual = self.residual(features)
        bounded = self.maximum_logit_correction * torch.tanh(
            residual / self.maximum_logit_correction
        )
        gate = torch.sigmoid(self.gate_logit)
        return base_logits + gate * bounded


def check_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch 1.3 からCUDAデバイスが見つかりません")
    # Exercise a real CUDA kernel, not just torch.cuda.is_available().
    value = torch.ones((1, 1, 8, 8), device="cuda")
    layer = nn.Conv2d(1, 1, 3).cuda()
    result = layer(value)
    if tuple(result.shape) != (1, 1, 6, 6):
        raise RuntimeError("CUDA演算の出力形状が一致しません")
    return torch.cuda.get_device_name(0)


def run_inference(request_path, output_path, model_dir):
    gpu_name = check_cuda()
    request = np.load(request_path)
    features = np.asarray(request["features"], dtype=np.float32)
    mean = np.asarray(request["mean"], dtype=np.float32)
    scale = np.asarray(request["scale"], dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != len(mean) or len(mean) != len(scale):
        raise ValueError("GPU workerへ渡された特徴量の形が不正です")
    normalized = (features - mean) / scale
    outputs = {}
    for task in ("pitch", "ornament", "boundary"):
        checkpoint = os.path.join(model_dir, "current-" + task + ".pt")
        payload = torch.load(checkpoint, map_location="cpu")
        model = PriorResidualAcousticModel(
            task,
            int(payload["featureCount"]),
            int(payload["outputCount"]),
            int(payload["channels"]),
            int(payload["transformerLayers"]),
        )
        model.load_state_dict(payload["bestState"])
        model.cuda().eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(normalized), 256):
                batch = torch.from_numpy(normalized[start:start + 256][None]).cuda()
                logits = model(batch)
                if task == "boundary":
                    probability = torch.sigmoid(logits)
                else:
                    probability = torch.softmax(logits, -1)
                chunks.append(probability[0].cpu().numpy())
        outputs[task] = np.concatenate(chunks, axis=0) if chunks else np.empty((0, int(payload["outputCount"])), dtype=np.float32)
        del model
        del payload
        torch.cuda.empty_cache()
    np.savez(output_path, **outputs)
    print("GPU_DEVICE=" + gpu_name)
    print("GPU_FRAMES=" + str(len(features)))


def check_model_bundle(model_dir):
    gpu_name = check_cuda()
    for task in ("pitch", "ornament", "boundary"):
        checkpoint = os.path.join(model_dir, "current-" + task + ".pt")
        payload = torch.load(checkpoint, map_location="cpu")
        model = PriorResidualAcousticModel(
            task,
            int(payload["featureCount"]),
            int(payload["outputCount"]),
            int(payload["channels"]),
            int(payload["transformerLayers"]),
        )
        model.load_state_dict(payload["bestState"])
        model.cuda().eval()
        dummy = torch.zeros((1, 8, int(payload["featureCount"])), device="cuda")
        dummy[:, :, 134:227] = 0.01
        if task == "pitch":
            dummy[:, :, 134] = 0.14
            dummy[:, :, 135:220] = 0.01
        elif task == "ornament":
            dummy[:, :, 220] = 0.34
            dummy[:, :, 221:223] = 0.33
        else:
            dummy[:, :, 223:227] = 0.25
        with torch.no_grad():
            result = model(dummy)
        if tuple(result.shape) != (1, 8, int(payload["outputCount"])):
            raise RuntimeError(task + "モデルの出力形状が一致しません")
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError(task + "モデルのCUDA出力にNaN/Infがあります")
        print("MODEL_OK=" + task)
        del result
        del dummy
        del model
        del payload
        torch.cuda.empty_cache()
    print("GPU_DEVICE=" + gpu_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--check-models", action="store_true")
    parser.add_argument("--request")
    parser.add_argument("--output")
    parser.add_argument("--model-dir")
    args = parser.parse_args()
    try:
        if args.self_test:
            print("GPU_DEVICE=" + check_cuda())
            return 0
        if args.check_models:
            if not args.model_dir:
                raise ValueError("--model-dir が必要です")
            check_model_bundle(args.model_dir)
            return 0
        if not args.request or not args.output or not args.model_dir:
            raise ValueError("--request / --output / --model-dir が必要です")
        run_inference(args.request, args.output, args.model_dir)
        return 0
    except Exception as error:
        print("GPU_WORKER_ERROR=" + str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
