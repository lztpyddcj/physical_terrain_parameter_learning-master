"""Evaluate the bundled friction decoder on a legacy RNNDataset PT file."""

import argparse
import math
import pickle
import types
from collections import OrderedDict

import torch

from physical_decoder import ModelFactory, ModelParams
from physical_decoder.deployment_wrapper import load_decoder_deploying


class LegacyRNNDataset:
    """Receive saved dataset attributes without importing the old training project."""


class CompatibleUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("legged_gym.Decoder_training.Decoder_training_seperateData", "RNNDataset"):
                LegacyRNNDataset,
            ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
            ("torch", "FloatStorage"): torch.FloatStorage,
            ("torch", "Size"): torch.Size,
            ("collections", "OrderedDict"): OrderedDict,
            ("torch.distributions.normal", "Normal"): torch.distributions.Normal,
        }
        if (module, name) not in allowed:
            raise pickle.UnpicklingError("Unsupported type: {}.{}".format(module, name))
        return allowed[(module, name)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="Path to the legacy validation PT file")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--checkpoint", help="Evaluate this PTH instead of the bundled model")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; check the container or use --device cpu")

    config = ModelParams(device=args.device, input_type="pro+exte", output_type="fric",
                         mode="parallel", reset_hidden_each_epoch=True).to_dict()
    if args.checkpoint:
        model = ModelFactory.create_model(config)
        model.load(args.checkpoint)
        model.eval()
        print("Loaded checkpoint: {}".format(args.checkpoint), flush=True)
    else:
        model = load_decoder_deploying(config, "fric")
    print("Loading validation data on CPU...", flush=True)
    dataset = torch.load(
        args.dataset, map_location="cpu", weights_only=False,
        pickle_module=types.SimpleNamespace(
            __name__="compat_pickle", Unpickler=CompatibleUnpickler
        ),
    )
    if not isinstance(dataset, LegacyRNNDataset):
        raise ValueError("Expected a legacy RNNDataset")
    if (dataset.output_type != "fric" or dataset.input_type != "pro+exte"
            or not dataset.use_rnn):
        raise ValueError("Expected an RNN pro+exte friction dataset")
    if not isinstance(dataset.data, list) or not dataset.data:
        raise ValueError("Dataset must contain a nonempty sample list")

    print("Checking all {} samples...".format(len(dataset.data)), flush=True)
    for index, sample in enumerate(dataset.data):
        if not isinstance(sample, (tuple, list)) or len(sample) != 2:
            raise ValueError("Invalid sample pair at index {}".format(index))
        x, y = sample
        for tensor, shape in ((x, (config["seq_length"], 441)),
                              (y, (config["seq_length"], 4))):
            if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape
                    or tensor.dtype != torch.float32):
                raise ValueError("Invalid shape or dtype at sample {}".format(index))
            if not torch.isfinite(tensor).all():
                raise ValueError("Non-finite data at sample {}".format(index))

    # Aggregate by scalar count, including the smaller final batch.
    totals = {key: [0.0, 0.0, 0] for key in ("all_steps", "last_step")}
    with torch.inference_mode():
        for start in range(0, len(dataset.data), args.batch_size):
            samples = dataset.data[start:start + args.batch_size]
            x = torch.stack([sample[0] for sample in samples]).to(args.device)
            y = torch.stack([sample[1] for sample in samples]).to(args.device)
            prediction = model.predict(x)
            if prediction.shape != y.shape or not torch.isfinite(prediction).all():
                raise ValueError("Invalid model output at sample {}".format(start))
            error = (prediction - y).double()
            for key, values in (("all_steps", error), ("last_step", error[:, -1])):
                totals[key][0] += values.abs().sum().item()
                totals[key][1] += values.square().sum().item()
                totals[key][2] += values.numel()
            if start == 0 or (start // args.batch_size + 1) % 100 == 0:
                print("Evaluated {}/{} samples".format(
                    min(start + args.batch_size, len(dataset.data)), len(dataset.data)
                ), flush=True)

    print("\nEvaluation complete: {} samples; raw, unclipped predictions".format(
        len(dataset.data)))
    for key, (absolute, squared, count) in totals.items():
        print("{}: MAE={:.6f}, RMSE={:.6f}, scalar_count={}".format(
            key, absolute / count, math.sqrt(squared / count), count))
    print("last_step matches the deployment output position; all_steps uses the full window.")


if __name__ == "__main__":
    main()
