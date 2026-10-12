"""Train a friction decoder from H5, then evaluate a legacy PT validation set.

Uses the original sequence builder and loss, without Neptune or Lightning.
New weights and local logs are saved in a unique results directory.
"""

import argparse
import csv
import gc
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import deepdish as dd
import numpy as np
import torch
from torch.utils.data import DataLoader

from physical_decoder import ModelParams
from training_utils.data_manager import organize_data
from training_utils.loss_wrapper import LossWrapperFactory


def load_training_samples(path, sequence_length):
    print("Loading training H5...", flush=True)
    raw = dd.io.load(str(path))
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Expected a nonempty environment dictionary in the H5 file")
    dimensions = {"policy": 133, "priv_exte": 208, "hidden": 100,
                  "fric": 4, "stiff": 36}
    for env_id, sensors in raw.items():
        if not isinstance(sensors, dict):
            raise ValueError("Environment {} is not a sensor dictionary".format(env_id))
        for key in ("counter", *dimensions):
            if key not in sensors or not isinstance(sensors[key], torch.Tensor):
                raise ValueError("Environment {}: missing tensor {}".format(env_id, key))
            sensors[key] = sensors[key].detach().cpu()
            if not torch.isfinite(sensors[key]).all():
                raise ValueError("Environment {}: non-finite {}".format(env_id, key))
        counter = sensors["counter"]
        if counter.ndim == 2 and counter.shape[1] == 1:
            counter = counter[:, 0]
            sensors["counter"] = counter
        if counter.ndim != 1:
            raise ValueError("Environment {}: counter must have shape (N,) or (N, 1), got {}".format(
                env_id, tuple(counter.shape)))
        for key, width in dimensions.items():
            value = sensors[key]
            if value.shape != (len(counter), width):
                raise ValueError("Environment {}: {} has unexpected shape {}".format(
                    env_id, key, tuple(value.shape)))
            sensors[key] = value.float()
        if (sensors["fric"] < 0).any() or (sensors["fric"] > 1).any():
            raise ValueError("Environment {}: friction labels outside [0, 1]".format(env_id))
    print("Loaded {} environments; building sequences...".format(len(raw)), flush=True)
    organized = organize_data(raw, sequence_length=sequence_length, overlapp=25)
    samples = [(sample["input"], sample["output"][:, :4])
               for points in organized.values() for sample in points]
    if not samples:
        raise ValueError("No complete training sequences found")
    return samples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-h5", type=Path, required=True)
    parser.add_argument("--val-pt", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("epochs and batch-size must be positive")
    for path in (args.train_h5, args.val_pt):
        if not path.is_file():
            parser.error("File not found: {}".format(path))
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable; use --device cpu or fix GPU access")

    torch.manual_seed(42)
    np.random.seed(42)
    config = ModelParams(device=args.device, input_type="pro+exte", output_type="fric",
                         mode="parallel", reset_hidden_each_epoch=True).to_dict()
    samples = load_training_samples(args.train_h5, config["seq_length"])
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    print("Training sequences: {}".format(len(samples)), flush=True)

    # Match the original friction-bin weighting without copying all labels at once.
    histogram = np.zeros(10, dtype=np.float64)
    for _, label in samples:
        histogram += np.histogram(label.numpy(), bins=np.arange(0, 1.1, 0.1))[0]
    weights = 1.0 / (histogram + 1e-8)
    weights /= weights.sum()
    config["label_weights"] = torch.tensor(weights, device=args.device)
    config["label_weighting_bin_interval"] = 0.1
    wrapper = LossWrapperFactory.create_loss_wrapper(config).to(args.device)
    optimizer = torch.optim.Adam(wrapper.parameters(), lr=0.001, weight_decay=0.00001)
    loader = DataLoader(samples, batch_size=args.batch_size, shuffle=True, num_workers=0)

    root = Path(__file__).resolve().parents[1] / "results" / "physical_decoder"
    root.mkdir(parents=True, exist_ok=True)
    output = Path(tempfile.mkdtemp(prefix="fric_", dir=str(root)))
    with (output / "config.json").open("w") as handle:
        json.dump({"train_h5": str(args.train_h5), "val_pt": str(args.val_pt),
                   "epochs": args.epochs, "batch_size": args.batch_size, "seed": 42,
                   "model": {k: v for k, v in config.items() if k != "label_weights"},
                   "label_weights": weights.tolist()}, handle, indent=2)
    print("Output directory: {}".format(output), flush=True)
    checkpoint = output / "friction.pth"
    with (output / "training.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "batch", "samples", "loss"])
        for epoch in range(args.epochs):
            wrapper.train()
            total_loss, count = 0.0, 0
            for step, (x, y) in enumerate(loader, 1):
                x, y = x.to(args.device), y.to(args.device)
                optimizer.zero_grad()
                loss = wrapper.forward_and_get_loss(x, y, is_training=True)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss at batch {}".format(step))
                loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all()
                       for p in wrapper.parameters()):
                    raise ValueError("Non-finite gradients at batch {}".format(step))
                optimizer.step()
                value = loss.item()
                total_loss += value * len(x)
                count += len(x)
                writer.writerow([epoch + 1, step, len(x), value])
                if step == 1 or step % 100 == 0:
                    handle.flush()
                    print("Epoch {}/{} batch {}/{} loss={:.6f}".format(
                        epoch + 1, args.epochs, step, len(loader), value), flush=True)
            wrapper.save(str(checkpoint))
            print("Epoch {} mean loss={:.6f}; saved {}".format(
                epoch + 1, total_loss / count, checkpoint), flush=True)

    # Release training memory before loading the large validation set.
    del x, y, loss, wrapper, optimizer, loader, samples, config
    gc.collect()
    if args.device == "cuda":
        torch.cuda.empty_cache()
    command = [sys.executable, str(Path(__file__).with_name("evaluate_legacy_pt.py")),
               str(args.val_pt), "--checkpoint", str(checkpoint),
               "--batch-size", str(args.batch_size), "--device", args.device]
    print("Evaluating saved weights...", flush=True)
    with (output / "evaluation.txt").open("w") as handle:
        process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end="", flush=True)
            handle.write(line)
        if process.wait() != 0:
            raise RuntimeError("Evaluation failed; weights are saved in {}".format(output))
    print("Training and evaluation complete: {}".format(output), flush=True)


if __name__ == "__main__":
    main()
