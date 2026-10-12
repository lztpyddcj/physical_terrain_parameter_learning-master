"""Small CPU checks for the local H5 training path."""

import tempfile
import unittest
from pathlib import Path

import deepdish as dd
import torch

from physical_decoder import ModelFactory, ModelParams
from train_local import load_training_samples
from training_utils.loss_wrapper import LossWrapperFactory


class LocalTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.path = Path(self.folder.name) / "samples.h5"
        self.sensors = {
            "counter": torch.cat((torch.zeros(55), torch.ones(55))),
            "policy": torch.randn(110, 133),
            "priv_exte": torch.randn(110, 208),
            "hidden": torch.randn(110, 100),
            "fric": torch.rand(110, 4),
            "stiff": torch.ones(110, 36),
        }

    def test_windows_do_not_cross_episodes(self):
        dd.io.save(str(self.path), {0: self.sensors})
        samples = load_training_samples(self.path, 50)
        self.assertEqual(len(samples), 2)
        self.assertEqual(tuple(samples[0][0].shape), (50, 441))
        torch.testing.assert_close(samples[1][0][:, :133], self.sensors["policy"][55:105])
        torch.testing.assert_close(samples[1][1], self.sensors["fric"][55:105])

    def test_reject_nonfinite_input(self):
        self.sensors["policy"][0, 0] = float("nan")
        dd.io.save(str(self.path), {0: self.sensors})
        with self.assertRaisesRegex(ValueError, "non-finite"):
            load_training_samples(self.path, 50)

    def test_column_counter_matches_vector_counter(self):
        dd.io.save(str(self.path), {0: self.sensors})
        expected = load_training_samples(self.path, 50)
        self.sensors["counter"] = self.sensors["counter"].unsqueeze(1)
        dd.io.save(str(self.path), {0: self.sensors})
        actual = load_training_samples(self.path, 50)
        self.assertEqual(len(actual), 2)
        for (x, y), (expected_x, expected_y) in zip(actual, expected):
            torch.testing.assert_close(x, expected_x)
            torch.testing.assert_close(y, expected_y)

    def test_reject_multicolumn_counter(self):
        self.sensors["counter"] = self.sensors["counter"].unsqueeze(1).repeat(1, 2)
        dd.io.save(str(self.path), {0: self.sensors})
        with self.assertRaisesRegex(ValueError, "counter.*shape"):
            load_training_samples(self.path, 50)

    def test_optimizer_and_saved_predictions(self):
        config = ModelParams(device="cpu").to_dict()
        config.update(label_weights=torch.ones(10) / 10, label_weighting_bin_interval=0.1)
        wrapper = LossWrapperFactory.create_loss_wrapper(config)
        optimizer = torch.optim.Adam(wrapper.parameters(), lr=0.001)
        x, y = torch.randn(2, 50, 441), torch.rand(2, 50, 4)
        before = [p.detach().clone() for p in wrapper.parameters()]
        loss = wrapper.forward_and_get_loss(x, y, is_training=True)
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        optimizer.step()
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, wrapper.parameters())))
        path = Path(self.folder.name) / "model.pth"
        wrapper.save(str(path))
        restored = ModelFactory.create_model(config)
        restored.load(str(path))
        restored.eval()
        wrapper.decoder.eval()
        torch.testing.assert_close(wrapper.decoder.predict(x), restored.predict(x))


if __name__ == "__main__":
    unittest.main()
