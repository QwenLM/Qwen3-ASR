"""Regression cases for stable longest-nondecreasing-subsequence repair."""

import unittest

import torch

from qwen_asr.inference.qwen3_forced_aligner import Qwen3ForceAlignProcessor


class TimestampRepairTests(unittest.TestCase):
    def test_stable_anchors_and_interpolation(self):
        cases = [
            ([7], [7]),
            ([0, 100, 100, 100, 250], [0, 100, 100, 100, 250]),
            ([400, 300, 200, 100], [400, 400, 400, 400]),
            ([0, 400, 200, 800], [0, 400, 400, 800]),
            ([4, 1, 3, 2, 5], [1, 1, 3, 3, 5]),
            ([0, 800, 700, 600, 500, 1000], [0, 800, 850, 900, 950, 1000]),
            ([5, 4, 3, 0, 1, 2, 3], [0, 0, 0, 0, 1, 2, 3]),
        ]
        for values, expected in cases:
            with self.subTest(values=values):
                data = torch.tensor(values)
                actual = Qwen3ForceAlignProcessor.fix_timestamp(None, data)
                self.assertEqual(actual, expected)
                self.assertEqual(data.tolist(), values)

    def test_empty_input_keeps_legacy_error(self):
        with self.assertRaises(ValueError):
            Qwen3ForceAlignProcessor.fix_timestamp(None, torch.tensor([]))


if __name__ == "__main__":
    unittest.main()
