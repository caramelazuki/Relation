import unittest

import torch

from relation import Relation


class RelationTest(unittest.TestCase):
    def test_masked_sequence_mean_precision(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                features = torch.zeros(1, 257, 1, dtype=dtype)
                features[:, :128] = 1
                mask = torch.ones(features.shape[:-1], dtype=torch.bool)

                mean = Relation._sequence_mean(features, mask)

                expected = features.float().mean(dim=-2, keepdim=True).to(dtype)
                torch.testing.assert_close(mean, expected, rtol=0, atol=0)
                torch.testing.assert_close(
                    mean, Relation._sequence_mean(features, None), rtol=0, atol=0
                )

    def test_partial_and_empty_mask_sequence_mean(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                features = torch.tensor(
                    [[[1.0], [3.0], [float("nan")]],
                     [[float("nan")], [float("inf")], [5.0]]],
                    dtype=dtype,
                )
                mask = torch.tensor([[True, True, False], [False, False, False]])

                mean = Relation._sequence_mean(features, mask)

                expected = torch.tensor([[[2.0]], [[0.0]]], dtype=dtype)
                torch.testing.assert_close(mean, expected, rtol=0, atol=0)

    def test_long_fp16_masked_sequence_mean(self):
        features = torch.ones(1, 65_536, 1, dtype=torch.float16)
        mask = torch.ones(features.shape[:-1], dtype=torch.bool)

        mean = Relation._sequence_mean(features, mask)

        expected = torch.ones(1, 1, 1, dtype=torch.float16)
        torch.testing.assert_close(mean, expected, rtol=0, atol=0)
