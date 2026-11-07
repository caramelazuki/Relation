import unittest

import torch

from relation import Relation


class RelationTest(unittest.TestCase):
    def test_long_fp16_masked_sequence_mean(self):
        features = torch.ones(1, 65_536, 1, dtype=torch.float16)
        mask = torch.ones(features.shape[:-1], dtype=torch.bool)

        mean = Relation._sequence_mean(features, mask)

        expected = torch.ones(1, 1, 1, dtype=torch.float16)
        torch.testing.assert_close(mean, expected, rtol=0, atol=0)
