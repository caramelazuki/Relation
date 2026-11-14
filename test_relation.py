import unittest

import torch

from relation import Relation


class RelationTest(unittest.TestCase):
    def test_entirely_masked_sequence_output(self):
        for activation in (torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Softplus):
            for dtype in (torch.float16, torch.bfloat16, torch.float32):
                for shape in ((4, 3), (2, 4, 3), (2, 3, 4, 3)):
                    with self.subTest(activation=activation, dtype=dtype, shape=shape):
                        layer = Relation(3, 2, activation=activation()).to(dtype=dtype)
                        inputs = torch.randn(shape, dtype=dtype)
                        mask = torch.zeros(shape[:-1], dtype=torch.bool)

                        output = layer(inputs, mask)

                        self.assertEqual(output.shape, shape[:-1] + (2,))
                        self.assertEqual(output.dtype, dtype)
                        torch.testing.assert_close(
                            output, torch.zeros_like(output), rtol=0, atol=0
                        )

    def test_mixed_mask_sequence_outputs(self):
        layer = Relation(1, activation=torch.nn.Sigmoid())
        with torch.no_grad():
            for parameter in layer.parameters():
                parameter.fill_(1 if parameter.ndim == 2 else 0)
        inputs = torch.tensor(
            [[[0.1], [0.2], [0.3]],
             [[0.4], [0.5], [0.6]],
             [[0.7], [0.8], [0.9]]]
        )
        mask_values = [[0, 0, 0], [1, 0, 1], [1, 1, 1]]
        expected = torch.cat(
            (torch.zeros(1, 3, 1),
             torch.sigmoid(torch.tensor([[[0.2], [0.25], [0.3]]])),
             torch.sigmoid(torch.tensor([[[0.56], [0.64], [0.72]]])))
        )

        for dtype in (torch.bool, torch.int64, torch.float32):
            with self.subTest(mask_dtype=dtype):
                mask = torch.tensor(mask_values, dtype=dtype)
                if dtype != torch.bool:
                    mask *= -2
                torch.testing.assert_close(layer(inputs, mask), expected)

        expected_unmasked = torch.sigmoid(inputs * inputs.mean(dim=-2, keepdim=True))
        torch.testing.assert_close(layer(inputs), expected_unmasked)

    def test_entirely_masked_nonfinite_input_output(self):
        inputs = torch.tensor([[[float("nan")], [float("inf")], [-float("inf")]]])
        mask = torch.zeros(1, 3, dtype=torch.bool)
        for activation in (torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Softplus):
            with self.subTest(activation=activation):
                layer = Relation(1, activation=activation())
                output = layer(inputs, mask)
                torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)

    def test_entirely_masked_sequence_gradients(self):
        layer = Relation(3, 2, activation=torch.nn.Sigmoid())
        inputs = torch.randn(2, 4, 3, requires_grad=True)
        mask = torch.zeros(2, 4, dtype=torch.bool)

        layer(inputs, mask).sum().backward()

        torch.testing.assert_close(inputs.grad, torch.zeros_like(inputs), rtol=0, atol=0)
        for parameter in layer.parameters():
            torch.testing.assert_close(
                parameter.grad, torch.zeros_like(parameter), rtol=0, atol=0
            )

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
