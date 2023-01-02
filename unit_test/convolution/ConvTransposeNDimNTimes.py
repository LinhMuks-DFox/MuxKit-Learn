import unittest

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from mklearn.convolution.ConvoNTime import ConvTransposeNDimNTimes


class ConvTransposeNDimNTimesTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.conv_transpose = ConvTransposeNDimNTimes(**{
            "input_dim": np.array([1, 28, 28]),
            "conv_transpose_n_times": 5,
            "kernel_size": np.array([[3, 3] for _ in range(5)]),
            "out_channels": np.array([1, 2, 4, 5, 16]),
            "padding": np.array([[0, 0] for _ in range(5)]),
        })

        self.dataset = datasets.mnist.FashionMNIST(**{
            "download": True,
            "train": True,
            "root": "../../data",
            "transform": transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
        })
        self.demo_data = self.dataset[0][0].unsqueeze(0)

        self.excepted_conv_transpose_type = nn.ConvTranspose2d

    def test_conv_transpose_type_selection(self):
        self.assertEqual(self.conv_transpose.convolution_transpose_layer_type_, self.excepted_conv_transpose_type)

    def test_properties(self):
        print(self.conv_transpose.properties_dict())

    def test_summary(self):
        print(self.conv_transpose.summary())

    def test_output_shape(self):
        self.assertTrue(np.all(
            self.conv_transpose.forward(self.demo_data).shape
            == np.array([1, 16, *self.conv_transpose.shape_after_n_time_convolution_transpose_])
        ))

    def test_to_device(self):
        self.conv_transpose.to_device("cpu")
        self.assertEqual(self.conv_transpose.device_,
                         next(self.conv_transpose.conv_transpose_seq_.parameters()).device,
                         torch.device("cpu"))
        self.conv_transpose.to_device("cuda:0")
        self.assertEqual(self.conv_transpose.device_,
                         next(self.conv_transpose.conv_transpose_seq_.parameters()).device,
                         torch.device("cuda:0"))


if __name__ == '__main__':
    unittest.main()
