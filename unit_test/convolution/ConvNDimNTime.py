import unittest

import numpy as np
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from mklearn.convolution.ConvoNTime import ConvNDimNTimes


class ConvNDimNTimesTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.conv = ConvNDimNTimes(**{
            "input_dim": np.array([1, 28, 28]),
            "conv_n_times": 5,
            "kernel_sizes": np.array([[3, 3] for _ in range(5)]),
            "out_channels": np.array([1, 2, 4, 8, 16]),
            "paddings": np.array([[1, 1] for _ in range(5)]),
            "strides": np.array([[1, 1] for _ in range(5)]),
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

        self.excepted_conv_type = nn.Conv2d

    def test_conv_type_selection(self):
        self.assertEqual(self.conv.conv_layer_type_, self.excepted_conv_type)

    def test_convolution(self):
        self.assertTrue(np.all(
            self.conv.forward(self.demo_data).shape == np.array([1, 16, *self.conv.shape_after_convolution_])
        ))

    def test_conv_properties(self):
        print(self.conv.properties_dict())

    def test_summary(self):
        print(self.conv.summary())

    def test_to_device(self):
        self.conv.to_device("cpu")
        self.assertEqual(self.conv.device_, next(self.conv.conv_seq_.parameters()).device, torch.device("cpu"))
        self.conv.to_device("cuda:0")
        self.assertEqual(self.conv.device_, next(self.conv.conv_seq_.parameters()).device, torch.device("cuda:0"))


if __name__ == '__main__':
    unittest.main()
