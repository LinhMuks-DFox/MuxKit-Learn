import unittest
from mklearn.basic.fully_connected_nn import FullConnectedClassifier
from mklearn.util.model_predictions import is_same_device
import torch


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.fully = FullConnectedClassifier(3, 1)
        # print(self.fully.device_)  # cpu
        # self.fully.to_device("cuda:0")
        # print(self.fully.device_)  # cuda:0
        #
        # self.fully.to_device("cpu")
        # print(self.fully.device_)  # cpu
        #
        # self.fully.to(torch.device("cuda:0"))
        # print(self.fully.device_)  # cuda:0

    def test_is_same_device(self):
        self.assertTrue(is_same_device(torch.nn.Linear(1, 1).to("cuda:0"), "cuda:0"))
        self.assertTrue(is_same_device(torch.nn.Linear(1, 1).to("cuda:0"), torch.device("cuda:0")))
        self.assertTrue(is_same_device(torch.Tensor([1]).to("cuda:0"), "cuda:0"))
        self.assertTrue(is_same_device(self.fully.to(torch.device("cuda:0")), "cuda:0"))
        self.assertTrue(is_same_device(self.fully.to_device("cuda:0"), torch.device("cuda:0")))
        self.assertTrue(is_same_device(self.fully.to_device("cuda:0"), torch.Tensor([1]).to("cuda:0")))

        self.assertFalse(is_same_device(torch.nn.Linear(1, 1).to("cuda:0"), torch.device("cpu")))
        self.assertFalse(is_same_device(None, torch.nn.Linear(1, 1)))
        self.assertFalse(is_same_device(None, None))
        self.assertFalse(is_same_device(self.fully.to_device("cuda:0"), torch.device("cpu")))


if __name__ == '__main__':
    unittest.main()
