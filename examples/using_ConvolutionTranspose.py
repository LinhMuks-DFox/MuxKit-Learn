import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms

import mklearn.convolution.ConvoNTime as ConvoNTime
from examples.example_module import ExampleRunnable


class ConvolutionTransposeDemo(ExampleRunnable):

    def __init__(self):
        self.convolution_times = 5
        self.conv_kernel_size = np.array([[3, 3] for _ in range(self.convolution_times)])
        self.conv_padding = np.array([[0, 0] for _ in range(self.convolution_times)])
        self.conv_stride = np.array([[1, 1] for _ in range(self.convolution_times)])
        self.conv_dilation = np.array([[1, 1] for _ in range(self.convolution_times)])
        self.test_output_channel = np.array([1, 2, 4, 8, 16])

        self.input_dims = np.array([1, 28, 28])
        self.transformer = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,)),
        ])
        self.dataset = torchvision.datasets.mnist.FashionMNIST(
            download=True,
            train=True,
            root="../data",
            transform=self.transformer)

        self.conv = ConvoNTime.ConvNDimNTimes(
            self.input_dims,
            conv_n_times=self.convolution_times,
            kernel_sizes=self.conv_kernel_size,
            out_channels=self.test_output_channel,
            paddings=self.conv_padding,
            strides=self.conv_stride,
        )
        self.demo_data = self.dataset[0][0].unsqueeze(0)

        self.conv_transpose = ConvoNTime.ConvTransposeNDimNTimes(
            first_conv_layer_in_channel=self.test_output_channel[-1],
            conv_transpose_n_times=self.convolution_times,
            kernel_size=self.conv_kernel_size,
            out_channels=np.flipud(self.test_output_channel),
            padding=self.conv_padding,
            stride=self.conv_stride,
            # dilation=self.conv_dilation,
        )

    def run(self) -> None:
        conv_output = self.conv(self.demo_data)
        conv_transpose_output = self.conv_transpose(conv_output)
        print("Shape after convolution: ", conv_output.shape)
        print("Shape after convolution transpose:", conv_transpose_output.shape)
        fig, axs = plt.subplots(5, 4, figsize=(18, 18))
        # set plot title
        fig.suptitle("Convolution output", fontsize=32)

        for i in range(16):
            axs[i // 4, i % 4].matshow(conv_output[0, i].detach().numpy())
            axs[i // 4, i % 4].set_title(f"Channel {i}")

        for i in range(4):
            axs[4, i].matshow(self.demo_data.squeeze().numpy())
            axs[4, i].set_title("Original image")
        # plt.savefig("convolution_output.png", dpi=300)
        plt.show()

        fig, axs = plt.subplots(1, 2)
        fig.suptitle("Transpose Convolution")
        axs[0].matshow(self.demo_data.squeeze().numpy())
        axs[0].set_title("Original image")
        axs[1].matshow(conv_transpose_output.squeeze().detach().numpy())
        axs[1].set_title("Convolution Transpose Output")

        plt.show()


if __name__ == "__main__":
    demo = ConvolutionTransposeDemo()
    demo.run()
