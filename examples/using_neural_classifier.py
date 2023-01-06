import torch
import torch.utils.data.dataloader as dataloader
import torchvision.datasets as datasets
import torchvision.transforms as transform

from examples.example_module import ExampleRunnable
from mklearn.basic.fully_connected_nn import FullConnectedClassifier
from mklearn.model_train.model_training import AlchemyFurnace, AlchemyParameters

data_path = "../data"


class NeuralClassifierShowCase(ExampleRunnable):
    def __init__(self):
        self.model_ = FullConnectedClassifier(input_shape=28 * 28, output_shape=10)
        self.dataset = datasets.FashionMNIST("../data", transform=transform.Compose([
            transform.ToTensor(),
            transform.Normalize((0.1307,), (0.3081,)),
            transform.Lambda(lambda x: x.view(-1))
        ]), train=True)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = 50
        self.alchemy_furnace_ = AlchemyFurnace(AlchemyParameters(
            model_name="NeuralClassifierShowCase-Show-Case",
            model=self.model_,
            optimizer=torch.optim.Adam(self.model_.parameters(), lr=1e-3),
            loss_function=torch.nn.CrossEntropyLoss(),
            device=self.device,
            epochs=4,
            train_set=self.dataset,
            train_data_loader=dataloader.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=True),
            test_data_loader=dataloader.DataLoader(dataset=self.dataset, batch_size=self.batch_size, shuffle=False),
            verbose=True,
        ))

    def run(self) -> None:
        self.alchemy_furnace_.train().score().plot_loss().save()
        test_data, label = self.dataset[666]
        print(
            f"True label: {label}, model predict: {self.alchemy_furnace_.model_.predict(test_data.to(self.device))}")


if __name__ == "__main__":
    demo = NeuralClassifierShowCase()
    demo.run()
