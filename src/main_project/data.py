from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision import datasets
from sklearn.model_selection import KFold, cross_val_score


def getData():
    training_data = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())

    test_data = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())
    return training_data, test_data


def getDataloader(data) -> DataLoader:
    dataloader = DataLoader(data, batch_size=64, shuffle=True)
    return dataloader


if __name__ == "__main__":
    training_data,test_data = getData()
    train_dataloader, test_dataloader = getDataloader(training_data),getDataloader(test_data)
    dummy_x, dummy_y = next(iter(train_dataloader))
    print(dummy_x.size(), dummy_y.size())


# class MyDataset(Dataset):
#     """My custom dataset."""

#     def __init__(self, data_path: Path) -> None:
#         self.data_path = data_path

#     def __len__(self) -> int:
#         """Return the length of the dataset."""

#     def __getitem__(self, index: int):
#         """Return a given sample from the dataset."""

#     def preprocess(self, output_folder: Path) -> None:
#         """Preprocess the raw data and save it to the output folder."""


# def preprocess(data_path: Path, output_folder: Path) -> None:
#     print("Preprocessing data...")
#     dataset = MyDataset(data_path)
#     dataset.preprocess(output_folder)
if __name__ == "main":
    training_data, test_data = getData()
    print(len(training_data),len(test_data))
