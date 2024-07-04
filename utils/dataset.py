datapath = "E:\Data\MNIST"
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

class MnistDataset:
    def __init__(self, datapath, batch_size) -> None:
        transform = transforms.ToTensor()

        # 训练集
        self.train_dataset = datasets.MNIST(root=datapath, train=True, download=False, transform=transform)
        # 测试集
        self.test_dataset = datasets.MNIST(root=datapath, train=False, download=False, transform=transform)

        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False)
    def show_images(self, images, labels):
        _, axs = plt.subplots(2, 2, figsize=(8, 8))
        for i, ax in enumerate(axs.flatten()):
            ax.imshow(images[i][0], cmap='gray')
            ax.set_title(f"Label: {labels[i]}")
            ax.axis('off')
        plt.show()


class SwissRollDataset:
    def __init__(self) -> None:
        from sklearn.datasets import make_swiss_roll
        self.X, self.labels = make_swiss_roll(n_samples=2000, noise=0.05, random_state=5)
    def draw(self, X, labels):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d", elev=7, azim=-80)
        ax.set_position([0,0,0.95,1])
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=labels, edgecolors='k', cmap=plt.cm.cool)