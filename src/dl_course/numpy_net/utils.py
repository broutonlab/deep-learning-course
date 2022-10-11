import numpy as np
import ipyplot
import torch
import random
import torchvision
import tqdm
from typing import Tuple


def preview_CIFAR10(root, classes):
    # create non-transformed dataset for display purposes
    displayset = torchvision.datasets.CIFAR10(train=True,
                                              root=root,
                                              transform=None)

    dataset_samples = list(torch.utils.data.Subset(displayset, range(40)))
    images, labels = list(zip(*dataset_samples))
    labels = [classes[l] for l in labels]

    ipyplot.plot_class_representations(images, labels, img_width=90)


class Optimizer:
    def __init__(self, optim_type: str = 'sgd', params: dict = None):
        if params is None:
            self.params = {}
        else:
            self.params = params
        self.params.setdefault('lr', 1e-3)
        self.params.setdefault('momentum', 0.9)
        self.params.setdefault("beta1", 0.9)
        self.params.setdefault("beta2", 0.999)

        if optim_type == 'sgd':
            self.optimize = self.sgd
        elif optim_type == 'momentum':
            self.optimize = self.momentum
        elif optim_type == 'adam':
            self.optimize = self.adam

    def __call__(self, W: np.ndarray, dW: np.ndarray, key: str):
        return self.optimize(W, dW, key)

    def sgd(self, W: np.ndarray, dW: np.ndarray, key: str) -> np.ndarray:
        W -= self.params['lr'] * dW
        return W

    def momentum(self, W: np.ndarray, dW: np.ndarray, key: str) -> np.ndarray:
        v = self.params.get("velocity %s" % key, np.zeros_like(W))
        v = self.params["momentum"] * v + self.params["lr"] * dW
        W -= v
        self.params["velocity %s" % key] = v
        return W

    def adam(self, W: np.ndarray, dW: np.ndarray, key: str) -> np.ndarray:
        m = self.params.get("m %s" % key, np.zeros_like(W))
        v = self.params.get("v %s" % key, np.zeros_like(W))
        self.params.setdefault("t %s" % key, 0)
        self.params['t %s' % key] += 1
        self.params['m %s' % key] = self.params["beta1"] * \
            m + (1 - self.params["beta1"]) * dW
        self.params['v %s' % key] = self.params["beta2"] * \
            v + (1 - self.params["beta2"]) * dW ** 2
        mt = self.params['m %s' % key] / \
            (1 - np.power(self.params["beta1"], self.params["t %s" % key]))
        vt = self.params["v %s" % key] / \
            (1 - np.power(self.params["beta2"], self.params["t %s" % key]))
        W -= self.params["lr"] * mt / (np.sqrt(vt) + 1e-8)
        return W


class BaseNet:
    def __init__(self):
        pass

    def _training_step(self, X_batch: np.ndarray, y_batch: np.ndarray, optimizer: Optimizer):
        pass

    def check_accuracy(self, X: torch.Tensor, y: torch.Tensor) -> float:
        y = y.numpy()
        X = X.numpy()
        preds = self._predict(X)
        preds = np.argmax(preds, axis=1)
        return np.mean(preds == y)

    def train(self, num_epochs: int = 10):
        pass

    def forward(self, X: np.ndarray) -> Tuple[list, dict]:
        pass

    def backward(self, X: np.ndarray, y: np.ndarray, cache: dict) -> Tuple[float, dict]:
        pass

    def _predict(self, X: np.ndarray):
        output, _ = self.forward(X)
        return self.output_activation(output)

    def predict(self, X: np.ndarray) -> str:
        prediction = self._predict(X)
        label = np.argmax(prediction)
        predictions = []
        for i, p in enumerate(prediction):
            predictions.append("%s : %.5f" % (self.labels2names[i], p))

        return "Prediction: %s \nProbabilities:\n%s" % (self.labels2names[label],
                                                        "\n".join(predictions))

    def test(self):
        running_accuracy = 0
        i = 0
        for data in tqdm(self.testloader):
            x, y = data
            i += 1
            running_accuracy += self.check_accuracy(x, y)

        return "Test accuracy: %.4f" % (running_accuracy / i)

    def validation(self):
        # validation
        val_accuracy = 0
        val_i = 0
        for data in tqdm(self.valloader):
            X_val_batch, y_val_batch = data
            val_accuracy += self.check_accuracy(X_val_batch, y_val_batch)
            val_i += 1
        return val_accuracy / val_i


def test_CIFAR10(model, classes, root, transform, num_images=10):
    displayset = torchvision.datasets.CIFAR10(train=True,
                                              root=root,
                                              transform=None)
    display_indices = random.sample(range(len(displayset)), num_images)
    datapoints = [displayset[i][0] for i in display_indices]

    predicted = [model._predict(transform(dp).numpy()) for dp in datapoints]
    labels = [classes[np.argmax(p)] for p in predicted]
    ipyplot.plot_images(datapoints, labels, img_width=90)
