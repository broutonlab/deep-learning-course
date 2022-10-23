import random
import torch
import ipyplot


def test_model_on_datamodule(model, datamodule, classes, device='cuda', num_images=10):
    display_indices = random.sample(range(len(datamodule.test)), num_images)
    datapoints = [datamodule.test[i][0] for i in display_indices]
    datapoints = torch.stack(datapoints, dim=0).to(device)
    model = model.to(device)
    with torch.no_grad():
        predicted = model.forward(datapoints)

    predicted = predicted.cpu().numpy()
    labels = [classes[np.argmax(p)] for p in predicted]
    t = T.ToPILImage()
    datapoints = [t(img) for img in datapoints]
    ipyplot.plot_images(datapoints, labels, img_width=90)


def preview_datamodule(datamodule, classes, num_samples=10):
    datamodule.setup()
    indices = [random.randrange(len(datamodule.train))
               for i in range(num_samples)]
    dataset_samples = list(torch.utils.data.Subset(datamodule.train, indices))
    images, labels = list(zip(*dataset_samples))
    t = T.ToPILImage()
    images = [t(img) for img in images]
    labels = [classes[l] for l in labels]
    ipyplot.plot_images(images, labels)
