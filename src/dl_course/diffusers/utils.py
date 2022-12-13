from IPython.display import display
from PIL import Image
import matplotlib.pyplot as plt
import torch
import math
import os

def make_grid(images, rows, cols):
    w, h = images[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    for i, image in enumerate(images):
        grid.paste(image, box=(i%cols*w, i//cols*h))
    return grid


def preview_pipeline(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size = config.preview_images, 
        generator=torch.manual_seed(config.seed),
    ).images

    # Make a grid out of the images
    image_grid = make_grid(images, rows=math.ceil(config.preview_images/4), cols=4)
    
    # Resize image grid:
    image_grid = image_grid.resize((image_grid.size[0] * config.preview_size_factor, 
                                    image_grid.size[1] * config.preview_size_factor))

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")
    display(image_grid)
    

def preview_transformed_dataset(dataset, column_name):
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    for i, image in enumerate(dataset[:4][column_name]):
        axs[i].imshow(image.permute(1, 2, 0).numpy() / 2 + 0.5)
        axs[i].set_axis_off()
    fig.show()