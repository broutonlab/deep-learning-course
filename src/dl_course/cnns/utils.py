import numpy as np
import pandas as pd
import os


def preview_dataset(dataset_path, num_images=5):
    import ipyplot
    csvfile = os.path.join(dataset_path, "test.csv")
    df_test = pd.read_csv(csvfile)
    classes = df_test.iloc[:, 1].unique()
    images_classes = [df_test[df_test.iloc[:, 1] ==
                              cls].iloc[:num_images] for cls in classes]
    total_df = images_classes[0]
    for cls in images_classes[1:]:
        total_df = pd.concat([total_df, cls])

    image_names = total_df.iloc[:, 0]
    image_names = [os.path.join(dataset_path, name) for name in image_names]

    ipyplot.plot_class_tabs(image_names, np.repeat(
        classes, num_images), img_width=150, force_b64=True)


def download_dataset(dataset_path, file_id):
    """
    Download dataset in a specified path
    """
    
    if os.path.isdir(dataset_path):
        print("Dataset is probably already installed...")
        return

    import importlib
    import subprocess
    import sys
    try:
        importlib.import_module("torch_optimizer")
    except ImportError:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', 'gdown'])

    import gdown
    import zipfile
    url = "https://drive.google.com/uc?id={}".format(file_id)
    filename = gdown.download(url, quiet=False)
    print("Unzipping...")
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(dataset_path)
    print("Done!")
