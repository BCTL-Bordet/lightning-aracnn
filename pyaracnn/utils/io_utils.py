import glob
import os
import torch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np
from typing import List


def makedir(*tokens, exist_ok=False):
    path = os.path.join(*tokens)
    os.makedirs(path, exist_ok=True)

    return path


def sglob(*path):
    return sorted(glob.glob(os.path.join(*path)))


def confusion_matrix_to_image(matrix: torch.Tensor, class_names: List[str]):
    # Prepare canvas to draw confusion matrix
    fig = Figure()
    canvas = FigureCanvas(fig)
    ax = fig.gca()

    # generate confusion matrix figure
    confmatrix_image = ConfusionMatrixDisplay(
        np.array(matrix),
        display_labels=class_names,
    )
    # plot figure on axis
    confmatrix_image.plot(ax=ax)
    # rotate x axis labels for readibility
    ax.tick_params(rotation=45, axis="x")
    # flush canvas
    canvas.draw()
    # get width & height in pixels
    width, height = (fig.get_size_inches() * fig.get_dpi()).astype(int)
    # convert figure to (flattened) array and reshape
    image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(height, width, 3)
    # convert to tensor for experiment.add_image
    image = torch.tensor(image)

    return image
