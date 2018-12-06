from .base import *
from .cityscapes import CityscapesSegmentation
datasets = {
    'cityscapes': CityscapesSegmentation,
}
def get_segmentation_dataset(name, **kwargs):
    return datasets[name.lower()](**kwargs)
