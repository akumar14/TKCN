from .model_zoo import get_model
from .model_store import get_model_file
from .base import *
from .fcn import *
from .tkcnet import *

def get_segmentation_model(name, **kwargs):
    from .fcn import get_fcn
    models = {
        'fcn': get_fcn,
        'tkcnet': get_tkcnet,
    }
    return models[name.lower()](**kwargs)
