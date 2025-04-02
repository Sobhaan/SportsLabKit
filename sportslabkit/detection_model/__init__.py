import os
import inspect

from sportslabkit.detection_model.base import BaseDetectionModel
from sportslabkit.detection_model.dummy import DummyDetectionModel
from sportslabkit.detection_model.yolov8 import YOLOv8, YOLOv8l, YOLOv8m, YOLOv8n, YOLOv8s, YOLOv8x
from sportslabkit.logger import logger


__all__ = [
    "BaseDetectionModel",
    "load",
    "show_available_models",
    "YOLOv8",
    "YOLOv8n",
    "YOLOv8s",
    "YOLOv8m",
    "YOLOv8l",
    "YOLOv8x",
    "DummyDetectionModel",
]


def inheritors(cls):
    """
    Get all subclasses of a given class.

    Args:
        cls (type): The class to find subclasses of.

    Returns:
        set[type]: A set of the subclasses of the input class.
    """
    subclasses = set()
    work = [cls]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                work.append(child)
    return subclasses


def show_available_models():
    """
    Print the names of all available BaseDetectionModel models.

    The models are subclasses of BaseDetectionModel. The names are printed as a list to the console.
    """
    print(sorted([cls.__name__ for cls in inheritors(BaseDetectionModel)]))


def load(model_name, **model_config):
    """
    Load a model by name or path.
    
    The function first checks if model_name is a file path. If it is, it loads the custom model.
    Otherwise, it searches subclasses of BaseDetectionModel for a match with the given name.
    
    Args:
        model_name (str): The name of the model to load or a path to a custom model file.
        model_config (dict, optional): The model configuration to use when instantiating the model.
    
    Returns:
        BaseDetectionModel: An instance of the requested model, or None if no match was found.
    """
    # First, let's handle absolute and relative paths
    if isinstance(model_name, str):
        # Try absolute path first
        if os.path.isfile(os.path.abspath(model_name)):
            logger.info(f"Found model at {os.path.abspath(model_name)}")
            config = {k.lower(): v for k, v in model_config.items()}
            config["model"] = os.path.abspath(model_name)
            return YOLOv8(**config)
        
        # Try relative to current working directory
        cwd_path = os.path.join(os.getcwd(), model_name)
        if os.path.isfile(cwd_path):
            logger.info(f"Found model at {cwd_path}")
            config = {k.lower(): v for k, v in model_config.items()}
            config["model"] = cwd_path
            return YOLOv8(**config)
            
        # Check with and without .pt extension
        extensions = ['']
        if not model_name.endswith('.pt'):
            extensions.append('.pt')
            
        for ext in extensions:
            test_name = model_name + ext
            # Try absolute path
            if os.path.isfile(os.path.abspath(test_name)):
                path = os.path.abspath(test_name)
                logger.info(f"Found model at {path}")
                config = {k.lower(): v for k, v in model_config.items()}
                config["model"] = path
                return YOLOv8(**config)
                
            # Try relative to CWD
            if os.path.isfile(os.path.join(os.getcwd(), test_name)):
                path = os.path.join(os.getcwd(), test_name)
                logger.info(f"Found model at {path}")
                config = {k.lower(): v for k, v in model_config.items()}
                config["model"] = path
                return YOLOv8(**config)
                
            # Check common subdirectories
            for subdir in ['weights', 'models', 'checkpoints']:
                path = os.path.join(os.getcwd(), subdir, test_name)
                if os.path.isfile(path):
                    logger.info(f"Found model at {path}")
                    config = {k.lower(): v for k, v in model_config.items()}
                    config["model"] = path
                    return YOLOv8(**config)
    
    # Continue with the original approach for named models
    for cls in inheritors(BaseDetectionModel):
        if model_name in [cls.__name__.lower(), cls.__name__]:
            # Filtering the model_config to only include keys that match the parameters of the target class
            config = {k.lower(): v for k, v in model_config.items()}
            return cls(**config)

    logger.warning(
        f"Model {model_name} not found. Available models: {[cls.__name__ for cls in inheritors(BaseDetectionModel)]} (lowercase is allowed)"
    )
    return None
