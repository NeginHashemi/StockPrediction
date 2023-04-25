import torch
import sys
import numpy
import logging
import os
import random
from ppo import DictList


class RawImagePreprocessor(object):
    def __call__(self, obss, device=None):
        obs = numpy.array(obss)
        obs = torch.tensor(obs, device=device, dtype=torch.float)
        return obs


class ObssPreprocessor:
    def __init__(self, model_name, obs_space=None):
        self.preproc = RawImagePreprocessor()

    def __call__(self, obss, device=None):
        return self.preproc(obss, device=device)
    

def storage_dir():
    # defines the storage directory to be in the root (Same level as babyai folder)
    return os.environ.get("STOCK_STORAGE", '.')

def create_folders_if_necessary(path):
    dirname = os.path.dirname(path)
    if not(os.path.isdir(dirname)):
        os.makedirs(dirname)

def seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_log_dir(log_name):
    path = os.path.join(storage_dir(), "logs", log_name)
    return path

def get_log_path(log_name):
    return os.path.join(get_log_dir(log_name), "log.log")

def synthesize(array):
    import collections
    d = collections.OrderedDict()
    d["mean"] = numpy.mean(array)
    d["std"] = numpy.std(array)
    d["min"] = numpy.amin(array)
    d["max"] = numpy.amax(array)
    return d

def configure_logging(log_name):
    path = get_log_path(log_name)
    create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s: %(asctime)s: %(message)s",
        handlers=[
            logging.FileHandler(filename=path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
def get_model_dir(model_name):
    return os.path.join(storage_dir(), "models", model_name)

def get_model_path(model_name):
    return os.path.join(get_model_dir(model_name), "model.pt")

def load_model(model_name, raise_not_found=True):
    path = get_model_path(model_name)
    try:
        model = torch.load(path)
        model.eval()
        return model
    except FileNotFoundError:
        if raise_not_found:
            raise FileNotFoundError("No model found at {}".format(path))

def save_model(model, model_name):
    path = get_model_path(model_name)
    create_folders_if_necessary(path)
    torch.save(model, path)

def get_path(name, prefix):
    return os.path.join(get_model_dir(name), prefix)

def save_obj(obj, name, prefix):
    path = get_path(name, prefix)
    create_folders_if_necessary(path)
    torch.save(obj, path)