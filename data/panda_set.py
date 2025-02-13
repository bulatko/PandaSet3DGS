from base import Scene, Dataset
from pandaset import DataSet
from pandaset.sequence import Sequence
import numpy as np

class PandaSetScene(Scene):
    def __init__(self, sequence: Sequence):
        # TODO: define scene
        self.sequence = sequence

    def filter_points(self):
        # TODO: filter 3D points of dynamic objects and ego vehicle
        self.points = ...
        pass
    def make_masks(self) -> np.ndarray:
        # TODO: make masks of dynamic objects and ego vehicle
        pass
    def load_data(self):
        # TODO: load data (points, frames) from scene
        # TODO: convert each frame to Frames class and apply mask
        self.points = ...
        self.frames = ...
        self.masks = self.make_masks()
        self.cameras = ...

        self.filter_points()

        

        pass

class PandaSetDataset(Dataset):
    def __init__(self, path: str):
        self.loader = DataSet(path)
        self.data = self.loader.scenes
    
    def __getitem__(self, idx) -> PandaSetScene:
        # TODO: return loaded scene by index
        pass
