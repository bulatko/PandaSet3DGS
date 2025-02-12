from abc import ABC, abstractmethod
import numpy as np
import cv2
from pathlib import Path
from typing import List
from pycolmap import SceneReconstructor


class Scene(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def load_data(self):
        pass

    def make_reconstruction(self):
        # TODO: make pycolmap.SceneReconstructor
        # TODO: add frames to SceneReconstructor
        # TODO: get point cloud
        # TODO: get cameras
        # TODO: get frames
        # TODO: colorize points

        self.reconstructor = SceneReconstructor()
        
    def export(self, path: str):
        # TODO: export data to colmap format
        # TODO: export masks if exist

        pass

class Dataset(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs):
        pass

    @property
    def scenes(self):
        return self.data
    def __len__(self):
        return len(self.data)
    @abstractmethod
    def __getitem__(self, idx) -> Scene:
        return self.data[idx]
