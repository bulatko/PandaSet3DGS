from abc import ABC, abstractmethod
import numpy as np
import cv2
from pathlib import Path
from typing import List
from pycolmap import ReconstructionManager
from pycolmap import Camera

    

class Scene(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def load_data(self) -> None:
        pass

    def make_reconstruction(self):
        # TODO: make pycolmap.SceneReconstructor
        # TODO: add data to SceneReconstructor
        # TODO: colorize points

        self.load_data()
        points = self.points
        frames = self.frames
        cameras = self.cameras
        
        self.reconstructor = Reconstruction()
        
    def export(self, path: str) -> None:
        # TODO: export data to colmap format
        # TODO: export masks if exist

        pass

class Dataset(ABC):
    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        pass

    @property
    def scenes(self) -> List[Scene]:
        return self.data
    def __len__(self) -> int:
        return len(self.data)
    @abstractmethod
    def __getitem__(self, idx) -> Scene:
        return self.data[idx]
