from .base import Scene, Dataset, PointCloud, Frame
from .pandaset import PandaSet, Scene as PSScene


class PandaSetScene(Scene):
    def __init__(self, scene: PSScene):
        # TODO: define scene
        self.scene = scene

    def filter_points(self):
        # TODO: filter points by mask
        pass

    def load_data(self):
        # TODO: load data (points, frames) from scene
        # TODO: convert each frame to Frames class and apply mask
        self.points = ...
        self.frames = ...
        self.masks = ...
        self.cameras = ...

        pass


class PandaSetDataset(Dataset):
    def __init__(self, path: str):
        self.loader = PandaSet(path)
        self.data = self.loader.scenes

    def __getitem__(self, idx) -> PandaSetScene:
        # TODO: return loaded scene by index
        pass
