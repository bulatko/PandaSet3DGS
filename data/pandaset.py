from .base import Scene, Dataset, PointCloud, Frame
from pandaset import PandaSet, Scene as PSScene

class PandaSetDataset(Dataset):
    def __init__(self, path: str):
        self.loader = PandaSet(path)
        self.data = self.loader.scenes
    
    def __getitem__(self, idx) -> Scene:
        # TODO: return loaded scene by index
        pass


class PandaSetScene(Scene):
    def __init__(self, scene: PSScene):
        # TODO: load scene
        # TODO: load frames
        # TODO: load points
        # TODO: load masks
        self.scene = scene
        self.frames = ...
        self.points = ...

    def load_data(self):
        # TODO: load data (points, frames) from scene
        # TODO: convert each frame to Frames class and apply mask
        pass

