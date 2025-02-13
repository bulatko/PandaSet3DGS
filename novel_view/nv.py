from pycolmap import Image
from data import Scene
import numpy as np
from typing import List


class NovelViewGenerator:
    def __init__(self, scene: Scene):
        self.scene = scene
        self.frames : List[Image] = scene.frames

    def transform_frame(self, 
                        frame: Image, 
                        transform: np.ndarray, 
                        new_frame_id: int,
                        new_frame_name: str) -> Image:
        # TODO: transform frame

        new_transform = ...

        frame = Image(new_frame_name, frame.points2D, new_transform, frame.camera_id, new_frame_id)
        return frame

    def apply_transforms_to_frame(self, frame: Image, transforms: np.ndarray, ids: List[int], names: List[str]) -> List[Image]:
        # TODO: transform frames
        
        assert len(transforms) == len(ids) == len(names)

        new_frames = []
        for i in range(len(transforms)):
            new_frames.append(self.transform_frame(frame, transforms[i], ids[i], names[i]))

        return new_frames
    
    def export_data(self, path_to_data: str, frames: List[Image]):
        # TODO: export data of new frames to colmap-style formated directory
        pass
