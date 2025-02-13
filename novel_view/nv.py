from pycolmap import Image, Reconstruction, Camera
import numpy as np
from typing import List

class NovelViewGenerator:
    def __init__(self, reconstruction: Reconstruction):
        self.reconstruction = reconstruction
        self.frames : List[Image] = reconstruction.images
        self.cameras : List[Camera] = reconstruction.cameras
        
    def transform_camera_frames(self, camera_id: int, transform: np.ndarray) -> List[Image]:
        # TODO: get images with camera_id and apply same transform to them
        images = [image for image in self.frames if image.camera_id == camera_id]
        new_images = [self.apply_transform(image, transform, ...) for image in images]

        return new_images

    def transform_frame(self, 
                        frame: Image, 
                        transform: np.ndarray, 
                        new_frame_id: int,
                        new_frame_name: str) -> Image:
        # TODO: transform frame

        new_transform = ...

        frame = Image(new_frame_name, frame.points2D, new_transform, frame.camera_id, new_frame_id)
        return frame

    
    def export_data(self, images: List[Image]):
        # TODO: export data of new frames in colmap-style formated directory

        pass
