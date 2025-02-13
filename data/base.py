from abc import ABC, abstractmethod
import numpy as np
# import cv2
from pathlib import Path
from typing import List

import os
import pycolmap
from pycolmap import Reconstruction
from pycolmap import Camera

from dataclasses import dataclass

import typing as tp


@dataclass
class CameraData:
    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @property
    def params(self):
        return [self.fx, self.fy, self.cx, self.cy]

    def build(self, camera_id: int, model: str = "PINHOLE") -> pycolmap.Camera:
        return pycolmap.Camera(
            camera_id=camera_id,
            model=model,
            width=self.width,
            height=self.height,
            params=self.params
        )


@dataclass
class FrameData:
    moment_id: int
    camera_path: Path
    pose: pycolmap.Rigid3d

    @property
    def camera_name(self) -> str:
        return self.camera_path.name

    @property
    def image_path(self) -> Path:
        return self.camera_path / f'{self.moment_id:02d}.jpg'

    @property
    def image_name(self):
        return "{}_{}".format(self.camera_path.name, self.image_path.name)

    def build(self, image_id, camera_id) -> pycolmap.Image:
        return pycolmap.Image(
            image_id=image_id,
            camera_id=camera_id,
            cam_from_world=self.pose,
            name=self.image_name
        )


class Scene(ABC):
    cameras: tp.List[CameraData] = None
    points: np.ndarray = None
    frames: tp.List[FrameData] = None

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        pass

    def make_reconstruction(self):
        points = self.points
        frames = self.frames
        cameras = self.cameras

        self.reconstruction = Reconstruction()

        # from camera_name to camera_id
        cameras_by_names: tp.Dict[str, int] = {}

        for camera_id, camera_data in enumerate(cameras):
            self.reconstruction.add_camera(camera_data.build(camera_id))
            cameras_by_names[camera_data.name] = camera_id

        for image_id, frame_data in enumerate(frames):
            camera_id = cameras_by_names[frame_data.camera_name]
            self.reconstruction.add_image(frame_data.build(image_id, camera_id))



    def export(self, path: str, as_text: bool) -> None:
        if isinstance(path, str):
            path = Path(path)

        if as_text:
            self.reconstruction.write_text(path)
        else:
            self.reconstruction.write(path)


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
