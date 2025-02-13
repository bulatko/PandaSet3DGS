from abc import ABC, abstractmethod

import cv2
import numpy as np
# import cv2
from pathlib import Path
from typing import List

import os
import pycolmap
from pycolmap import Reconstruction
from pycolmap import Camera

from dataclasses import dataclass

import utils
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
    masks_enabled: bool
    masks_output_path: tp.Optional[Path]

    @abstractmethod
    def __init__(self, *args, **kwargs) -> None:
        pass

    @abstractmethod
    def _get_mask_path(self, frame: FrameData) -> Path:
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

        points_color_sum = np.zeros((points.shape[0], 3), dtype=np.int32)
        points_color_cnt = np.zeros((points.shape[0], ), dtype=np.int32)

        for image_id, frame_data in enumerate(frames):
            print(image_id)
            image = self.reconstruction.images[image_id]
            image_colors = cv2.imread(str(frame_data.image_path.absolute()), cv2.IMREAD_COLOR_RGB)

            binary_mask = None

            if self.masks_enabled:
                image_mask = cv2.imread(str(self._get_mask_path(frame_data)), cv2.IMREAD_GRAYSCALE)
                _, binary_mask = cv2.threshold(image_mask, 127, 255, cv2.THRESH_BINARY)
                binary_mask = binary_mask // 255
                binary_mask = binary_mask.astype(np.uint8)

            points_colors, points_mask = utils.vectorized_getting_points3d_colors(
                points3d=self.points,
                image=image,
                image_colores=image_colors,
                image_mask=binary_mask
            )

            points_color_sum[points_mask] += points_colors[points_mask]
            points_color_cnt[points_mask] += 1

        points_color_cnt = np.maximum(points_color_cnt, 1)
        points_colors = points_color_sum // points_color_cnt[..., np.newaxis]

        for point_id, point in enumerate(self.points):
            self.reconstruction.add_point3D(
                xyz=point,
                color=points_colors[point_id].astype(dtype=np.uint8),
                track=pycolmap.Track()
            )

    def export(self, path: str | Path, as_text: bool) -> None:
        if isinstance(path, Path):
            path = str(path.absolute())

        if as_text:
            self.reconstruction.write_text(path)
        else:
            self.reconstruction.write(path)
