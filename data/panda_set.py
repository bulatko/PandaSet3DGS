from base import Scene, Dataset, CameraData, FrameData
from pandaset import DataSet
from pandaset.sequence import Sequence
import numpy as np

import pycolmap

import json

import PIL

import pickle

import typing as tp

import trimesh

from pathlib import Path

from pandas import DataFrame

import utils

import logging

from collections import defaultdict


def load_json(path: Path) -> tp.Any:
    with open(path, 'r') as file:
        return json.load(file)


def load_pickle(path: Path) -> DataFrame:
    with open(path, 'rb') as file:
        return pickle.load(file)


class PandaSetScene(Scene):
    def _load_points(self, moment_id: int) -> np.ndarray:
        """
        Loading all lidar points in given moment

        Args:
            moment_id: int
        """
        points_df = load_pickle(self.scene_path / 'lidar' / f'{moment_id:02d}.pkl')
        return points_df[['x', 'y', 'z']].to_numpy(dtype=np.float64)

    def _load_cuboids(self, moment_id: int) -> tp.List[trimesh.Trimesh]:
        """
        Loading all cuboids, which contain a dynamic objects in given moment

        Args:
            moment_id: int
        """
        cuboids_df = load_pickle(self.scene_path / 'annotations' / 'cuboids' / f'{moment_id:02d}.pkl')

        result: tp.List[trimesh.Trimesh] = []

        for _, cuboid_row in cuboids_df.iterrows():
            if cuboid_row['stationary']:
                continue
            cuboid_size = cuboid_row[['dimensions.x', 'dimensions.y', 'dimensions.z']].to_numpy(dtype=np.float64)
            cuboid_translation = cuboid_row[['position.x', 'position.y', 'position.z']].to_numpy(dtype=np.float64)
            cuboid_rotation = cuboid_row['yaw']
            cuboid = utils.get_cuboid_trimesh(
                size=cuboid_size,
                translation=cuboid_translation,
                z_axis_rotation=cuboid_rotation
            )
            result.append(cuboid)

        return result

    def _load_moments_count(self) -> int:
        return len(load_json(self.scene_path / 'meta' / 'timestamps.json'))

    def _load_cameras(self) -> tp.List[CameraData]:
        result: tp.List[CameraData] = []
        for camera_folder in (self.scene_path / 'camera').iterdir():
            if camera_folder.name.startswith('.'):
                continue

            intrinsics = load_json(camera_folder / 'intrinsics.json')

            try:
                any_image = next(camera_folder.glob('*.jpg'))
            except StopIteration:
                raise RuntimeError("Could not find any image in {}".format(camera_folder.absolute()))

            image = PIL.Image.open(any_image)
            width, height = image.size

            result.append(
                CameraData(
                    name=camera_folder.name,
                    width=width,
                    height=height,
                    fx=intrinsics['fx'],
                    fy=intrinsics['fy'],
                    cx=intrinsics['cx'],
                    cy=intrinsics['cy']
                )
            )
        return result

    def _load_frame(self, camera_name: str, moment_id) -> FrameData:
        camera_path = self.scene_path / 'camera' / camera_name
        poses = load_json(camera_path / 'poses.json')

        if len(poses) != self.moments_count:
            raise RuntimeError("Number of pose in {} must be equal to {}!".format(
                (camera_path / 'poses.json').absolute(),
                self.moments_count)
            )

        image_path = camera_path / f'{moment_id:02d}.jpg'

        if not image_path.exists():
            raise RuntimeError("Could not find image in {}".format(image_path.absolute()))

        image_pose = poses[moment_id]

        position = image_pose['position']
        translation = np.array([
            position[key] for key in ['x', 'y', 'z']
        ])

        heading = image_pose['heading']
        rotation = np.array([
            heading[key] for key in ['x', 'y', 'z', 'w']
        ])

        cam_from_world = pycolmap.Rigid3d(
            translation=translation,
            rotation=rotation,
        ).inverse()

        return FrameData(
            pose=cam_from_world,
            camera_path=camera_path,
            moment_id=moment_id,
        )

    def _save_image_mask(self, frame: FrameData, image_mask: np.ndarray) -> None:
        image_mask = (image_mask * 255).astype(np.uint8)
        mask_image = PIL.Image.fromarray(image_mask)
        save_path = self.masks_output_path / (frame.image_name + ".png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        mask_image.save(save_path)

    def __init__(self, scene_path: Path | str, masks_output_path: tp.Optional[Path | str] = None) -> None:
        self.scene_path = Path(scene_path)

        if masks_output_path is not None:
            self.masks_output_path = Path(masks_output_path)
            self.masks_enabled = True
        else:
            self.masks_output_path = None
            self.masks_enabled = False

        self.moments_count = self._load_moments_count()
        self.cameras = self._load_cameras()
        self.frames: tp.List[FrameData] = []
        self.points: tp.Optional[np.ndarray] = None

        for moment_id in range(self.moments_count):
            current_cuboids = self._load_cuboids(moment_id)
            current_points = self._load_points(moment_id)
            current_frames: tp.List[tp.Tuple[FrameData, CameraData]] = []

            for camera_data in self.cameras:
                frame = self._load_frame(
                    camera_name=camera_data.name,
                    moment_id=moment_id
                )

                current_frames.append((
                    frame,
                    camera_data
                ))
                self.frames.append(frame)

            if self.masks_enabled:
                outside_cuboids_mask = np.ones((current_points.shape[0],), dtype=np.bool)
                for frame, camera_data in current_frames:
                    # create fake image for calculation project points
                    fake_reconstruction = pycolmap.Reconstruction()
                    fake_reconstruction.add_camera(
                        camera_data.build(0)
                    )
                    fake_reconstruction.add_image(
                        frame.build(0, 0)
                    )

                    image = fake_reconstruction.images[0]
                    image_mask = np.zeros((camera_data.height, camera_data.width), dtype=np.uint8)

                    projected_points, before_camera_mask = utils.vectorized_project_points(
                        current_points,
                        image,
                    )

                    for cuboid in current_cuboids:
                        cuboid.apply_scale(scaling=1.05)

                        in_cuboid_mask = cuboid.contains(current_points)

                        utils.mark_dynamic_object_on_mask(
                            projected_points[np.logical_and(before_camera_mask, in_cuboid_mask)],
                            image_mask
                        )

                        print(np.logical_and(before_camera_mask, in_cuboid_mask).sum())

                        outside_cuboids_mask = np.logical_and(outside_cuboids_mask, ~in_cuboid_mask)

                    self._save_image_mask(frame, image_mask)
            else:  # if masks is disabled
                outside_cuboids_mask, inside_cuboids_mask = utils.filter_points_by_cuboids(
                    current_points,
                    current_cuboids
                )

            if self.points is None:
                self.points = current_points[outside_cuboids_mask]
            else:
                self.points = np.concatenate((self.points, current_points[outside_cuboids_mask]), axis=0)


# class PandaSetDataset(Dataset):
#     def __init__(self, path: str):
#         self.loader = DataSet(path)
#         self.data = self.loader.scenes
#
#     def __getitem__(self, idx) -> PandaSetScene:
#         # TODO: return loaded scene by index
#         pass
