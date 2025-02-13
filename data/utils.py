import typing
import typing as tp
from typing import Tuple, Any

import numpy as np
import pycolmap

import trimesh
from numpy import ndarray, dtype

import cv2

import scipy
from scipy.spatial import ConvexHull


def vectorized_project_points(points: np.ndarray, image: pycolmap.Image) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Projects 3D points onto an image plane using camera parameters.

    Args:
        points: Array of shape (n_points, 3) representing 3D points in world coordinates.
        image: pycolmap.Image object containing camera pose and calibration data.

    Returns:
        Tuple of:
        1. Array of shape (n_points, 2): 2D image coordinates of projected points.
           Non-projectable points are filled with invalid values.
        2. Array of shape (n_points,): Boolean mask indicating projectable points (True if valid).
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (n, 3), got {}".format(points.shape))

    transform_matrix = image.cam_from_world.matrix()  # World to camera transform
    calibration_matrix = image.camera.calibration_matrix()  # Camera intrinsics

    # Convert to homogeneous coordinates
    points_homog = np.hstack([points, np.ones((len(points), 1))])

    # Transform points to camera coordinates
    points_cam = (transform_matrix @ points_homog.T).T

    # Mask for points in front of the camera
    before_camera_mask = points_cam[:, 2] > 0

    # Normalize by depth (z-coordinate)
    points_cam[before_camera_mask] /= points_cam[:, 2][before_camera_mask, None]

    # Project to 2D image coordinates
    projected_points = (calibration_matrix @ points_cam.T)[:2].T

    return projected_points, before_camera_mask


def vectorized_getting_points2d_colors(
        points2d: np.ndarray,
        image_colores: np.ndarray,
        image_mask: tp.Optional[np.ndarray] = None
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Retrieves colors from an image for given 2D points.

    Args:
        points2d: Array of shape (n_points, 2) representing 2D points (float data type) in image coordinates.
        image_colores: Array of shape (height, width, 3) representing the image in RGB format.
        image_mask: Optional binary mask of shape (height, width) to filter valid regions.

    Returns:
        Tuple of:
        1. Array of shape (n_points, 3): Colors for each point. Non-valid points may contain trash.
        2. Array of shape (n_points,): Boolean mask indicating valid points (True if color is valid).
    """
    if points2d.ndim != 2 or points2d.shape[1] != 2:
        raise ValueError("points2d must have shape (n_points, 2), got {}".format(points2d.shape))
    if image_colores.ndim != 3 or image_colores.shape[2] != 3:
        raise ValueError(
            "image_colores must have shape (height, width, 3), got {}".format(image_colores.shape)
        )
    if image_mask is not None and image_mask.shape != image_colores.shape[:2]:
        raise ValueError(
            "image_mask must match image_colores shape {}, got {}".format(
                image_colores.shape[:2], image_mask.shape
            )
        )

    height, width = image_colores.shape[:2]

    # Shift points to image center
    points2d[:, 0] += width / 2
    points2d[:, 1] += height / 2

    # Convert to pixel coordinates
    pixel_coords = np.floor(points2d).astype(int)

    # Check if points are within image bounds
    in_image_mask = np.logical_and.reduce([
        pixel_coords[:, 0] >= 0,
        pixel_coords[:, 0] < width,
        pixel_coords[:, 1] >= 0,
        pixel_coords[:, 1] < height,
    ])

    # Apply image mask if provided
    if image_mask is not None:
        in_mask_mask = image_mask.astype(bool)[
            pixel_coords[in_image_mask][:, 1],
            pixel_coords[in_image_mask][:, 0]
        ]
        in_image_mask = np.logical_and(in_mask_mask, in_image_mask)

    # Retrieve colors for valid points
    points_colors = np.zeros((len(pixel_coords), 3), dtype=np.uint8)
    points_colors[in_image_mask] = image_colores[
        pixel_coords[in_image_mask][:, 1], pixel_coords[in_image_mask][:, 0]
    ]

    return points_colors, in_image_mask


def vectorized_getting_points3d_colors(
        points3d: np.ndarray,
        image: pycolmap.Image,
        image_colores: np.ndarray,
        image_mask: tp.Optional[np.ndarray] = None
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Retrieves colors from an image for given 3D points by projecting them onto the image plane.

    Args:
        points3d: Array of shape (n_points, 3) representing 3D points in world coordinates.
        image: pycolmap.Image object containing camera pose and calibration data.
        image_colores: Array of shape (height, width, 3) representing the image in RGB format.
        image_mask: Optional binary mask of shape (height, width) to filter valid regions.

    Returns:
        Tuple of:
        1. Array of shape (n_points, 3): Colors for each point. Non-valid points may contain trash.
        2. Array of shape (n_points,): Boolean mask indicating valid points (True if point is valid).
    """
    # Project 3D points to 2D image coordinates
    points2d, before_camera_mask = vectorized_project_points(points3d, image)

    # Retrieve colors for projected 2D points
    points_colors, in_image_mask = vectorized_getting_points2d_colors(points2d, image_colores, image_mask)

    return points_colors, np.logical_and(before_camera_mask, in_image_mask)


def get_cuboid_trimesh(
        size: np.ndarray,
        translation: tp.Optional[np.array] = None,
        z_axis_rotation: float = 0,
) -> trimesh.base.Trimesh:
    """
    Creates Trimesh object from cuboid

    Args:
        size: numpy array of shape (3,) representing the size along axises
        translation: numpy array of shape (3,) representing the shift cuboid center
        z_axis_rotation: a float representing the rotation angle through z axis

    Returns:
        Cuboid kept in a Trimesh object
    """
    cuboid = trimesh.primitives.Box(extents=size.astype(np.float64))
    rotation_matrix = trimesh.transformations.rotation_matrix(
        z_axis_rotation,
        [0, 0, 1]
    )
    translation_matrix = trimesh.transformations.translation_matrix(
        translation.astype(np.float64)
    )
    homogeneous_matrix = trimesh.transformations.concatenate_matrices(translation_matrix, rotation_matrix)
    cuboid.apply_transform(homogeneous_matrix)
    return cuboid


def mark_dynamic_object_on_mask(points2d: np.ndarray, output_mask: np.ndarray) -> None:
    """
    Marking dynamic object on an output_mask

    Args:
        points2d: numpy array of shape (n_points, 2), which contains projected on image float points of dynamic object
        output_mask: numpy array of shape (image_height, image_width), place to be written mask

    """
    assert output_mask.dtype == np.uint8

    if len(points2d) == 0:
        return

    try:
        hull = ConvexHull(points2d)
        cv2.fillPoly(output_mask, [points2d[hull.vertices]], 0)
    except scipy.spatial._qhull.QhullError:
        pass


def filter_points_by_cuboids(
        points: np.ndarray,
        cuboids: tp.List[trimesh.base.Trimesh],
        cuboid_scale: float = 1.05
) -> tp.Tuple[np.ndarray, np.ndarray]:
    """
    Dividing points by cuboids

    Args:
        points: numpy array of shape (n_points, 3)
        cuboids: list of cuboids
        cuboid_scale: scale size of cuboids scaling

    Returns:
        Tuple of:
        1. numpy array of shape (n_points, 3) - outside cuboids points mask
        2. numpy array of shape (n_points, 3) - inside cuboids points mask
    """
    inside_mask = np.zeros(points.shape[0], dtype=bool)
    for cuboid in cuboids:
        copied_cuboid = cuboid.copy()
        copied_cuboid.apply_scale(cuboid_scale)
        inside_mask = np.logical_or(inside_mask, copied_cuboid.contains(points))
    return ~inside_mask, inside_mask
