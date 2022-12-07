import argparse
import contextlib
from dataclasses import dataclass, field
from typing import Any, Dict, NamedTuple, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
import pupil_labs.pupil_core_network_client as pcnc

# DEFAULT_NAME = "<surface-name>"
DEFAULT_NAME = "dell_window1"


def main(ip: str = "127.0.01", port: int = 50020, name: str = DEFAULT_NAME):
    device = pcnc.Device(ip, port)

    # TODO: These are pre-recorded intrinsics for the default wide-angle lens resolution
    # of 1280x720 pixels. If you use a different resolution or lens,adjust accordingly.
    camera = RadialDistortionCamera(
        resolution=(1280, 720),
        cam_matrix=[
            [794.3311439869655, 0.0, 633.0104437728625],
            [0.0, 793.5290139393004, 397.36927353414865],
            [0.0, 0.0, 1.0],
        ],
        dist_coefs=[
            [
                -0.3758628065070806,
                0.1643326166951343,
                0.00012182540692089567,
                0.00013422608638039466,
                0.03343691733865076,
                0.08235235770849726,
                -0.08225804883227375,
                0.14463365333602152,
            ]
        ],
    )

    mapper = NoDelaySurfaceGazeMapper(camera)
    mapped_points = []

    with contextlib.suppress(KeyboardInterrupt):
        topics = [f"surfaces.{name}", "gaze"]
        with device.subscribe(topics) as sub:
            while True:
                message = sub.recv_new_message()
                topic = message.payload["topic"]
                if topic.startswith("surfaces"):
                    mapper.update_homography(message.payload["img_to_surf_trans"])
                elif topic.startswith("gaze"):
                    mapped_gaze = mapper.gaze_to_surface(message.payload["norm_pos"])
                    if mapped_gaze is None:
                        continue  # waiting for first surface detection
                    mapped_gaze.timestamp = message.payload["timestamp"]
                    mapped_gaze.confidence = message.payload["confidence"]
                    print("Mapped gaze location:", mapped_gaze)
                    mapped_points.append(mapped_gaze)

    pd.DataFrame(mapped_points).to_csv("test.csv")


@dataclass
class NoDelaySurfaceGazeMapper:
    camera: "RadialDistortionCamera"
    homography: Optional[npt.NDArray[np.float64]] = field(default=None)

    def update_homography(self, homography: npt.ArrayLike):
        homography = np.array(homography)
        if homography.shape != (3, 3):
            raise ValueError(
                f"Input has incorrect shape! Expected 3x3, got {homography}"
            )
        self.homography = homography

    def gaze_to_surface(
        self, norm_pos: Tuple[float, float]
    ) -> Optional["SurfaceMappedGaze"]:
        if self.homography is None:
            return None

        x, y = self.denormalize(*norm_pos)
        points = self.camera.undistort_points_on_image_plane([x, y])
        points.shape = -1, 1, 2
        points = cv2.perspectiveTransform(points, self.homography)
        points.shape = -1
        return SurfaceMappedGaze(*points)

    def denormalize(self, x: float, y: float) -> Tuple[float, float]:
        return self.camera.resolution[0] * x, (1 - y) * self.camera.resolution[1]


@dataclass
class SurfaceMappedGaze:
    norm_x: float
    norm_y: float
    on_surface: bool = field(init=False)
    timestamp: Optional[float] = field(default=None)
    confidence: Optional[float] = field(default=None)

    def __post_init__(self):
        self.on_surface = (0.0 <= self.norm_x <= 1.0) and (0.0 <= self.norm_y <= 1.0)


@dataclass
class RadialDistortionCamera:
    """Camera model assuming a lense with radial distortion."""

    resolution: Tuple[int, int]
    cam_matrix: npt.ArrayLike
    dist_coefs: npt.ArrayLike

    def __post_init__(self):
        self.cam_matrix = np.array(self.cam_matrix)
        self.dist_coefs = np.array(self.dist_coefs)

    # CameraModel Interface

    def undistort_points_on_image_plane(self, points):
        points = self.__unprojectPoints(points, use_distortion=True)
        points = self.__projectPoints(points, use_distortion=False)
        return points

    # Private

    def __projectPoints(self, object_points, rvec=None, tvec=None, use_distortion=True):
        """
        Projects a set of points onto the camera plane as defined by the camera model.
        :param object_points: Set of 3D world points
        :param rvec: Set of vectors describing the rotation of the camera when recording the corresponding object point
        :param tvec: Set of vectors describing the translation of the camera when recording the corresponding object point
        :return: Projected 2D points
        """
        input_dim = object_points.ndim

        object_points = object_points.reshape((1, -1, 3))

        if rvec is None:
            rvec = np.zeros(3).reshape(1, 1, 3)
        else:
            rvec = np.array(rvec).reshape(1, 1, 3)

        if tvec is None:
            tvec = np.zeros(3).reshape(1, 1, 3)
        else:
            tvec = np.array(tvec).reshape(1, 1, 3)

        if use_distortion:
            _D = self.dist_coefs
        else:
            _D = np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0]])

        image_points, jacobian = cv2.projectPoints(
            object_points, rvec, tvec, self.cam_matrix, _D
        )

        if input_dim == 2:
            image_points.shape = (-1, 2)
        elif input_dim == 3:
            image_points.shape = (-1, 1, 2)
        return image_points

    def __unprojectPoints(self, pts_2d, use_distortion=True, normalize=False):
        """
        Undistorts points according to the camera model.
        :param pts_2d, shape: Nx2
        :return: Array of unprojected 3d points, shape: Nx3
        """
        pts_2d = np.array(pts_2d, dtype=np.float32)

        # Delete any posibly wrong 3rd dimension
        if pts_2d.ndim == 1 or pts_2d.ndim == 3:
            pts_2d = pts_2d.reshape((-1, 2))

        # Add third dimension the way cv2 wants it
        if pts_2d.ndim == 2:
            pts_2d = pts_2d.reshape((-1, 1, 2))

        if use_distortion:
            _D = self.dist_coefs
        else:
            _D = np.asarray([[0.0, 0.0, 0.0, 0.0, 0.0]])

        pts_2d_undist = cv2.undistortPoints(pts_2d, self.cam_matrix, _D)

        pts_3d = cv2.convertPointsToHomogeneous(pts_2d_undist)
        pts_3d.shape = -1, 3

        if normalize:
            pts_3d /= np.linalg.norm(pts_3d, axis=1)[:, np.newaxis]

        return pts_3d


if __name__ == "__main__":
    from rich.traceback import install as enable_pretty_traceback

    enable_pretty_traceback()
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--address", type=str, default="127.0.0.1")
    parser.add_argument("-p", "--port", type=int, default=50020)
    parser.add_argument("-n", "--name", type=str, default=DEFAULT_NAME)
    args = parser.parse_args()

    main(args.address, args.port)
