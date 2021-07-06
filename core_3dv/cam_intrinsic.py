"""
Encapsulates camera intrinsic parameters for projecting / deprojecitng points
Author: Jeff Mahler
"""
import copy
import logging
import numpy as np
import json
import os

class CameraIntrinsics(object):
    """A set of intrinsic parameters for a camera. This class is used to project
    and deproject points.
    """

    def __init__(self, K:np.ndarray, height=-1, width=-1):
        """Initialize a CameraIntrinsics model.

        Parameters
        ----------
        frame : :obj:`str`
            The frame of reference for the point cloud.
        fx : float
            The x-axis focal length of the camera in pixels.
        fy : float
            The y-axis focal length of the camera in pixels.
        cx : float
            The x-axis optical center of the camera in pixels.
        cy : float
            The y-axis optical center of the camera in pixels.
        skew : float
            The skew of the camera in pixels.
        height : float
            The height of the camera image in pixels.
        width : float
            The width of the camera image in pixels
        """

        self._fx = K[0, 0]
        self._fy = K[1, 1]
        self._cx = K[0, 2]
        self._cy = K[1, 2]
        self._skew = K[0, 1]
        self._height = int(height)
        self._width = int(width)

        # set camera projection matrix
        self._K = np.array([[self._fx, self._skew, self._cx],
                            [0, self._fy, self._cy],
                            [0, 0, 1]])

    @property
    def fx(self):
        """float : The x-axis focal length of the camera in pixels.
        """
        return self._fx

    @property
    def fy(self):
        """float : The y-axis focal length of the camera in pixels.
        """
        return self._fy

    @property
    def cx(self):
        """float : The x-axis optical center of the camera in pixels.
        """
        return self._cx

    @cx.setter
    def cx(self, z):
        self._cx = z
        self._K = np.array([[self._fx, self._skew, self._cx],
                            [0, self._fy, self._cy],
                            [0, 0, 1]])

    @property
    def cy(self):
        """float : The y-axis optical center of the camera in pixels.
        """
        return self._cy

    @cy.setter
    def cy(self, z):
        self._cy = z
        self._K = np.array([[self._fx, self._skew, self._cx],
                            [0, self._fy, self._cy],
                            [0, 0, 1]])

    @property
    def skew(self):
        """float : The skew of the camera in pixels.
        """
        return self._skew

    @property
    def height(self):
        """float : The height of the camera image in pixels.
        """
        return self._height

    @property
    def width(self):
        """float : The width of the camera image in pixels
        """
        return self._width

    @property
    def proj_matrix(self):
        """:obj:`numpy.ndarray` : The 3x3 projection matrix for this camera.
        """
        return self._K

    @property
    def K(self):
        """:obj:`numpy.ndarray` : The 3x3 projection matrix for this camera.
        """
        return self._K

    @K.setter
    def K(self, K_mat):
        self._fx = K_mat[0, 0]
        self._skew = K_mat[0, 1]
        self._cx = K_mat[0, 2]
        self._fy = K_mat[1, 1]
        self._cy = K_mat[1, 2]
        self._K = np.array([[self._fx, self._skew, self._cx],
                            [0, self._fy, self._cy],
                            [0, 0, 1]])

    @property
    def vec(self):
        """:obj:`numpy.ndarray` : Vector representation for this camera.
        """
        return np.r_[self.fx, self.fy, self.cx, self.cy, self.skew, self.height, self.width]

    @staticmethod
    def from_vec(vec):
        K = np.array([[vec[0], vec[4], vec[2]],
                            [0, vec[1], vec[3]],
                            [0, 0, 1]])
        return CameraIntrinsics(K=K,
                                height=vec[5],
                                width=vec[6])

    def crop(self, height, width, crop_ci, crop_cj):
        """ Convert to new camera intrinsics for crop of image from original camera.

        Parameters
        ----------
        height : int
            height of crop window
        width : int
            width of crop window
        crop_ci : int
            row of crop window center
        crop_cj : int
            col of crop window center

        Returns
        -------
        :obj:`CameraIntrinsics`
            camera intrinsics for cropped window
        """
        cx = self.cx + float(width - 1) / 2 - crop_cj
        cy = self.cy + float(height - 1) / 2 - crop_ci
        K = np.array([[self._fx, self._skew, cx],
                            [0, self._fy, cy],
                            [0, 0, 1]])
        cropped_intrinsics = CameraIntrinsics(K,
                                              height=height,
                                              width=width)
        return cropped_intrinsics

    def resize(self, scale_x, scale_y):
        """ Convert to new camera intrinsics with parameters for resized image.

        Parameters
        ----------
        scale : float
            the amount to rescale the intrinsics

        Returns
        -------
        :obj:`CameraIntrinsics`
            camera intrinsics for resized image
        """
        center_x = float(self.width - 1) / 2
        center_y = float(self.height - 1) / 2
        orig_cx_diff = self.cx - center_x
        orig_cy_diff = self.cy - center_y
        height = scale_y * self.height
        width = scale_x * self.width
        scaled_center_x = float(width - 1) / 2
        scaled_center_y = float(height - 1) / 2
        fx = scale_x * self.fx
        fy = scale_y * self.fy
        skew = scale_x * self.skew
        cx = scaled_center_x + scale_x * orig_cx_diff
        cy = scaled_center_y + scale_y * orig_cy_diff
        K = np.array([[fx, skew, cx],
                      [0, fy, cy],
                      [0, 0, 1]])
        scaled_intrinsics = CameraIntrinsics(K=K,
                                             height=height, width=width)
        return scaled_intrinsics

