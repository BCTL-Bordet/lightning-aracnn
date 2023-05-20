from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Tuple

import cv2
import numpy as np
from openslide import OpenSlide


class Tile:
    def __init__(self, top_left: Tuple[int, int], size: int):
        """A Tile is a square element on an image.

        Parameters
        ----------
        top_left : Tuple[int, int]
            X, Y coordinates of the top-left pixel of the square
        size : int
            size of the square
        """
        self.top_left = top_left
        self.size = size
        self.center: Tuple[int, int] = (
            top_left[0] + size // 2,
            top_left[1] + size // 2,
        )

    def get_shrunk_corners(self, center_shift: float = 0.5) -> List[Tuple[int, int]]:
        """
        Returns the corner of the shrunk version of the tile.
        The tile is shrinked to a subtile whose diagonal = size * center_shift

        Args:
            center_shift (float, optional): Shrink factor. Defaults to 0.5.

        Returns:
            List[Tuple[int, int]]: List of shrunk corners
        """
        shift = int(self.size // 2 * center_shift)

        if shift > 0:
            corners = [
                (self.center[0] - shift, self.center[1] - shift),
                (self.center[0] + shift, self.center[1] + shift),
                (self.center[0] + shift, self.center[1] - shift),
                (self.center[0] - shift, self.center[1] + shift),
            ]
        else:
            corners = [self.center]

        return corners


class WSIMorphology:
    """
    Wraps cv2.contour objects for ease of use
    """

    ###
    @staticmethod
    def find_contours(img: np.ndarray):
        # find contours in binary image
        contours, hierarchy = cv2.findContours(
            img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE
        )
        contours = np.array(contours, dtype=object)

        # hierarchy information:
        # [next, previous, 1st_child, parent]
        # keep only information about 1st_child and parent for each contour line
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]

        # necessary for filtering out artifacts
        contours, holes = WSIMorphology._retrieve_contours_holes(
            contours,
            hierarchy,
        )

        # scale contours back up to the original WSI size

        return contours, holes

    ###
    @staticmethod
    def _retrieve_contours_holes(contours: np.ndarray, hierarchy: np.ndarray):
        # https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html
        # https://docs.opencv.org/4.x/d9/d8b/tutorial_py_contours_hierarchy.html

        # find indices of foreground contours (parent == -1)
        hierarchy_1 = np.flatnonzero(hierarchy[:, 1] == -1)

        # list of contours
        filtered_contours: List[np.ndarray] = []

        # list of holes for each contour
        contours_holes: List[List[np.ndarray]] = []

        # loop throug hierarchy_1 contour indices
        # (islands of tissue in the slide)
        for contour_idx in hierarchy_1:
            contour = contours[contour_idx]

            # indices of holes contained in this contour (contours whose parent is the current contour)
            # (lakes in the island)
            holes = contours[hierarchy[:, 1] == contour_idx]

            # compute the contour area
            # (surface of the island, lakes included)
            contour_area = cv2.contourArea(contour)

            # take the area of the holes
            # (surface of the lakes)
            hole_areas = [cv2.contourArea(hole) for hole in holes]

            # get contour area minus the holes
            # (surface of the land in the island)
            contour_area = contour_area - np.array(hole_areas).sum()

            # retain contour if it is larger than the minimum area required
            # if contour_area > 0 and contour_area > self.min_tumor_area:
            if contour_area > 0:
                filtered_contours.append(contour)
                contours_holes.append(holes)

        filtered_holes: List[List[np.ndarray]] = []
        # for the holes of each contour
        for holes in contours_holes:
            # sort holes descending by area
            holes = sorted(holes, key=cv2.contourArea, reverse=True)

            # take max_n_holes largest holes by area
            # holes = holes[: self.max_n_holes]

            # filter holes by minimum area
            # holes = [hole for hole in holes if cv2.contourArea(hole) > self.min_hole_area]

            filtered_holes.append(holes)

        return filtered_contours, filtered_holes

    @staticmethod
    def draw_on_slide(
        image: np.ndarray,
        contours: List[np.ndarray],
        holes: List[List[np.ndarray]],
        line_thickness: int = 5,
        tissue_color: Tuple[int, int, int] = (0, 255, 0),
        hole_color: Tuple[int, int, int] = (0, 0, 255),
    ) -> np.ndarray:
        if contours is not None:
            cv2.drawContours(
                image,
                contours,
                -1,
                tissue_color,
                line_thickness,
                lineType=cv2.LINE_8,
            )
            for hole in holes:
                cv2.drawContours(
                    image,
                    hole,
                    -1,
                    hole_color,
                    line_thickness,
                    lineType=cv2.LINE_8,
                )

        return image

    @staticmethod
    def _point_in_contour(
        coord: tuple[int, int], contour: np.ndarray, allow_edge: bool = True
    ) -> bool:
        """
        Wrapper for cv2.pointPolygonTest

        Args:
            coord (np.ndarray): Point coordinate
            contour (np.ndarray): Contour line

        Returns:
            bool: returns True if point is strictly inside the contour line. False otherwise.

        """
        if allow_edge:
            return cv2.pointPolygonTest(contour, tuple(coord), False) >= 0
        else:
            return cv2.pointPolygonTest(contour, tuple(coord), False) > 0

    @staticmethod
    def _is_in_contour_v1(tile: Tile, contour: np.ndarray) -> bool:
        """
        Checks if the top-left corner of a tile is in a given contour.

        Args:
            tile (Tile): Tile object
            contour (np.ndarray): Contour line

        Returns:
            bool: returns True if coord is strictly inside the contour line. False otherwise.
        """
        return WSIMorphology._point_in_contour(tile.top_left, contour)

    @staticmethod
    def _is_in_contour_v2(tile: Tile, contour: np.ndarray) -> bool:
        """
        Checks if the center of a tile is in a given contour.

        Args:
            tile (Tile): Tile object
            contour (np.ndarray): Contour line

        Returns:
            bool: returns True if coord is strictly inside the contour line. False otherwise.
        """

        return WSIMorphology._point_in_contour(tile.center, contour)

    @staticmethod
    def _is_in_contour_v3(tile: Tile, contour: np.ndarray, center_shift: float = 0.5):
        corners = tile.get_shrunk_corners(center_shift)

        # test if corners are inside the contour
        return [WSIMorphology._point_in_contour(corner, contour) for corner in corners]

    @staticmethod
    def _is_in_contour_v3_easy(
        tile: Tile, contour: np.ndarray, center_shift: float = 0.5
    ):
        inside = WSIMorphology._is_in_contour_v3(tile, contour, center_shift)

        # easy only requires one corner to be contained in the contour
        return any(inside)

    @staticmethod
    def _is_in_contour_v3_hard(
        tile: Tile, contour: np.ndarray, center_shift: float = 0.5
    ):
        inside = WSIMorphology._is_in_contour_v3(tile, contour, center_shift)

        # easy requires all corners to be contained in the contour
        return all(inside)

    @staticmethod
    def _is_in_holes(tile: Tile, holes: List[np.ndarray]) -> bool:
        # check if tile is outside every hole
        outside = [
            not WSIMorphology._point_in_contour(tile.center, hole, allow_edge=False)
            for hole in holes
        ]

        return not all(outside)

    @staticmethod
    def is_in_tissue(
        tile: Tile,
        contour: np.ndarray,
        holes: List[np.ndarray],
        contour_check_fn: Callable,
    ) -> bool:
        # if tile is outside tissue, we return False
        if not contour_check_fn(tile, contour):
            return False

        # if there are no holes, we return True
        if not holes or len(holes) == 0:
            return True

        # otherwise we check if the center of the tile is strictly outside all hole
        return not WSIMorphology._is_in_holes(tile, holes)
