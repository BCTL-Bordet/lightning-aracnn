import multiprocessing as mp
import warnings
from os.path import basename, join, splitext
from time import time
from typing import List, Tuple
from histomicstk.preprocessing.color_normalization import reinhard

import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from openslide import OpenSlide
from PIL import Image
from tqdm import tqdm


from lightning_aracnn.tiling.morphology import Tile, WSIMorphology
from lightning_aracnn.utils import get_pylogger, makedir

Image.MAX_IMAGE_PIXELS = 2e30

log = get_pylogger(__name__)


class ImageAnnotTiler:
    #

    def __init__(
        self,
        annot_path: str,
        wsi_path: str,
        out_dir: str,
        tile_size: int = 256,
        step_size: int = 256,
        background_idx: int = 0,
        fat_tissue_idx: int = 5,
        custom_ds_level: int = 2,
        wsi_ds_level: int = 1,
        shift: float = 0.5,
        tile_annotation: bool = True,
        prefix: str = "",
        normalization: str = None,
        to_disk: bool = True,
        num_workers: int = 4,
    ):
        """
        Image object with extended tiling properties

        Parameters
        ----------
        path : str
            path to the image file
        out_dir : str
            path to output the tiles and visualizations
        tile_size : int, optional
            tile size after tiling, by default 256
        step_size : int, optional
            stride between two consecutive tiles, by default 256
        background_idx : List[int], optional
            index of the background class, by default 0
        fat_tissue_idx : List[int], optional
            index of the fat tissue class, by default 5
        custom_ds_level : int, optional
            custom downsample level for tile extraction, by default 2
        wsi_ds_level : int, optional
            downsample level between slide and annotation, by default 1
        shift : float, optional
            center shift factor for bbox checking, by default 0.5
        """
        self.annot_path = annot_path
        self.sample_id = splitext(basename(annot_path))[0]

        self.annot = Image.open(annot_path)
        self.wsi = OpenSlide(wsi_path)

        self.tile_annotation = tile_annotation
        self.normalization = normalization

        self.to_disk = to_disk
        self.num_workers = num_workers

        # color norm. standard (from TCGA-A2-A3XS-DX1, Amgad et al, 2019)
        self.cnorm = {
            "mu": np.array([8.74108109, -0.12440419, 0.0444982]),
            "sigma": np.array([0.6135447, 0.10989545, 0.0286032]),
        }

        try:
            self.wsi_ds_level = self.wsi.level_dimensions.index(self.annot.size)
        except ValueError:
            log.error(
                f"Annotation of sample {basename(annot_path)} is not a perfect downsample of the WSI"
            )
            exit(1)

        self.palette = [c for rgb in Image.open(annot_path).palette.colors for c in rgb]
        self.annot = np.array(self.annot)
        self.canvas = np.zeros_like(self.annot)

        self.background_idx = background_idx
        self.fat_tissue_idx = fat_tissue_idx

        self.custom_ds_level = custom_ds_level

        # convert downsampling level into power of 2
        self.custom_ds = 2**custom_ds_level
        self.wsi_ds = 2**wsi_ds_level

        # np.ndarray is transposed wrt PIL.Image                \/
        self.ds_size = np.round(np.array(self.annot.shape[:2][::-1]) / self.custom_ds).astype(
            np.int32
        )

        # downscale the tile and step size
        self.tile_size = (tile_size // self.custom_ds) // self.wsi_ds
        self.step_size = (step_size // self.custom_ds) // self.wsi_ds

        self.shift = shift

        self.canvas_out_dir = makedir(join(out_dir, prefix, "grid"))
        self.annot_out_dir = makedir(join(out_dir, prefix, "annot_tiles"))
        self.wsi_out_dir = makedir(join(out_dir, prefix, "wsi_tiles"))

        # self.tile()

    def tile(self):
        """
        (Entrypoint) Extract tiles from image
        """
        log.info(f"Tiling annotation {self.sample_id}")

        # rescale annotation to find grid faster
        img = self.__rescale(self.annot)
        # rescale empty canvas
        self.canvas = self.__rescale(self.canvas)

        # binarize annotation
        img = self.__binarize(img)

        # find countours, holes and bounding box
        self.contours, self.holes = self.__contour(img)

        log.info(f"extracting coordinates")
        # extract coordinates grid
        self.coords = self.__extract_coords(img)

        log.info(f"Saving tiles")

        infos = self.__extract_tiles(img)

        canvas = Image.fromarray(self.canvas).convert("P")
        canvas.putpalette(self.palette)
        canvas.save(join(self.canvas_out_dir, f"{self.sample_id}.png"))

        log.info(f"Finished tiling annotation {self.sample_id}")

        return infos

    def __normalize_reinhard(self, image, mask):
        # get an "outside" mask (1 when not tissue), fat tissue(5) and hole(11) are considered background
        mask = np.isin(mask, (0, 11, 5))
        normalized = reinhard(
            np.array(image),
            target_mu=self.cnorm["mu"],
            target_sigma=self.cnorm["sigma"],
            mask_out=mask,
        )

        return normalized

    def __rescale(self, img: np.ndarray) -> np.ndarray:
        """
        Rescale image

        Parameters
        ----------
        img : np.ndarray
            Numpy array of shape (W, H, C)

        Returns
        -------
        np.ndarray
            Rescaled copy of the image
        """
        return cv2.resize(img, self.ds_size, interpolation=cv2.INTER_NEAREST)

    def __binarize(self, img: np.ndarray) -> np.ndarray:
        """
        Binarize annotation by putting at 0 the indices of the non-background classes

        Parameters
        ----------
        img : np.ndarray
            Numpy array of shape (W, H, C)

        Returns
        -------
        np.ndarray
            Binarized copy of the image
        """
        img = np.where(np.isin(img, self.background_idx), 0, 255).astype(np.uint8)

        # img = (img * 255).astype(np.uint8)

        return img

    # def __extract_coords(self) -> None:
    #     """
    #     Extract tiles at coordinates
    #     """

    #     # pool = mp.Pool(4)
    #     # iterable = [coord for coord in self.coords]
    #     # _ = pool.starmap(self._extract_tile, iterable)
    #     # pool.close()

    #     [self.__extract_tile(coord) for coord in self.coords]

    def __contour(self, img: np.ndarray) -> Tuple[List[int], List[int]]:
        """
        Find contour lines of an image

        Parameters
        ----------
        img : np.ndarray
            Numpy array of shape (W, H, C)

        Returns
        -------
        _type_
            _description_
        """
        contours, holes = WSIMorphology.find_contours(img)

        return contours, holes

    def _tile_contour(
        self,
        img: np.ndarray,
        contour: np.ndarray,
        holes: List[np.ndarray],
    ):
        if contour is None:
            print("contour is none... exiting")
            exit()

        # get rectangular area aroun the contour line
        start_x, start_y, w, h = cv2.boundingRect(contour)

        img_w, img_h = img.shape[:2][::-1]

        # accomodate last tile
        stop_y = min(start_y + h, img_h - self.tile_size)
        stop_x = min(start_x + w, img_w - self.tile_size)

        # compute coordinate mesh
        x_range = np.arange(start_x, stop_x, step=self.tile_size)
        y_range = np.arange(start_y, stop_y, step=self.tile_size)
        x_coords, y_coords = np.meshgrid(x_range, y_range, indexing="ij")

        coords = np.array([x_coords.flatten(), y_coords.flatten()]).astype(np.float32).transpose()

        # buil process pool
        # num_workers = max(mp.cpu_count(), 4)
        pool = mp.Pool(self.num_workers)

        # check if each coordinate is inside contour (and outside the corresponding holes)
        iterable = [
            (
                Tile(tuple(coord), self.tile_size),
                contour,
                holes,
                WSIMorphology._is_in_contour_v3_easy,
            )
            for coord in coords
        ]

        results = pool.starmap(WSIMorphology.is_in_tissue, iterable)
        pool.close()

        # filter the coordinates
        coords_inside = coords[results]

        return coords_inside

    def __extract_coords(self, img: np.ndarray):
        coords = []
        for contour, holes in zip(self.contours, self.holes):
            coords_contour = self._tile_contour(img, contour, holes)
            coords.extend(coords_contour)

        coords = list(np.array(coords).astype(np.int32))

        return coords

    def __extract_tiles(self, img: np.ndarray):
        infos = []
        for coord in tqdm(self.coords, total=len(self.coords)):
            infos.append(self.__extract_tile(coord, img))

        return infos

    def __extract_tile(self, coord, img: np.ndarray):
        # TODO: Split function between wsi and annot

        # coordinates were computed on a downsampled version of the image
        # upscaling to the original dimension
        coord_up = coord * self.custom_ds

        # same goes for tile size
        tile_size_up = self.tile_size * self.custom_ds

        # extract wsi crop
        wsi_crop = self.wsi.read_region(
            location=(coord_up[0] * self.wsi_ds, coord_up[1] * self.wsi_ds),
            level=0,
            size=(tile_size_up * self.wsi_ds, tile_size_up * self.wsi_ds),
        ).convert("RGB")

        if self.tile_annotation:
            # read the corresponding crop from original image
            annot_crop = self.annot[
                coord_up[1] : coord_up[1] + tile_size_up,
                coord_up[0] : coord_up[0] + tile_size_up,
            ]

            # canvas is still downsampled wrt the original image, so let's downsample the resulting crop
            # nearest neighbour interpolation to avoid creating new colors (labels)
            annot_crop_ds = cv2.resize(
                annot_crop,
                np.array((self.tile_size,) * 2),
                interpolation=cv2.INTER_NEAREST,
            )

            # paint the crop at the correct position into the canvas
            self.canvas[
                coord[1] : coord[1] + self.tile_size,
                coord[0] : coord[0] + self.tile_size,
            ] = annot_crop_ds

            self.canvas = cv2.rectangle(
                self.canvas,
                (coord[0], coord[1]),
                (coord[0] + self.tile_size, coord[1] + self.tile_size),
                color=(255, 0, 0),
                thickness=3,
            )

            # upscale the crop to the wsi resolution
            # nearest neighbour interpolation to avoid creating new colors (labels)
            annot_crop_up = cv2.resize(
                annot_crop,
                np.array(annot_crop.shape) * self.wsi_ds,
                interpolation=cv2.INTER_NEAREST,
            )

            # put back original palette
            annot_crop_up = Image.fromarray(annot_crop_up).convert("P")
            annot_crop_up.putpalette(self.palette)

            # organize cropping point to name files (or return as info if self.to_disk is false)
            p0_wsi = (coord_up[0] * self.wsi_ds, coord_up[1] * self.wsi_ds)
            p1_wsi = (
                coord_up[0] * self.wsi_ds + tile_size_up * self.wsi_ds,
                coord_up[1] * self.wsi_ds + tile_size_up * self.wsi_ds,
            )

            p0_annot = (coord_up[0], coord_up[1])
            p1_annot = (
                coord_up[0] + tile_size_up,
                coord_up[1] + tile_size_up,
            )
            
            if self.normalization == "reinhard":
                wsi_crop = self.__normalize_reinhard(wsi_crop, annot_crop_up)

            if self.to_disk:
                wsi_crop = Image.fromarray(wsi_crop)
                wsi_crop.save(
                    join(
                        self.wsi_out_dir,
                        f"{self.sample_id}-{p0_wsi[0]}_{p0_wsi[1]}.png",
                    ),
                )
                # (note that the coordinates in the annotation crop are not correct because we need names to align, they should be downsampled)
                annot_crop_up.save(
                    join(
                        self.annot_out_dir,
                        f"{self.sample_id}-{p0_wsi[0]}_{p0_wsi[1]}.png",
                    )
                )
            else:

                return (
                    wsi_crop,
                    (p0_wsi, p1_wsi),
                    annot_crop_up,
                    (p0_annot, p1_annot),
                )


class WSITiler:
    def __init__(
        self,
        path: str,
        out_dir: str,
        coords: np.ndarray,
        tile_size: int,
        custom_ds_level: int,
        wsi_ds_level: int,
        contours: List[np.ndarray],
        holes: List[List[np.ndarray]],
    ):
        self.sample_id = splitext(basename(path))[0]
        self.coords = coords
        self.tile_size = tile_size
        self.contours = contours
        self.holes = holes

        self.ds_level = custom_ds_level + wsi_ds_level
        self.ds = 2**self.ds_level

        self.image = OpenSlide(path)
        self.canvas = np.array(
            self.image.read_region(
                (0, 0),
                self.ds_level,
                self.image.level_dimensions[self.ds_level],
            ).convert("RGB")
        )

        self.grids_out_dir = makedir(join(out_dir, "grids"))
        self.tiles_out_dir = makedir(join(out_dir, "slide_tiles"))

        self._draw_grid()

    # def _draw_grid(self, color=(255, 0, 255), thickness: int = 3):
    #     self.canvas = WSIMorphology.draw_on_slide(
    #         self.canvas, self.contours, self.holes
    #     )

    #     tile_size = self.tile_size // self.ds

    #     for coord in self.coords:
    #         cv2.rectangle(
    #             self.canvas,
    #             tuple(coord),
    #             tuple(coord + tile_size),
    #             color,
    #             thickness,
    #         )

    #     Image.fromarray(self.canvas).save(
    #         join(self.grids_out_dir, f"{self.sample_id}_slide.jpg")
    #     )

    def tile_slide(self):
        log.info(f"Tiling slide {self.sample_id}")
        log.info(f"Saving tiles")
        self._extract_tiles()
        log.info(f"Finished tiling annotation {self.sample_id}")

    def _extract_tiles(self):
        for coord in tqdm(self.coords, total=len(self.coords)):
            self._extract_tile(coord)

    def _extract_tile(self, coord):
        coord_ds = coord * self.ds
        tile_size = self.tile_size

        crop = self.image.read_region(
            location=coord_ds,
            level=0,
            size=(tile_size, tile_size),
        )
        # crop.resize(np.array(crop.size) * self.ds_to_wsi)

        # crop = Image.fromarray(crop)
        crop.save(join(self.tiles_out_dir, f"{self.sample_id}-{coord[0]}_{coord[1]}.png"))


if __name__ == "__main__":
    tiler = ImageAnnotTiler(
        annot_path="/mnt/bctl/nocc0001/lobular/lobular_clean_annotations_2/ST001.png",
        wsi_path="/mnt/bctl/nocc0001/lobular/lobular_slides/ST001.ndpi",
        out_dir="/home/nocc0001/tmp",
    )
    tiler.tile()
