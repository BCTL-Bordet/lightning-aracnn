import warnings
from typing import List, Optional, Tuple

import hydra
import pandas as pd
import pyrootutils
from os import path
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from tqdm import tqdm
from joblib import Parallel, delayed

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from pyaracnn.tiling.tiler import ImageAnnotTiler
from pyaracnn.utils import get_pylogger, sglob

log = get_pylogger(__name__)


def tile(
    annot_path, wsi_path, output_dir, tile_size, tile_annotation, prefix, normalization
):
    tiler = ImageAnnotTiler(
        annot_path,
        wsi_path,
        output_dir,
        tile_size,
        tile_size,
        tile_annotation=tile_annotation,
        prefix=prefix,
        normalization=normalization,
    )
    tiler.tile()
    del tiler
    return


# @utils.task_wrapper
def tile_dataset(cfg: DictConfig) -> Tuple[dict, dict]:
    wsi_paths = sglob(cfg.paths.wsi_dir, "*.ndpi")
    annot_paths = sglob(cfg.paths.annot_dir, "*.png")
    tile_annotations = cfg.tile_annotations
    prefix = cfg.prefix
    normalization = cfg.normalization

    # files_info = pd.read_csv(cfg.files_info)

    # for wsi_path, annot_path in tqdm(zip(wsi_paths, annot_paths), total=len(wsi_paths)):
    #     tile(annot_path, wsi_path, cfg.local.out_path, cfg.tile_size, tile_annotations)

    _ = Parallel(n_jobs=20)(
        delayed(tile)(
            annot_path,
            wsi_path,
            cfg.local.out_path,
            cfg.tile_size,
            tile_annotations,
            prefix,
            normalization,
        )
        for wsi_path, annot_path in tqdm(
            zip(wsi_paths, annot_paths), total=len(wsi_paths)
        )
    )


@hydra.main(
    version_base="1.3", config_path="../configs", config_name="tile_dataset.yaml"
)
def main(cfg: DictConfig) -> Optional[float]:
    tile_dataset(cfg)


if __name__ == "__main__":
    main()
