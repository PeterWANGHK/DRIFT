import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from track_visualizer import TrackVisualizer, DataError
from tracks_import import get_rotated_bbox, read_from_dataset

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_DIR_NAMES = {
    "exid": "exiD",
    "ind": "inD",
    "round": "rounD",
    "roundd": "rounD",
    "unid": "uniD",
    "highd": "highD",
    "sqm-n-4": "SQM-N-4",
    "ytdj-3": "YTDJ-3",
    "xam-n-5": "XAM-N-5",
    "xam-n-6": "XAM-N-6",
    "pkdd-8": "PKDD-8",
}
SPECIAL_DATASET_NAMES = {"SQM-N-4", "YTDJ-3", "XAM-N-5", "XAM-N-6", "PKDD-8"}


def create_args():
    cs = argparse.ArgumentParser(description="Dataset Tracks Visualizer")
    # --- Input ---
    cs.add_argument(
        '--dataset_dir',
        default=None,
        help="Path to the directory that contains the dataset csv files. Defaults to data/<dataset>.",
        type=str,
    )
    cs.add_argument('--dataset', default="exid",
                    help="Name of the dataset. Needed to apply dataset specific visualization adjustments.",
                    type=str)
    cs.add_argument('--recording', default="26",
                    help="Name of the recording given by a number with a leading zero.", type=str)
    cs.add_argument(
        '--visualizer_params_dir',
        default=None,
        help="Optional directory or file path for visualizer_params.json.",
        type=str,
    )

    # --- Visualization settings ---
    cs.add_argument('--playback_speed', default=4,
                    help="During playback, only consider every nth frame. This option also applies to the outer"
                         "backward/forward jump buttons.",
                    type=int)
    cs.add_argument('--suppress_track_window', default=False,
                    help="Do not show the track window when clicking on a track. Only surrounding vehicle colors are"
                         " displayed.",
                    type=str2bool)
    cs.add_argument('--show_bounding_box', default=True,
                    help="Plot the rotated bounding boxes of all vehicles. Please note, that for vulnerable road users,"
                         " no bounding box is given.",
                    type=str2bool)
    cs.add_argument('--show_orientation', default=False,
                    help="Indicate the orientation of all vehicles by triangles.",
                    type=str2bool)
    cs.add_argument('--show_trajectory', default=False,
                    help="Show the trajectory up to the current frame for every track.",
                    type=str2bool)
    cs.add_argument('--show_future_trajectory', default=False,
                    help="Show the remaining trajectory for every track.",
                    type=str2bool)
    cs.add_argument('--annotate_track_id', default=True,
                    help="Annotate every track by its id.",
                    type=str2bool)
    cs.add_argument('--annotate_class', default=False,
                    help="Annotate every track by its class label.",
                    type=str2bool)
    cs.add_argument('--annotate_speed', default=False,
                    help="Annotate every track by its current speed.",
                    type=str2bool)
    cs.add_argument('--annotate_orientation', default=False,
                    help="Annotate every track by its current orientation.",
                    type=str2bool)
    cs.add_argument('--annotate_age', default=False,
                    help="Annotate every track by its current age.",
                    type=str2bool)
    cs.add_argument('--show_maximized', default=False,
                    help="Show the track Visualizer maximized. Might affect performance.",
                    type=str2bool)

    return vars(cs.parse_args())


def canonical_dataset_dirname(dataset: str) -> str:
    return DATASET_DIR_NAMES.get(dataset.lower(), dataset)


def resolve_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return path.resolve()


def resolve_dataset_dir(dataset: str, dataset_dir: str | None) -> Path:
    dataset_folder = canonical_dataset_dirname(dataset)
    if dataset_dir is None:
        return (SCRIPT_DIR / "data" / dataset_folder).resolve()

    resolved = resolve_path(dataset_dir)
    if resolved is None:
        return (SCRIPT_DIR / "data" / dataset_folder).resolve()

    if not any(resolved.glob("*_tracks.csv")) and (resolved / dataset_folder).exists():
        nested_candidate = (resolved / dataset_folder).resolve()
        if any(nested_candidate.glob("*_tracks.csv")):
            return nested_candidate

    return resolved


def resolve_visualizer_params_path(path_value: str | None) -> str | None:
    resolved = resolve_path(path_value)
    if resolved is None:
        return None
    return str(resolved)


def resolve_background_image_path(dataset_dir: Path, recording: str) -> Path | None:
    candidates = [
        dataset_dir / f"{recording}_background.png",
        dataset_dir / f"{recording}_highway.png",
        dataset_dir / f"{recording}_background.jpg",
        dataset_dir / f"{recording}_highway.jpg",
        dataset_dir / "road map.png",
        dataset_dir / "road_map.png",
        dataset_dir / "background.png",
        dataset_dir / "highway.png",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def align_highd_tracks_to_background(tracks: list[dict], background_image_path: str | None) -> None:
    if not tracks or not background_image_path:
        return

    background_image = cv2.imread(background_image_path)
    if background_image is None:
        logger.warning("Could not load highD background image from {} for alignment.", background_image_path)
        return

    image_height, image_width = background_image.shape[:2]
    all_centers = np.concatenate([track["centerVis"] for track in tracks], axis=0)
    x_min = float(np.nanmin(all_centers[:, 0]))
    x_max = float(np.nanmax(all_centers[:, 0]))
    y_min = float(np.nanmin(all_centers[:, 1]))
    y_max = float(np.nanmax(all_centers[:, 1]))

    x_span = max(1e-6, x_max - x_min)
    y_span = max(1e-6, y_max - y_min)
    scale = (image_width - 1.0) / x_span
    used_height = y_span * scale
    y_offset = (image_height - used_height) / 2.0

    for track in tracks:
        x_center_vis = (track["xCenterVis"] - x_min) * scale
        y_center_vis = (track["yCenterVis"] - y_min) * scale + y_offset
        track["xCenterVis"] = x_center_vis
        track["yCenterVis"] = y_center_vis
        track["centerVis"] = np.stack([x_center_vis, y_center_vis], axis=-1)
        track["widthVis"] = track["widthVis"] * scale
        track["lengthVis"] = track["lengthVis"] * scale
        if track["bboxVis"] is not None:
            track["bboxVis"] = get_rotated_bbox(
                track["xCenterVis"],
                track["yCenterVis"],
                track["lengthVis"],
                track["widthVis"],
                np.deg2rad(track["headingVis"]),
            )

    logger.info(
        "Aligned highD tracks to background {} using scale {:.3f}px/m, x range [{:.2f}, {:.2f}], y range [{:.2f}, {:.2f}].",
        background_image_path,
        scale,
        x_min,
        x_max,
        y_min,
        y_max,
    )


def main():
    config = create_args()

    dataset = canonical_dataset_dirname(config["dataset"])
    dataset_dir = resolve_dataset_dir(dataset, config["dataset_dir"])
    recording = config["recording"]

    if recording is None:
        logger.error("Please specify a recording!")
        sys.exit(1)

    recording = "{:02d}".format(int(recording))
    config["dataset"] = dataset
    config["dataset_dir"] = str(dataset_dir)
    config["visualizer_params_dir"] = resolve_visualizer_params_path(config["visualizer_params_dir"])
    if dataset in SPECIAL_DATASET_NAMES:
        config["recording"] = dataset

    logger.info("Loading recording {} from dataset {}", recording, config["dataset"])

    tracks, static_info, meta_info = read_from_dataset(
        str(dataset_dir),
        dataset_name=dataset,
        recording=recording,
        include_px_coordinates=True,
    )

    # Load background image for visualization
    background_image_path = resolve_background_image_path(dataset_dir, recording)
    if background_image_path is None:
        logger.warning(
            "No background image found for recording {} in {}. Fallback to using a black background.",
            recording,
            dataset_dir,
        )
    elif dataset.lower() == "highd":
        align_highd_tracks_to_background(tracks, str(background_image_path))
    config["background_image_path"] = str(background_image_path) if background_image_path is not None else None

    try:
        visualization_plot = TrackVisualizer(config, tracks, static_info, meta_info)
        visualization_plot.show()
    except DataError:
        sys.exit(1)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':
    main()
