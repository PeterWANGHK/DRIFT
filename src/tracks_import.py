import glob
import os
import pickle
from typing import List, Tuple

import numpy as np
import pandas
from PIL import Image
from loguru import logger


N_MAX_OVERLAPPING_LANELETS = 5
ALONGSIDE_ID_COLUMNS = {"leftAlongsideId", "rightAlongsideId"}
PADDED_INT_COLUMNS = {"laneletId"}
PADDED_FLOAT_COLUMNS = {"latLaneCenterOffset", "lonLaneletPos", "laneletLength", "laneWidth"}
HIGH_D_NEIGHBOR_RENAMES = {
    "precedingId": "leadId",
    "followingId": "rearId",
    "leftPrecedingId": "leftLeadId",
    "leftFollowingId": "leftRearId",
    "rightPrecedingId": "rightLeadId",
    "rightFollowingId": "rightRearId",
}
SPECIAL_DATASET_CACHE_VERSION = 3
SPECIAL_DATASET_LAYOUTS = {
    "sqm-n-4": {
        "frame_rate": 30.0,
        "speed_limit_mps": 80.0 / 3.6,
        "section_length_m": 387.0,
        "location_id": 401,
        "background_name": "road map.png",
        "x_margin_frac": (0.002, 0.998),
        "lateral_px_per_m": 6.6,
        "lane_center_profiles_px": {
            "up": {
                1: [(94.0, 649.5), (356.0, 626.5)],
                2: [(0.0, 649.5)],
                3: [(0.0, 672.5)],
                4: [(0.0, 696.0)],
                5: [(0.0, 719.0)],
            },
            "down": {
                1: [(0.0, 742.5)],
                2: [(0.0, 766.5)],
                3: [(0.0, 789.5)],
                4: [(0.0, 812.5), (224.0, 812.5), (356.0, 888.5)],
                5: [(224.0, 888.5), (356.0, 968.5)],
            },
        },
        "direction_offsets": {"up": 0, "down": 1_000_000},
        "reverse_x_for_direction": {"up": False, "down": False},
    },
    "ytdj-3": {
        "frame_rate": 24.0,
        "speed_limit_mps": 80.0 / 3.6,
        "section_length_m": 362.0,
        "location_id": 402,
        "background_name": "road map.png",
        "x_margin_frac": (0.002, 0.998),
        "lateral_px_per_m": 13.6,
        "lane_center_profiles_px": {
            1: [(0.0, 701.0)],
            2: [(0.0, 649.5)],
            3: [(0.0, 598.5)],
            4: [(153.0, 598.5), (278.0, 549.5)],
            5: [(189.0, 549.5), (278.0, 504.5)],
            6: [(0.0, 754.0)],
            7: [(0.0, 807.5)],
            8: [(0.0, 860.0)],
            9: [(0.0, 912.0), (185.0, 860.0)],
            10: [(0.0, 962.5), (164.0, 912.0)],
        },
        "fallback_center_y_px": {
            "upper": 650.0,
            "lower": 807.5,
        },
        "reverse_x_for_lane": {1: True, 2: True, 3: True, 4: True, 5: True},
    },
    "xam-n-5": {
        "frame_rate": 30.0,
        "speed_limit_mps": 60.0 / 3.6,
        "section_length_m": 140.0,
        "location_id": 403,
        "background_name": "road map.png",
    },
    "xam-n-6": {
        "frame_rate": 60.0,
        "speed_limit_mps": 60.0 / 3.6,
        "section_length_m": 140.0,
        "location_id": 404,
        "background_name": "road map.png",
    },
}
SPECIAL_DATASET_FALLBACK_CENTER_Y_PX = {
    "sqm-n-4": {
        "up": 672.5,
        "down": 789.5,
    },
    "ytdj-3": {
        "upper": 650.0,
        "lower": 807.5,
    },
}
SPECIAL_DATASET_KEYS = set(SPECIAL_DATASET_LAYOUTS)


def read_all_recordings_from_csv(base_path: str = "../data/") -> List[dict]:
    """
    Read tracks and meta information for all recordings in a directory.
    Warning: This might need a lot of memory.
    """
    tracks_files = sorted(glob.glob(base_path + "*_tracks.csv"))
    tracks_meta_files = sorted(glob.glob(base_path + "*_tracksMeta.csv"))
    recording_meta_files = sorted(glob.glob(base_path + "*_recordingMeta.csv"))

    recordings = []
    for track_file, tracks_meta_file, recording_meta_file in zip(
        tracks_files, tracks_meta_files, recording_meta_files
    ):
        logger.info("Loading csv files {}, {} and {}", track_file, tracks_meta_file, recording_meta_file)
        tracks, tracks_meta, recording_meta = read_from_csv(track_file, tracks_meta_file, recording_meta_file)
        recordings.append({"tracks": tracks, "tracks_meta": tracks_meta, "recording_meta": recording_meta})

    return recordings


def read_from_csv(
    tracks_file: str,
    tracks_meta_file: str,
    recording_meta_file: str,
    include_px_coordinates: bool = False,
) -> Tuple[List[dict], List[dict], dict]:
    """
    Read tracks and meta data for a single recording from csv files.
    """
    recording_meta = read_recording_meta(recording_meta_file)
    tracks_meta = read_tracks_meta(tracks_meta_file)
    tracks = read_tracks(tracks_file, recording_meta, include_px_coordinates)
    return tracks, tracks_meta, recording_meta


def read_from_dataset(
    dataset_dir: str,
    dataset_name: str | None = None,
    recording: str | None = None,
    include_px_coordinates: bool = False,
) -> Tuple[List[dict], List[dict], dict]:
    """
    Read one dataset recording, including special non-drone-tool layouts such as SQM-N-4 / YTDJ-3.
    """
    dataset_key = _normalize_dataset_key(dataset_name or os.path.basename(os.path.normpath(dataset_dir)))
    if dataset_key in SPECIAL_DATASET_KEYS:
        return _read_special_dataset(dataset_dir, dataset_key, include_px_coordinates)

    if recording is None:
        raise ValueError("recording must be provided for standard *_tracks.csv datasets.")

    tracks_file = os.path.join(dataset_dir, f"{recording}_tracks.csv")
    tracks_meta_file = os.path.join(dataset_dir, f"{recording}_tracksMeta.csv")
    recording_meta_file = os.path.join(dataset_dir, f"{recording}_recordingMeta.csv")
    return read_from_csv(tracks_file, tracks_meta_file, recording_meta_file, include_px_coordinates)


def read_tracks(tracks_file: str, recording_meta: dict, include_px_coordinates: bool = False) -> List[dict]:
    """
    Read tracks from a csv file and normalize all supported datasets to one internal schema.
    """
    raw_tracks = pandas.read_csv(tracks_file, low_memory=False)
    raw_tracks = _normalize_tracks_dataframe(raw_tracks, recording_meta)
    raw_tracks = raw_tracks.groupby(["trackId"], sort=True)
    ortho_px_to_meter = float(recording_meta.get("orthoPxToMeter", 1.0) or 1.0)

    tracks = []
    for track_id, track_rows in raw_tracks:
        track_rows = track_rows.sort_values("frame")
        track = track_rows.to_dict(orient="list")
        track = _finalize_track_dict(track, track_rows)

        track["center"] = np.stack([track["xCenter"], track["yCenter"]], axis=-1)
        if np.count_nonzero(track["length"]) and np.count_nonzero(track["width"]):
            track["bbox"] = get_rotated_bbox(
                track["xCenter"],
                track["yCenter"],
                track["length"],
                track["width"],
                np.deg2rad(track["heading"]),
            )
        else:
            track["bbox"] = None

        if include_px_coordinates:
            track["xCenterVis"] = track["xCenter"] / ortho_px_to_meter
            track["yCenterVis"] = -track["yCenter"] / ortho_px_to_meter
            track["centerVis"] = np.stack([track["xCenterVis"], track["yCenterVis"]], axis=-1)
            track["widthVis"] = track["width"] / ortho_px_to_meter
            track["lengthVis"] = track["length"] / ortho_px_to_meter
            track["headingVis"] = track["heading"] * -1
            track["headingVis"][track["headingVis"] < 0] += 360
            if np.count_nonzero(track["length"]) and np.count_nonzero(track["width"]):
                track["bboxVis"] = get_rotated_bbox(
                    track["xCenterVis"],
                    track["yCenterVis"],
                    track["lengthVis"],
                    track["widthVis"],
                    np.deg2rad(track["headingVis"]),
                )
            else:
                track["bboxVis"] = None

        tracks.append(track)
    return tracks


def read_tracks_meta(tracks_meta_file: str) -> List[dict]:
    """
    Read tracks meta from a csv file and normalize highD-style fields.
    """
    tracks_meta = pandas.read_csv(tracks_meta_file, low_memory=False).rename(columns={"id": "trackId"})
    if "length" not in tracks_meta.columns and {"width", "height"}.issubset(tracks_meta.columns):
        length = pandas.to_numeric(tracks_meta["width"], errors="coerce")
        width = pandas.to_numeric(tracks_meta["height"], errors="coerce")
        tracks_meta["length"] = length
        tracks_meta["width"] = width
    if "recordingId" not in tracks_meta.columns:
        tracks_meta["recordingId"] = -1
    return sorted(tracks_meta.to_dict(orient="records"), key=lambda entry: entry["trackId"])


def read_recording_meta(recording_meta_file: str) -> dict:
    """
    Read recording meta from a csv file and normalize field names/defaults.
    """
    recording_meta = pandas.read_csv(recording_meta_file, low_memory=False).to_dict(orient="records")[0]
    if "recordingId" not in recording_meta:
        recording_meta["recordingId"] = recording_meta.get("id", recording_meta.get("trackId", -1))
    if "weekday" not in recording_meta and "weekDay" in recording_meta:
        recording_meta["weekday"] = recording_meta["weekDay"]
    recording_meta.setdefault("locationId", 0)
    recording_meta.setdefault("orthoPxToMeter", 1.0)
    return recording_meta


def _normalize_tracks_dataframe(raw_tracks: pandas.DataFrame, recording_meta: dict) -> pandas.DataFrame:
    raw_tracks = raw_tracks.rename(columns={"id": "trackId"}).copy()

    if "trackId" not in raw_tracks.columns:
        raise KeyError("Expected a trackId/id column in the tracks csv.")

    is_highd_like = "xCenter" not in raw_tracks.columns and {"x", "y"}.issubset(raw_tracks.columns)
    if not is_highd_like:
        raw_tracks = _normalize_list_columns(raw_tracks)
        return raw_tracks

    raw_tracks["recordingId"] = int(recording_meta.get("recordingId", -1))

    raw_length = pandas.to_numeric(raw_tracks["width"], errors="coerce")
    raw_width = pandas.to_numeric(raw_tracks["height"], errors="coerce")
    raw_x = pandas.to_numeric(raw_tracks["x"], errors="coerce")
    raw_y = pandas.to_numeric(raw_tracks["y"], errors="coerce")
    raw_x_velocity = pandas.to_numeric(raw_tracks["xVelocity"], errors="coerce")
    raw_y_velocity = pandas.to_numeric(raw_tracks["yVelocity"], errors="coerce")
    raw_x_acceleration = pandas.to_numeric(raw_tracks["xAcceleration"], errors="coerce")
    raw_y_acceleration = pandas.to_numeric(raw_tracks["yAcceleration"], errors="coerce")

    raw_y_center = raw_y + raw_width / 2.0
    x_center = raw_x + raw_length / 2.0
    y_center = -raw_y_center
    y_velocity = -raw_y_velocity
    y_acceleration = -raw_y_acceleration

    heading = np.rad2deg(np.arctan2(y_velocity.to_numpy(), raw_x_velocity.to_numpy()))
    heading[heading < 0.0] += 360.0

    speed = np.hypot(raw_x_velocity.to_numpy(), y_velocity.to_numpy())
    safe_speed = np.where(speed > 1e-9, speed, 1.0)
    lon_acc = (
        raw_x_velocity.to_numpy() * raw_x_acceleration.to_numpy()
        + y_velocity.to_numpy() * y_acceleration.to_numpy()
    ) / safe_speed
    lat_acc = (
        -y_velocity.to_numpy() * raw_x_acceleration.to_numpy()
        + raw_x_velocity.to_numpy() * y_acceleration.to_numpy()
    ) / safe_speed

    raw_tracks["xCenter"] = x_center
    raw_tracks["yCenter"] = y_center
    raw_tracks["length"] = raw_length
    raw_tracks["width"] = raw_width
    raw_tracks["yVelocity"] = y_velocity
    raw_tracks["yAcceleration"] = y_acceleration
    raw_tracks["heading"] = heading
    raw_tracks["lonVelocity"] = speed
    raw_tracks["latVelocity"] = np.zeros_like(speed)
    raw_tracks["lonAcceleration"] = lon_acc
    raw_tracks["latAcceleration"] = lat_acc

    raw_tracks["trackLifetime"] = raw_tracks.groupby("trackId").cumcount()

    for source_key, target_key in HIGH_D_NEIGHBOR_RENAMES.items():
        if source_key in raw_tracks.columns:
            raw_tracks[target_key] = _normalize_id_series(raw_tracks[source_key])

    for alongside_key in ALONGSIDE_ID_COLUMNS:
        if alongside_key in raw_tracks.columns:
            raw_tracks[alongside_key] = raw_tracks[alongside_key].apply(_parse_alongside_ids)

    lead_id = raw_tracks.get("leadId", pandas.Series(-1, index=raw_tracks.index))
    lead_mask = _normalize_id_series(lead_id).to_numpy() != -1
    direction_sign = np.sign(raw_x_velocity.to_numpy())
    direction_sign[direction_sign == 0.0] = 1.0

    if "dhw" in raw_tracks.columns:
        lead_dhw = pandas.to_numeric(raw_tracks["dhw"], errors="coerce").to_numpy()
        lead_dhw[~lead_mask | (lead_dhw <= 0.0)] = -1.0
        raw_tracks["leadDHW"] = lead_dhw

    if "ttc" in raw_tracks.columns:
        lead_ttc = pandas.to_numeric(raw_tracks["ttc"], errors="coerce").to_numpy()
        lead_ttc[~lead_mask | (lead_ttc <= 0.0)] = -1.0
        raw_tracks["leadTTC"] = lead_ttc

    if "thw" in raw_tracks.columns:
        lead_thw = pandas.to_numeric(raw_tracks["thw"], errors="coerce").to_numpy()
        lead_thw[~lead_mask | (lead_thw <= 0.0)] = -1.0
        raw_tracks["leadTHW"] = lead_thw

    if "precedingXVelocity" in raw_tracks.columns:
        lead_velocity = pandas.to_numeric(raw_tracks["precedingXVelocity"], errors="coerce").to_numpy()
        lead_dv = direction_sign * (raw_x_velocity.to_numpy() - lead_velocity)
        lead_dv[~lead_mask] = -1000.0
        raw_tracks["leadDV"] = lead_dv

    raw_tracks["traveledDistance"] = 0.0
    raw_tracks["laneChange"] = 0

    if "laneId" in raw_tracks.columns:
        lane_id = pandas.to_numeric(raw_tracks["laneId"], errors="coerce").fillna(-1).astype(int)
        lane_info = _build_highd_lane_info(recording_meta)
        raw_tracks["laneletId"] = [_pack_numeric_value(v, int) for v in lane_id.to_numpy()]
        if lane_info:
            lane_width = []
            lane_offset = []
            for lane_value, y_center_value in zip(lane_id.to_numpy(), y_center.to_numpy()):
                info = lane_info.get(int(lane_value))
                if info is None or np.isnan(y_center_value):
                    lane_width.append(_empty_padded_list())
                    lane_offset.append(_empty_padded_list())
                    continue
                lane_center_world = -info["center"]
                lane_width.append(_pack_numeric_value(info["width"], float))
                lane_offset.append(_pack_numeric_value(y_center_value - lane_center_world, float))
            raw_tracks["laneWidth"] = lane_width
            raw_tracks["latLaneCenterOffset"] = lane_offset

    raw_tracks = _normalize_list_columns(raw_tracks)
    return raw_tracks


def _normalize_list_columns(raw_tracks: pandas.DataFrame) -> pandas.DataFrame:
    for column in ALONGSIDE_ID_COLUMNS:
        if column in raw_tracks.columns:
            raw_tracks[column] = raw_tracks[column].apply(_parse_alongside_ids)

    for column in PADDED_INT_COLUMNS:
        if column in raw_tracks.columns:
            raw_tracks[column] = raw_tracks[column].apply(lambda value: _parse_padded_numeric_list(value, int))

    for column in PADDED_FLOAT_COLUMNS:
        if column in raw_tracks.columns:
            raw_tracks[column] = raw_tracks[column].apply(lambda value: _parse_padded_numeric_list(value, float))

    return raw_tracks


def _finalize_track_dict(track: dict, track_rows: pandas.DataFrame) -> dict:
    for key, value in track.items():
        if key in {"trackId", "recordingId"}:
            track[key] = value[0]
        elif key in ALONGSIDE_ID_COLUMNS:
            continue
        else:
            track[key] = np.array(value)

    if "traveledDistance" not in track or np.allclose(track["traveledDistance"], 0.0):
        center = np.column_stack([track["xCenter"], track["yCenter"]])
        if len(center) > 1:
            increments = np.linalg.norm(np.diff(center, axis=0), axis=1)
            track["traveledDistance"] = np.concatenate(([0.0], np.cumsum(increments)))
        else:
            track["traveledDistance"] = np.zeros(len(center), dtype=float)

    if "laneChange" in track and np.any(track["laneChange"]):
        return track

    lanelet = track.get("laneletId")
    if lanelet is None:
        return track

    primary_lane = np.full(lanelet.shape[0], np.nan)
    for idx, row in enumerate(lanelet):
        row = np.asarray(row, dtype=float)
        valid_entries = row[~np.isnan(row)]
        if valid_entries.size:
            primary_lane[idx] = valid_entries[0]

    lane_change = np.zeros(primary_lane.shape[0], dtype=int)
    for idx in range(1, primary_lane.shape[0]):
        prev_lane = primary_lane[idx - 1]
        curr_lane = primary_lane[idx]
        if not np.isnan(prev_lane) and not np.isnan(curr_lane) and prev_lane != curr_lane:
            lane_change[idx] = 1
    track["laneChange"] = lane_change
    return track


def _parse_alongside_ids(value):
    if isinstance(value, list):
        return [int(v) for v in value if not pandas.isna(v) and int(v) > 0]
    if pandas.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    ids = []
    for entry in text.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        numeric = int(float(entry))
        if numeric > 0:
            ids.append(numeric)
    return ids


def _parse_padded_numeric_list(value, caster):
    if isinstance(value, list):
        values = value
    elif pandas.isna(value):
        values = []
    else:
        text = str(value).strip()
        values = [] if not text else text.split(";")

    output = _empty_padded_list()
    for idx, entry in enumerate(values[:N_MAX_OVERLAPPING_LANELETS]):
        if entry == "" or pandas.isna(entry):
            continue
        output[idx] = caster(float(entry)) if caster is int else caster(entry)
    return output


def _normalize_id_series(series: pandas.Series) -> pandas.Series:
    values = pandas.to_numeric(series, errors="coerce").fillna(-1).astype(int)
    values[values <= 0] = -1
    return values


def _empty_padded_list():
    return [np.nan] * N_MAX_OVERLAPPING_LANELETS


def _pack_numeric_value(value, caster):
    if pandas.isna(value):
        return _empty_padded_list()
    output = _empty_padded_list()
    output[0] = caster(value)
    return output


def _parse_markings(value) -> List[float]:
    if pandas.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [float(entry) for entry in text.split(";") if entry.strip()]


def _build_highd_lane_info(recording_meta: dict) -> dict:
    upper_key = "upperLaneMarkings" if "upperLaneMarkings" in recording_meta else "upperLaneMark"
    lower_key = "lowerLaneMarkings" if "lowerLaneMarkings" in recording_meta else "lowerLaneMark"
    upper_markings = _parse_markings(recording_meta.get(upper_key))
    lower_markings = _parse_markings(recording_meta.get(lower_key))
    if len(upper_markings) < 2 and len(lower_markings) < 2:
        return {}

    lane_info = {}
    lane_id = 2
    for boundaries in (upper_markings, lower_markings):
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            lane_info[lane_id] = {"center": (start + end) / 2.0, "width": abs(end - start)}
            lane_id += 1
        lane_id += 1
    return lane_info


def _normalize_dataset_key(dataset_name: str) -> str:
    return str(dataset_name).strip().lower()


def _read_special_dataset(dataset_dir: str, dataset_key: str, include_px_coordinates: bool) -> Tuple[List[dict], List[dict], dict]:
    cache_path = os.path.join(dataset_dir, f".track_visualizer_cache_{dataset_key}.pkl")
    source_paths = _special_dataset_source_paths(dataset_dir, dataset_key)
    if _is_cache_fresh(cache_path, source_paths):
        with open(cache_path, "rb") as fh:
            cached = pickle.load(fh)
        if cached.get("cache_version") == SPECIAL_DATASET_CACHE_VERSION:
            return cached["tracks"], cached["tracks_meta"], cached["recording_meta"]
        logger.info("Rebuilding {} cache because the cache schema version changed.", dataset_key)

    if dataset_key == "sqm-n-4":
        tracks, tracks_meta, recording_meta = _build_sqm_dataset(dataset_dir, include_px_coordinates)
    elif dataset_key == "ytdj-3":
        tracks, tracks_meta, recording_meta = _build_ytdj_dataset(dataset_dir, include_px_coordinates)
    elif dataset_key in {"xam-n-5", "xam-n-6"}:
        tracks, tracks_meta, recording_meta = _build_xam_dataset(dataset_dir, dataset_key, include_px_coordinates)
    else:
        raise ValueError(f"Unsupported special dataset: {dataset_key}")

    with open(cache_path, "wb") as fh:
        pickle.dump(
            {
                "cache_version": SPECIAL_DATASET_CACHE_VERSION,
                "tracks": tracks,
                "tracks_meta": tracks_meta,
                "recording_meta": recording_meta,
            },
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return tracks, tracks_meta, recording_meta


def _special_dataset_source_paths(dataset_dir: str, dataset_key: str) -> List[str]:
    if dataset_key == "sqm-n-4":
        return [
            os.path.join(dataset_dir, "frenet-up.xlsx"),
            os.path.join(dataset_dir, "frenet-down.xlsx"),
            os.path.join(dataset_dir, SPECIAL_DATASET_LAYOUTS[dataset_key]["background_name"]),
        ]
    if dataset_key == "ytdj-3":
        return [
            os.path.join(dataset_dir, "frenet.csv"),
            os.path.join(dataset_dir, SPECIAL_DATASET_LAYOUTS[dataset_key]["background_name"]),
        ]
    if dataset_key in {"xam-n-5", "xam-n-6"}:
        return [
            os.path.join(dataset_dir, "frenet.csv"),
            os.path.join(dataset_dir, "pixel.csv"),
            os.path.join(dataset_dir, SPECIAL_DATASET_LAYOUTS[dataset_key]["background_name"]),
        ]
    return []


def _is_cache_fresh(cache_path: str, source_paths: List[str]) -> bool:
    if not os.path.exists(cache_path):
        return False
    cache_mtime = os.path.getmtime(cache_path)
    for source_path in source_paths:
        if os.path.exists(source_path) and os.path.getmtime(source_path) > cache_mtime:
            return False
    return True


def _build_sqm_dataset(dataset_dir: str, include_px_coordinates: bool) -> Tuple[List[dict], List[dict], dict]:
    layout = SPECIAL_DATASET_LAYOUTS["sqm-n-4"]
    frames = []
    workbook_specs = [
        ("frenet-up.xlsx", "up"),
        ("frenet-down.xlsx", "down"),
    ]
    usecols = [
        "vehicleID",
        "laneID",
        "time(s)",
        "longitudinalDistance(m)",
        "distanceToUpperLaneLine(m)",
        "distanceToLowerLaneLine(m)",
        "speed(km/h)",
        "acceleration(m/s^2)",
        "vehicleLength(m)",
        "vehicleWidth(m)",
    ]

    for workbook_name, direction in workbook_specs:
        workbook_path = os.path.join(dataset_dir, workbook_name)
        xls = pandas.ExcelFile(workbook_path)
        logger.info("Loading {} sheets from {}", direction, workbook_path)
        for sheet_name in xls.sheet_names:
            df = pandas.read_excel(workbook_path, sheet_name=sheet_name, usecols=usecols)
            df = df.rename(
                columns={
                    "vehicleID": "vehicle_id",
                    "laneID": "lane_id",
                    "time(s)": "time_s",
                    "longitudinalDistance(m)": "longitudinal_m",
                    "distanceToUpperLaneLine(m)": "dist_upper_m",
                    "distanceToLowerLaneLine(m)": "dist_lower_m",
                    "speed(km/h)": "speed_kmh",
                    "acceleration(m/s^2)": "accel_mps2",
                    "vehicleLength(m)": "length_m",
                    "vehicleWidth(m)": "width_m",
                }
            )
            df["direction_group"] = direction
            df["recordingId"] = 1
            df["trackId"] = df["vehicle_id"].astype(int) + int(layout["direction_offsets"][direction])
            df["lane_width_m"] = df["dist_upper_m"] + df["dist_lower_m"]
            df["lat_offset_m"] = df["dist_upper_m"] - df["lane_width_m"] / 2.0
            frames.append(df)

    raw_rows = pandas.concat(frames, ignore_index=True)
    raw_rows["frame"] = np.rint(raw_rows["time_s"] * layout["frame_rate"]).astype(int)
    return _build_special_tracks_from_rows(raw_rows, dataset_dir, "sqm-n-4", layout, include_px_coordinates)


def _build_ytdj_dataset(dataset_dir: str, include_px_coordinates: bool) -> Tuple[List[dict], List[dict], dict]:
    layout = SPECIAL_DATASET_LAYOUTS["ytdj-3"]
    df = pandas.read_csv(
        os.path.join(dataset_dir, "frenet.csv"),
        low_memory=False,
        usecols=[
            "vehicleID",
            "laneID",
            "time(s)",
            "lateralDistance(m)",
            "longitudinalDistance(m)",
            "speed(km/h)",
            "acceleration(m/s^2)",
            "vehicleLength(m)",
            "vehicleWidth(m)",
        ],
    )
    raw_rows = df.rename(
        columns={
            "vehicleID": "vehicle_id",
            "laneID": "lane_id",
            "time(s)": "time_s",
            "lateralDistance(m)": "lat_offset_m",
            "longitudinalDistance(m)": "longitudinal_m",
            "speed(km/h)": "speed_kmh",
            "acceleration(m/s^2)": "accel_mps2",
            "vehicleLength(m)": "length_m",
            "vehicleWidth(m)": "width_m",
        }
    )
    raw_rows["recordingId"] = 1
    raw_rows["trackId"] = raw_rows["vehicle_id"].astype(int)
    raw_rows["frame"] = np.rint(raw_rows["time_s"] * layout["frame_rate"]).astype(int)
    raw_rows["lane_width_m"] = 3.75
    raw_rows["direction_group"] = np.where(raw_rows["lane_id"] <= 5, "upper", "lower")
    return _build_special_tracks_from_rows(raw_rows, dataset_dir, "ytdj-3", layout, include_px_coordinates)


def _build_xam_dataset(dataset_dir: str, dataset_key: str, include_px_coordinates: bool) -> Tuple[List[dict], List[dict], dict]:
    layout = SPECIAL_DATASET_LAYOUTS[dataset_key]
    image_width, image_height = Image.open(os.path.join(dataset_dir, layout["background_name"])).size
    aerial_height_px = _detect_special_aerial_panel_height(os.path.join(dataset_dir, layout["background_name"]))

    pixel_rows = pandas.read_csv(
        os.path.join(dataset_dir, "pixel.csv"),
        low_memory=False,
        usecols=lambda c: c in {"frame", "time(s)", "vehicleID", "x", "y", "w", "h", "class"},
    ).rename(
        columns={
            "vehicleID": "trackId",
            "time(s)": "time_s",
            "x": "xCenterVis",
            "y": "yCenterVis",
            "w": "lengthVis",
            "h": "widthVis",
            "class": "pixel_class",
        }
    )

    if "time_s" not in pixel_rows.columns:
        pixel_rows["time_s"] = pandas.to_numeric(pixel_rows["frame"], errors="coerce") / float(layout["frame_rate"])
    pixel_rows["recordingId"] = 1
    pixel_rows["frame"] = pandas.to_numeric(pixel_rows["frame"], errors="coerce").astype(int)
    pixel_rows["trackId"] = pandas.to_numeric(pixel_rows["trackId"], errors="coerce").astype(int)
    x_vis, y_vis, length_vis, width_vis = _transform_xam_pixel_coordinates(pixel_rows, image_width, aerial_height_px)
    pixel_rows["xCenterVis"] = x_vis
    pixel_rows["yCenterVis"] = y_vis
    pixel_rows["lengthVis"] = length_vis
    pixel_rows["widthVis"] = width_vis
    pixel_rows["xCenterVis"] = pixel_rows.groupby("trackId")["xCenterVis"].transform(_smooth_track_series)
    pixel_rows["yCenterVis"] = pixel_rows.groupby("trackId")["yCenterVis"].transform(_smooth_track_series)
    pixel_rows = pixel_rows.sort_values(["trackId", "frame", "time_s"]).reset_index(drop=True)
    pixel_rows["sample_idx"] = pixel_rows.groupby("trackId").cumcount()

    frenet_rows = pandas.read_csv(
        os.path.join(dataset_dir, "frenet.csv"),
        low_memory=False,
        usecols=lambda c: c in {
            "frame",
            "time(s)",
            "vehicleID",
            "laneID",
            "lateralDistance(m)",
            "longitudinalDistance(m)",
            "speed(km/h)",
            "acceleration(m/s^2)",
            "vehicleLength(m)",
            "vehicleWidth(m)",
        },
    ).rename(
        columns={
            "vehicleID": "trackId",
            "laneID": "lane_id",
            "time(s)": "time_s",
            "lateralDistance(m)": "lat_offset_m",
            "longitudinalDistance(m)": "longitudinal_m",
            "speed(km/h)": "speed_kmh",
            "acceleration(m/s^2)": "accel_mps2",
            "vehicleLength(m)": "length_m",
            "vehicleWidth(m)": "width_m",
        }
    )
    if "frame" not in frenet_rows.columns:
        frenet_rows["frame"] = np.rint(
            pandas.to_numeric(frenet_rows["time_s"], errors="coerce") * float(layout["frame_rate"])
        ).astype(int)
    frenet_rows["trackId"] = pandas.to_numeric(frenet_rows["trackId"], errors="coerce").astype(int)
    frenet_rows["frame"] = pandas.to_numeric(frenet_rows["frame"], errors="coerce").astype(int)
    frenet_rows = frenet_rows.sort_values(["trackId", "frame", "time_s"]).reset_index(drop=True)

    merged_rows = _merge_xam_pixel_and_frenet(pixel_rows, frenet_rows)
    ortho_px_to_meter_x, ortho_px_to_meter_y = _estimate_xam_meter_per_px(merged_rows, image_width, layout)
    merged_rows["xCenterVis"] = pandas.to_numeric(merged_rows["xCenterVis"], errors="coerce")
    merged_rows["yCenterVis"] = pandas.to_numeric(merged_rows["yCenterVis"], errors="coerce")
    merged_rows["lengthVis"] = pandas.to_numeric(merged_rows["lengthVis"], errors="coerce")
    merged_rows["widthVis"] = pandas.to_numeric(merged_rows["widthVis"], errors="coerce")
    merged_rows["xCenter"] = merged_rows["xCenterVis"] * ortho_px_to_meter_x
    merged_rows["yCenter"] = -merged_rows["yCenterVis"] * ortho_px_to_meter_y
    merged_rows["length_m"] = pandas.to_numeric(merged_rows.get("length_m"), errors="coerce")
    merged_rows["width_m"] = pandas.to_numeric(merged_rows.get("width_m"), errors="coerce")
    merged_rows["length_m"] = merged_rows["length_m"].fillna(merged_rows["lengthVis"] * ortho_px_to_meter_x)
    merged_rows["width_m"] = merged_rows["width_m"].fillna(merged_rows["widthVis"] * ortho_px_to_meter_y)
    merged_rows["lane_id"] = pandas.to_numeric(merged_rows.get("lane_id"), errors="coerce").fillna(-1).astype(int)
    merged_rows["lat_offset_m"] = pandas.to_numeric(merged_rows.get("lat_offset_m"), errors="coerce").fillna(0.0)
    merged_rows["speed_kmh"] = pandas.to_numeric(merged_rows.get("speed_kmh"), errors="coerce")
    merged_rows["accel_mps2"] = pandas.to_numeric(merged_rows.get("accel_mps2"), errors="coerce")

    tracks = []
    tracks_meta = []
    for track_id, track_rows in merged_rows.groupby("trackId", sort=True):
        track_rows = track_rows.sort_values(["frame", "time_s"]).drop_duplicates("frame", keep="last")
        track_rows = track_rows.reset_index(drop=True)

        frame = track_rows["frame"].to_numpy(dtype=int)
        x_center_vis = track_rows["xCenterVis"].to_numpy(dtype=float)
        y_center_vis = track_rows["yCenterVis"].to_numpy(dtype=float)
        x_center = track_rows["xCenter"].to_numpy(dtype=float)
        y_center = track_rows["yCenter"].to_numpy(dtype=float)
        length_vis = track_rows["lengthVis"].to_numpy(dtype=float)
        width_vis = track_rows["widthVis"].to_numpy(dtype=float)
        length = track_rows["length_m"].to_numpy(dtype=float)
        width = track_rows["width_m"].to_numpy(dtype=float)

        dt = 1.0 / float(layout["frame_rate"])
        x_velocity = np.gradient(x_center, dt)
        y_velocity = np.gradient(y_center, dt)
        x_acceleration = np.gradient(x_velocity, dt)
        y_acceleration = np.gradient(y_velocity, dt)
        heading = _stabilize_xam_heading(x_velocity, y_velocity)
        heading_vis = (-heading) % 360.0
        speed = np.hypot(x_velocity, y_velocity)
        safe_speed = np.where(speed > 1e-9, speed, 1.0)
        lon_acceleration = (x_velocity * x_acceleration + y_velocity * y_acceleration) / safe_speed
        lat_acceleration = (-y_velocity * x_acceleration + x_velocity * y_acceleration) / safe_speed
        traveled_distance = (
            np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(np.column_stack([x_center, y_center]), axis=0), axis=1))))
            if len(x_center) > 1
            else np.zeros(1)
        )

        lane_ids = track_rows["lane_id"].to_numpy(dtype=int)
        lane_change = np.zeros(len(lane_ids), dtype=int)
        if len(lane_ids) > 1:
            valid_prev = lane_ids[:-1] > 0
            valid_curr = lane_ids[1:] > 0
            lane_change[1:] = ((lane_ids[1:] != lane_ids[:-1]) & valid_prev & valid_curr).astype(int)

        track = {
            "recordingId": 1,
            "trackId": int(track_id),
            "frame": frame,
            "trackLifetime": np.arange(len(frame), dtype=int),
            "xCenter": x_center,
            "yCenter": y_center,
            "heading": heading,
            "width": width,
            "length": length,
            "xVelocity": x_velocity,
            "yVelocity": y_velocity,
            "xAcceleration": x_acceleration,
            "yAcceleration": y_acceleration,
            "lonVelocity": speed,
            "latVelocity": np.zeros(len(frame), dtype=float),
            "lonAcceleration": lon_acceleration,
            "latAcceleration": lat_acceleration,
            "traveledDistance": traveled_distance,
            "latLaneCenterOffset": np.array(
                [_pack_numeric_value(v, float) if lane_id > 0 else _empty_padded_list() for v, lane_id in zip(track_rows["lat_offset_m"], lane_ids)],
                dtype=float,
            ),
            "laneWidth": np.array(
                [_pack_numeric_value(3.5, float) if lane_id > 0 else _empty_padded_list() for lane_id in lane_ids],
                dtype=float,
            ),
            "laneletId": np.array(
                [_pack_numeric_value(v, int) if v > 0 else _empty_padded_list() for v in lane_ids],
                dtype=float,
            ),
            "laneChange": lane_change,
            "center": np.stack([x_center, y_center], axis=-1),
            "bbox": get_rotated_bbox(x_center, y_center, length, width, np.deg2rad(heading)),
            "xCenterVis": x_center_vis,
            "yCenterVis": y_center_vis,
            "centerVis": np.stack([x_center_vis, y_center_vis], axis=-1),
            "lengthVis": length_vis,
            "widthVis": width_vis,
            "headingVis": heading_vis,
            "bboxVis": get_rotated_bbox(x_center_vis, y_center_vis, length_vis, width_vis, np.deg2rad(heading_vis)),
        }
        tracks.append(track)

        object_class = _infer_vehicle_class(float(np.nanmean(length)))
        tracks_meta.append(
            {
                "recordingId": 1,
                "trackId": int(track_id),
                "initialFrame": int(frame[0]),
                "finalFrame": int(frame[-1]),
                "numFrames": int(len(frame)),
                "width": float(np.nanmean(width)),
                "length": float(np.nanmean(length)),
                "class": object_class,
            }
        )

    recording_meta = {
        "recordingId": 1,
        "locationId": int(layout["location_id"]),
        "frameRate": float(layout["frame_rate"]),
        "speedLimit": float(layout["speed_limit_mps"]),
        "duration": float((max(meta["finalFrame"] for meta in tracks_meta) - min(meta["initialFrame"] for meta in tracks_meta) + 1) / layout["frame_rate"]),
        "numTracks": int(len(tracks)),
        "numVehicles": int(len(tracks)),
        "orthoPxToMeter": float(0.5 * (ortho_px_to_meter_x + ortho_px_to_meter_y)),
    }
    return tracks, sorted(tracks_meta, key=lambda entry: entry["trackId"]), recording_meta


def _merge_xam_pixel_and_frenet(pixel_rows: pandas.DataFrame, frenet_rows: pandas.DataFrame) -> pandas.DataFrame:
    frenet_value_columns = [
        "trackId",
        "frame",
        "time_s",
        "lane_id",
        "lat_offset_m",
        "longitudinal_m",
        "speed_kmh",
        "accel_mps2",
        "length_m",
        "width_m",
    ]

    if {"trackId", "frame"}.issubset(frenet_rows.columns):
        merged_on_frame = pixel_rows.merge(
            frenet_rows[frenet_value_columns],
            on=["trackId", "frame"],
            how="left",
            suffixes=("", "_frenet"),
        )
        frame_match_ratio = float(merged_on_frame["length_m"].notna().mean()) if len(merged_on_frame) else 0.0
        if frame_match_ratio >= 0.9:
            merged_on_frame["time_s"] = merged_on_frame["time_s"].fillna(merged_on_frame.get("time_s_frenet"))
            return merged_on_frame.drop(columns=[col for col in ["time_s_frenet"] if col in merged_on_frame.columns])

    pixel_rows = pixel_rows.copy()
    frenet_rows = frenet_rows.copy()
    pixel_rows["sample_idx"] = pixel_rows.groupby("trackId").cumcount()
    frenet_rows["sample_idx"] = frenet_rows.groupby("trackId").cumcount()
    merged = pixel_rows.merge(
        frenet_rows[[col for col in frenet_value_columns if col != "frame"] + ["sample_idx"]],
        on=["trackId", "sample_idx"],
        how="left",
        suffixes=("", "_frenet"),
    )
    merged["time_s"] = merged["time_s"].fillna(merged.get("time_s_frenet"))
    return merged.drop(columns=[col for col in ["time_s_frenet", "sample_idx"] if col in merged.columns])


def _estimate_xam_meter_per_px(raw_rows: pandas.DataFrame, image_width: int, layout: dict) -> tuple[float, float]:
    x_candidates = []
    if {"length_m", "lengthVis"}.issubset(raw_rows.columns):
        length_ratio = pandas.to_numeric(raw_rows["length_m"], errors="coerce") / pandas.to_numeric(raw_rows["lengthVis"], errors="coerce")
        x_candidates.append(length_ratio.to_numpy(dtype=float))
    y_candidates = []
    if {"width_m", "widthVis"}.issubset(raw_rows.columns):
        width_ratio = pandas.to_numeric(raw_rows["width_m"], errors="coerce") / pandas.to_numeric(raw_rows["widthVis"], errors="coerce")
        y_candidates.append(width_ratio.to_numpy(dtype=float))

    fallback = float(layout["section_length_m"]) / float(max(image_width - 40, 1))
    x_ratio = _median_positive_ratio(x_candidates, fallback)
    y_ratio = _median_positive_ratio(y_candidates, x_ratio)
    return x_ratio, y_ratio


def _median_positive_ratio(candidate_arrays: List[np.ndarray], fallback: float) -> float:
    if candidate_arrays:
        merged = np.concatenate(candidate_arrays)
        merged = merged[np.isfinite(merged) & (merged > 1e-4)]
        if merged.size:
            return float(np.median(merged))
    return float(fallback)


def _detect_special_aerial_panel_height(background_path: str) -> int:
    image = np.array(Image.open(background_path).convert("RGB"))
    row_mean = image.mean(axis=(1, 2))
    white_rows = np.where(row_mean > 245.0)[0]
    if white_rows.size:
        return int(max(white_rows[0], 1))
    return int(image.shape[0] // 2)


def _transform_xam_pixel_coordinates(
    pixel_rows: pandas.DataFrame,
    image_width: int,
    aerial_height_px: int,
) -> tuple[pandas.Series, pandas.Series, pandas.Series, pandas.Series]:
    x_source = pandas.to_numeric(pixel_rows["xCenterVis"], errors="coerce")
    y_source = pandas.to_numeric(pixel_rows["yCenterVis"], errors="coerce")
    length_source = pandas.to_numeric(pixel_rows["lengthVis"], errors="coerce")
    width_source = pandas.to_numeric(pixel_rows["widthVis"], errors="coerce")

    x_low, x_high = x_source.quantile([0.001, 0.999])
    y_low, y_high = y_source.quantile([0.001, 0.999])

    target_left = 20.0
    target_right = float(image_width - 20.0)
    target_top = 20.0
    target_bottom = float(max(aerial_height_px - 20.0, target_top + 1.0))

    x_scale = (target_right - target_left) / max(float(x_high - x_low), 1e-6)
    y_scale = (target_bottom - target_top) / max(float(y_high - y_low), 1e-6)

    x_vis = (x_source - float(x_low)) * x_scale + target_left
    y_vis = (y_source - float(y_low)) * y_scale + target_top
    length_vis = length_source * x_scale
    width_vis = width_source * y_scale
    return x_vis, y_vis, length_vis, width_vis


def _smooth_track_series(series: pandas.Series, window: int = 5) -> pandas.Series:
    return series.rolling(window=window, center=True, min_periods=1).mean()


def _stabilize_xam_heading(x_velocity: np.ndarray, y_velocity: np.ndarray) -> np.ndarray:
    forward_mask = x_velocity >= 0.0
    base_heading = np.where(forward_mask, 0.0, 180.0)
    correction = np.rad2deg(np.arctan2(y_velocity, np.maximum(np.abs(x_velocity), 1e-6)))
    correction = np.clip(correction, -12.0, 12.0)
    return (base_heading + correction) % 360.0


def _build_special_tracks_from_rows(
    raw_rows: pandas.DataFrame,
    dataset_dir: str,
    dataset_key: str,
    layout: dict,
    include_px_coordinates: bool,
) -> Tuple[List[dict], List[dict], dict]:
    background_path = os.path.join(dataset_dir, layout["background_name"])
    image_width, image_height = Image.open(background_path).size
    x_margin_left = float(layout["x_margin_frac"][0]) * image_width
    x_margin_right = float(layout["x_margin_frac"][1]) * image_width
    usable_width_px = max(1.0, x_margin_right - x_margin_left)
    x_px_per_meter = usable_width_px / float(layout["section_length_m"])
    lateral_px_per_meter = float(layout.get("lateral_px_per_m", x_px_per_meter))

    raw_rows = raw_rows.copy()
    raw_rows["xCenterVis"] = _compute_special_x_vis(raw_rows, dataset_key, layout, x_margin_left, x_margin_right, x_px_per_meter)
    raw_rows["yCenterVis"] = _compute_special_y_vis(raw_rows, dataset_key, layout, image_height, lateral_px_per_meter)
    raw_rows["xCenter"] = raw_rows["xCenterVis"] / x_px_per_meter
    raw_rows["yCenter"] = -raw_rows["yCenterVis"] / lateral_px_per_meter

    tracks = []
    tracks_meta = []
    for track_id, track_rows in raw_rows.groupby("trackId", sort=True):
        track_rows = _reindex_special_track(track_rows.sort_values(["frame", "time_s"]), layout["frame_rate"])

        x_center_vis = track_rows["xCenterVis"].to_numpy(dtype=float)
        y_center_vis = track_rows["yCenterVis"].to_numpy(dtype=float)
        x_center = track_rows["xCenter"].to_numpy(dtype=float)
        y_center = track_rows["yCenter"].to_numpy(dtype=float)
        length = track_rows["length_m"].to_numpy(dtype=float)
        width = track_rows["width_m"].to_numpy(dtype=float)
        speed = track_rows["speed_kmh"].to_numpy(dtype=float) / 3.6
        speed = np.nan_to_num(speed, nan=0.0, posinf=0.0, neginf=0.0)

        dt = 1.0 / float(layout["frame_rate"])
        x_velocity = np.gradient(x_center, dt)
        y_velocity = np.gradient(y_center, dt)
        x_acceleration = np.gradient(x_velocity, dt)
        y_acceleration = np.gradient(y_velocity, dt)
        heading = (np.rad2deg(np.arctan2(y_velocity, x_velocity)) + 360.0) % 360.0
        heading_vis = (-heading) % 360.0
        safe_speed = np.where(np.hypot(x_velocity, y_velocity) > 1e-9, np.hypot(x_velocity, y_velocity), 1.0)
        lon_acceleration = (x_velocity * x_acceleration + y_velocity * y_acceleration) / safe_speed
        lat_acceleration = (-y_velocity * x_acceleration + x_velocity * y_acceleration) / safe_speed
        traveled_distance = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(np.column_stack([x_center, y_center]), axis=0), axis=1)))) if len(x_center) > 1 else np.zeros(1)
        lane_ids = track_rows["lane_id"].to_numpy(dtype=int)
        lane_width = track_rows["lane_width_m"].to_numpy(dtype=float)
        lat_offset = track_rows["lat_offset_m"].to_numpy(dtype=float)
        lane_change = np.zeros(len(lane_ids), dtype=int)
        if len(lane_ids) > 1:
            lane_change[1:] = (lane_ids[1:] != lane_ids[:-1]).astype(int)

        track = {
            "recordingId": int(track_rows["recordingId"].iloc[0]),
            "trackId": int(track_id),
            "frame": track_rows["frame"].to_numpy(dtype=int),
            "trackLifetime": np.arange(len(track_rows), dtype=int),
            "xCenter": x_center,
            "yCenter": y_center,
            "heading": heading,
            "width": width,
            "length": length,
            "xVelocity": x_velocity,
            "yVelocity": y_velocity,
            "xAcceleration": x_acceleration,
            "yAcceleration": y_acceleration,
            "lonVelocity": speed,
            "latVelocity": np.zeros(len(track_rows), dtype=float),
            "lonAcceleration": lon_acceleration,
            "latAcceleration": lat_acceleration,
            "traveledDistance": traveled_distance,
            "latLaneCenterOffset": np.array([_pack_numeric_value(v, float) for v in lat_offset], dtype=float),
            "laneWidth": np.array([_pack_numeric_value(v, float) for v in lane_width], dtype=float),
            "laneletId": np.array([_pack_numeric_value(v, int) for v in lane_ids], dtype=float),
            "laneChange": lane_change,
            "center": np.stack([x_center, y_center], axis=-1),
            "bbox": get_rotated_bbox(x_center, y_center, length, width, np.deg2rad(heading)),
        }

        length_vis = length * x_px_per_meter
        width_vis = width * lateral_px_per_meter
        track["xCenterVis"] = x_center_vis
        track["yCenterVis"] = y_center_vis
        track["centerVis"] = np.stack([x_center_vis, y_center_vis], axis=-1)
        track["lengthVis"] = length_vis
        track["widthVis"] = width_vis
        track["headingVis"] = heading_vis
        track["bboxVis"] = get_rotated_bbox(x_center_vis, y_center_vis, length_vis, width_vis, np.deg2rad(heading_vis))
        tracks.append(track)

        object_class = _infer_vehicle_class(float(np.nanmean(length)))
        tracks_meta.append(
            {
                "recordingId": int(track["recordingId"]),
                "trackId": int(track_id),
                "initialFrame": int(track["frame"][0]),
                "finalFrame": int(track["frame"][-1]),
                "numFrames": int(len(track["frame"])),
                "width": float(np.nanmean(width)),
                "length": float(np.nanmean(length)),
                "class": object_class,
            }
        )

    recording_meta = {
        "recordingId": 1,
        "locationId": int(layout["location_id"]),
        "frameRate": float(layout["frame_rate"]),
        "speedLimit": float(layout["speed_limit_mps"]),
        "duration": float((max(meta["finalFrame"] for meta in tracks_meta) - min(meta["initialFrame"] for meta in tracks_meta) + 1) / layout["frame_rate"]),
        "numTracks": int(len(tracks)),
        "numVehicles": int(len(tracks)),
        "orthoPxToMeter": float(1.0 / x_px_per_meter),
    }
    return tracks, sorted(tracks_meta, key=lambda entry: entry["trackId"]), recording_meta


def _compute_special_x_vis(
    raw_rows: pandas.DataFrame,
    dataset_key: str,
    layout: dict,
    x_margin_left: float,
    x_margin_right: float,
    x_px_per_meter: float,
) -> np.ndarray:
    longitudinal_m = pandas.to_numeric(raw_rows["longitudinal_m"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if dataset_key == "sqm-n-4":
        reverse = np.array(
            [bool(layout["reverse_x_for_direction"].get(direction, False)) for direction in raw_rows["direction_group"]],
            dtype=bool,
        )
    else:
        reverse = np.array(
            [bool(layout["reverse_x_for_lane"].get(int(lane_id), False)) for lane_id in raw_rows["lane_id"]],
            dtype=bool,
        )

    x_vis = x_margin_left + longitudinal_m * x_px_per_meter
    x_vis = np.where(reverse, x_margin_right - longitudinal_m * x_px_per_meter, x_vis)
    return np.clip(x_vis, x_margin_left, x_margin_right)


def _compute_special_y_vis(
    raw_rows: pandas.DataFrame,
    dataset_key: str,
    layout: dict,
    image_height: int,
    lateral_px_per_meter: float,
) -> np.ndarray:
    longitudinal_m = pandas.to_numeric(raw_rows["longitudinal_m"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    centers = _compute_special_lane_centers(raw_rows, longitudinal_m, dataset_key, layout, image_height)
    lat_offset = pandas.to_numeric(raw_rows["lat_offset_m"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return centers + lat_offset * lateral_px_per_meter


def _compute_special_lane_centers(
    raw_rows: pandas.DataFrame,
    longitudinal_m: np.ndarray,
    dataset_key: str,
    layout: dict,
    image_height: int,
) -> np.ndarray:
    centers = np.full(len(raw_rows), np.nan, dtype=float)
    profiles = layout.get("lane_center_profiles_px", {})
    fallback_map = SPECIAL_DATASET_FALLBACK_CENTER_Y_PX.get(dataset_key, {})

    if dataset_key == "sqm-n-4":
        direction_groups = raw_rows["direction_group"].astype(str).to_numpy()
        lane_ids = pandas.to_numeric(raw_rows["lane_id"], errors="coerce").fillna(-1).astype(int).to_numpy()
        for direction_group, lane_profiles in profiles.items():
            direction_mask = direction_groups == str(direction_group)
            if not np.any(direction_mask):
                continue
            for lane_id, profile in lane_profiles.items():
                mask = direction_mask & (lane_ids == int(lane_id))
                if np.any(mask):
                    centers[mask] = _interpolate_profile(profile, longitudinal_m[mask])
        for direction_group, fallback_center in fallback_map.items():
            mask = np.isnan(centers) & (direction_groups == str(direction_group))
            if np.any(mask):
                centers[mask] = float(fallback_center)
    else:
        lane_ids = pandas.to_numeric(raw_rows["lane_id"], errors="coerce").fillna(-1).astype(int).to_numpy()
        for lane_id, profile in profiles.items():
            mask = lane_ids == int(lane_id)
            if np.any(mask):
                centers[mask] = _interpolate_profile(profile, longitudinal_m[mask])
        direction_groups = raw_rows["direction_group"].astype(str).to_numpy()
        for direction_group, fallback_center in fallback_map.items():
            mask = np.isnan(centers) & (direction_groups == str(direction_group))
            if np.any(mask):
                centers[mask] = float(fallback_center)

    unresolved = np.isnan(centers)
    if np.any(unresolved):
        centers[unresolved] = np.nanmean(list(fallback_map.values())) if fallback_map else 0.7 * image_height
    return centers


def _interpolate_profile(profile: List[tuple[float, float]], values: np.ndarray) -> np.ndarray:
    if not profile:
        return np.zeros_like(values, dtype=float)
    profile_array = np.asarray(profile, dtype=float)
    x_points = profile_array[:, 0]
    y_points = profile_array[:, 1]
    if len(profile_array) == 1:
        return np.full(values.shape, y_points[0], dtype=float)
    return np.interp(values, x_points, y_points, left=y_points[0], right=y_points[-1])


def _reindex_special_track(track_rows: pandas.DataFrame, frame_rate: float) -> pandas.DataFrame:
    track_rows = track_rows.drop_duplicates("frame", keep="last").sort_values("frame").set_index("frame")
    full_index = np.arange(int(track_rows.index.min()), int(track_rows.index.max()) + 1, dtype=int)
    track_rows = track_rows.reindex(full_index)

    constant_cols = ["trackId", "recordingId"]
    categorical_cols = ["vehicle_id", "lane_id", "direction_group"]
    numeric_cols = [
        "time_s",
        "longitudinal_m",
        "lat_offset_m",
        "lane_width_m",
        "speed_kmh",
        "accel_mps2",
        "length_m",
        "width_m",
        "xCenterVis",
        "yCenterVis",
        "xCenter",
        "yCenter",
    ]

    for column in constant_cols:
        track_rows[column] = track_rows[column].ffill().bfill()
    for column in categorical_cols:
        if column in track_rows.columns:
            track_rows[column] = track_rows[column].ffill().bfill()
    for column in numeric_cols:
        if column in track_rows.columns:
            track_rows[column] = pandas.to_numeric(track_rows[column], errors="coerce").interpolate(limit_direction="both").ffill().bfill()

    track_rows["frame"] = full_index
    track_rows["trackLifetime"] = np.arange(len(track_rows), dtype=int)
    if "time_s" not in track_rows.columns or track_rows["time_s"].isna().all():
        track_rows["time_s"] = full_index / float(frame_rate)
    return track_rows.reset_index(drop=True)


def _infer_vehicle_class(length_m: float) -> str:
    if not np.isfinite(length_m):
        return "car"
    if length_m >= 8.0:
        return "truck_bus"
    if length_m <= 2.5:
        return "motorcycle"
    return "car"


def get_rotated_bbox(x_center: np.ndarray, y_center: np.ndarray,
                     length: np.ndarray, width: np.ndarray, heading: np.ndarray) -> np.ndarray:
    """
    Calculate the corners of a rotated bbox from the position, shape and heading for every timestamp.

    :param x_center: x coordinates of the object center positions [num_timesteps]
    :param y_center: y coordinates of the object center positions [num_timesteps]
    :param length: objects lengths [num_timesteps]
    :param width: object widths [num_timesteps]
    :param heading: object heading (rad) [num_timesteps]
    :return: Numpy array in the shape [num_timesteps, 4 (corners), 2 (dimensions)]
    """
    centroids = np.column_stack([x_center, y_center])

    # Precalculate all components needed for the corner calculation
    l = length / 2
    w = width / 2
    c = np.cos(heading)
    s = np.sin(heading)

    lc = l * c
    ls = l * s
    wc = w * c
    ws = w * s

    # Calculate all four rotated bbox corner positions assuming the object is located at the origin.
    # To do so, rotate the corners at [+/- length/2, +/- width/2] as given by the orientation.
    # Use a vectorized approach using precalculated components for maximum efficiency
    rotated_bbox_vertices = np.empty((centroids.shape[0], 4, 2))

    # Front-right corner
    rotated_bbox_vertices[:, 0, 0] = lc - ws
    rotated_bbox_vertices[:, 0, 1] = ls + wc

    # Rear-right corner
    rotated_bbox_vertices[:, 1, 0] = -lc - ws
    rotated_bbox_vertices[:, 1, 1] = -ls + wc

    # Rear-left corner
    rotated_bbox_vertices[:, 2, 0] = -lc + ws
    rotated_bbox_vertices[:, 2, 1] = -ls - wc

    # Front-left corner
    rotated_bbox_vertices[:, 3, 0] = lc + ws
    rotated_bbox_vertices[:, 3, 1] = ls - wc

    # Move corners of rotated bounding box from the origin to the object's location
    rotated_bbox_vertices = rotated_bbox_vertices + np.expand_dims(centroids, axis=1)
    return rotated_bbox_vertices
