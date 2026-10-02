"""Load RealSense calibration and map it to Isaac Sim pinhole parameters."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics plus the source distortion description."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    distortion_model: str
    coeffs: tuple[float, ...]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CameraIntrinsics":
        required = ("width", "height", "fx", "fy", "cx_ppx", "cy_ppy")
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"camera intrinsics missing fields: {missing}")
        intrinsics = cls(
            width=int(value["width"]),
            height=int(value["height"]),
            fx=float(value["fx"]),
            fy=float(value["fy"]),
            cx=float(value["cx_ppx"]),
            cy=float(value["cy_ppy"]),
            distortion_model=str(value.get("distortion_model", "unknown")),
            coeffs=tuple(float(item) for item in value.get("coeffs", ())),
        )
        intrinsics.validate()
        return intrinsics

    def validate(self) -> None:
        values = (self.fx, self.fy, self.cx, self.cy)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("camera width and height must be positive")
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("camera fx and fy must be positive")
        if not all(math.isfinite(value) for value in values):
            raise ValueError("camera intrinsics must be finite")

    def matrix(self) -> list[list[float]]:
        return [
            [self.fx, 0.0, self.cx],
            [0.0, self.fy, self.cy],
            [0.0, 0.0, 1.0],
        ]

    def center_crop_and_scale(self, width: int, height: int) -> "CameraIntrinsics":
        """Center-crop to the target aspect ratio, then scale to target pixels."""
        if width <= 0 or height <= 0:
            raise ValueError("target width and height must be positive")

        source_aspect = self.width / self.height
        target_aspect = width / height
        if source_aspect > target_aspect:
            crop_height = float(self.height)
            crop_width = crop_height * target_aspect
            crop_x = (self.width - crop_width) / 2.0
            crop_y = 0.0
        else:
            crop_width = float(self.width)
            crop_height = crop_width / target_aspect
            crop_x = 0.0
            crop_y = (self.height - crop_height) / 2.0

        scale_x = width / crop_width
        scale_y = height / crop_height
        return CameraIntrinsics(
            width=width,
            height=height,
            fx=self.fx * scale_x,
            fy=self.fy * scale_y,
            cx=(self.cx - crop_x) * scale_x,
            cy=(self.cy - crop_y) * scale_y,
            distortion_model=self.distortion_model,
            coeffs=self.coeffs,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "width": self.width,
            "height": self.height,
            "fx": self.fx,
            "fy": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "distortion_model": self.distortion_model,
            "coeffs": list(self.coeffs),
            "matrix": self.matrix(),
        }


@dataclass(frozen=True)
class CameraCalibration:
    """RealSense depth/color calibration loaded from one JSON file."""

    source_path: Path
    device_info: dict[str, Any]
    depth: CameraIntrinsics
    color: CameraIntrinsics
    depth_scale_m_per_unit: float
    extrinsics: dict[str, Any]
    raw: dict[str, Any]


@dataclass(frozen=True)
class IsaacSimIntrinsicsMapping:
    """Requested RealSense intrinsics and the effective Omniverse pinhole model."""

    original: CameraIntrinsics
    requested: CameraIntrinsics
    effective: CameraIntrinsics
    camera_info: CameraIntrinsics
    camera_info_mode: str
    focal_length_mm: float
    horizontal_aperture_mm: float
    vertical_aperture_mm: float
    limitations: tuple[str, ...]

    def effective_minus_requested(self) -> dict[str, float]:
        return {
            "fx": self.effective.fx - self.requested.fx,
            "fy": self.effective.fy - self.requested.fy,
            "cx": self.effective.cx - self.requested.cx,
            "cy": self.effective.cy - self.requested.cy,
        }

    def as_dict(self) -> dict[str, Any]:
        return {
            "original_color_intrinsics": self.original.as_dict(),
            "requested_intrinsics": self.requested.as_dict(),
            "cropped_scaled_requested_intrinsics": self.requested.as_dict(),
            "isaacsim_effective_intrinsics": self.effective.as_dict(),
            "camera_info_intrinsics": self.camera_info.as_dict(),
            "original_color_K": self.original.matrix(),
            "requested_K": self.requested.matrix(),
            "isaacsim_effective_K": self.effective.matrix(),
            "camera_info_K": self.camera_info.matrix(),
            "camera_info_mode": self.camera_info_mode,
            "isaacsim_effective_minus_requested": self.effective_minus_requested(),
            "focal_length_mm": self.focal_length_mm,
            "horizontal_aperture_mm": self.horizontal_aperture_mm,
            "vertical_aperture_mm": self.vertical_aperture_mm,
            "limitations": list(self.limitations),
            "depth_color_alignment": (
                "IsaacSim rgb and distance_to_image_plane are emitted by one CameraCfg, "
                "so simulated RGB-D is pixel-aligned like RealSense aligned_depth_to_color. "
                "This does not simulate RealSense raw depth-to-color alignment."
            ),
        }


def load_realsense_config(path: str | Path) -> CameraCalibration:
    """Load and validate the checked-in RealSense calibration JSON."""
    source_path = Path(path).expanduser().resolve()
    try:
        raw = json.loads(source_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"camera config does not exist: {source_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid camera config JSON {source_path}: {exc}") from exc

    if not isinstance(raw, dict):
        raise ValueError("camera config root must be a JSON object")
    try:
        depth_block = raw["depth"]
        color_block = raw["color"]
        extrinsics = raw["extrinsics"]
        depth_scale = float(depth_block["depth_scale_m_per_unit"])
        depth = CameraIntrinsics.from_dict(depth_block["intrinsics"])
        color = CameraIntrinsics.from_dict(color_block["intrinsics"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"incomplete RealSense camera config {source_path}: {exc}") from exc
    if depth_scale <= 0.0 or not math.isfinite(depth_scale):
        raise ValueError("depth_scale_m_per_unit must be finite and positive")
    if not isinstance(extrinsics, dict):
        raise ValueError("extrinsics must be a JSON object")
    for name in ("depth_to_color", "color_to_depth"):
        block = extrinsics.get(name)
        if not isinstance(block, dict):
            raise ValueError(f"extrinsics.{name} is missing")
        if len(block.get("rotation_row_major_3x3", ())) != 9:
            raise ValueError(f"extrinsics.{name}.rotation_row_major_3x3 must contain 9 values")
        if len(block.get("translation_m", ())) != 3:
            raise ValueError(f"extrinsics.{name}.translation_m must contain 3 values")

    return CameraCalibration(
        source_path=source_path,
        device_info=dict(raw.get("device_info", {})),
        depth=depth,
        color=color,
        depth_scale_m_per_unit=depth_scale,
        extrinsics=extrinsics,
        raw=raw,
    )


def map_color_intrinsics_to_isaacsim(
    calibration: CameraCalibration,
    width: int,
    height: int,
    *,
    focal_length_mm: float = 24.0,
    camera_info_mode: str = "effective",
) -> IsaacSimIntrinsicsMapping:
    """Map RealSense color intrinsics to the effective Omniverse pinhole camera."""
    if focal_length_mm <= 0.0 or not math.isfinite(focal_length_mm):
        raise ValueError("focal_length_mm must be finite and positive")
    if camera_info_mode not in ("effective", "real"):
        raise ValueError("camera_info_mode must be 'effective' or 'real'")
    requested = calibration.color.center_crop_and_scale(width, height)
    effective_focal_pixels = (requested.fx + requested.fy) / 2.0
    horizontal_aperture_mm = focal_length_mm * width / effective_focal_pixels
    vertical_aperture_mm = horizontal_aperture_mm * height / width
    effective = CameraIntrinsics(
        width=width,
        height=height,
        fx=effective_focal_pixels,
        fy=effective_focal_pixels,
        cx=width / 2.0,
        cy=height / 2.0,
        distortion_model="none",
        coeffs=(0.0, 0.0, 0.0, 0.0, 0.0),
    )
    if camera_info_mode == "effective":
        camera_info = effective
    else:
        camera_info = CameraIntrinsics(
            width=width,
            height=height,
            fx=requested.fx,
            fy=requested.fy,
            cx=requested.cx,
            cy=requested.cy,
            distortion_model="none",
            coeffs=(0.0, 0.0, 0.0, 0.0, 0.0),
        )
    return IsaacSimIntrinsicsMapping(
        original=calibration.color,
        requested=requested,
        effective=effective,
        camera_info=camera_info,
        camera_info_mode=camera_info_mode,
        focal_length_mm=focal_length_mm,
        horizontal_aperture_mm=horizontal_aperture_mm,
        vertical_aperture_mm=vertical_aperture_mm,
        limitations=(
            "camera_info_mode=effective publishes the same K that IsaacSim rendering uses.",
            "camera_info_mode=real publishes the RealSense color K after crop/scale to the output resolution.",
            "IsaacSim/Omniverse pinhole rendering uses square pixels, so fx and fy are averaged.",
            "IsaacSim/Omniverse pinhole rendering centers the principal point.",
            "RealSense lens distortion is recorded but not reproduced by this pinhole camera.",
            "RealSense depth/color extrinsics are internal sensor extrinsics, not passed to the single CameraCfg.",
        ),
    )
