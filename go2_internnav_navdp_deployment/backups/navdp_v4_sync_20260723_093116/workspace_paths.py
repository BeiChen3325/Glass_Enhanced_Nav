"""Machine-local workspace path configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEPLOYMENT_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE_ENV = DEPLOYMENT_ROOT / "configs" / "workspace_paths.env"


def _strip_optional_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _read_env_file(env_path: Path) -> dict[str, str]:
    if not env_path.is_file():
        raise FileNotFoundError(
            "workspace path config is missing: "
            f"{env_path}\n"
            "Expected deployment config file: configs/workspace_paths.env"
        )

    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            raise ValueError(f"invalid env line in {env_path}:{line_number}: {raw_line!r}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = _strip_optional_quotes(value.strip())
        if not key:
            raise ValueError(f"empty env key in {env_path}:{line_number}")
        values[key] = value
    return values


def _require_value(values: dict[str, str], key: str, env_path: Path) -> str:
    value = values.get(key, "").strip()
    if not value:
        raise ValueError(f"{key} is required in {env_path}")
    return value


def _require_int(values: dict[str, str], key: str, env_path: Path) -> int:
    value = _require_value(values, key, env_path)
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer in {env_path}: {value!r}") from exc


@dataclass(frozen=True)
class WorkspacePaths:
    linux_user: str
    workspace_name: str
    conda_dir_name: str
    server_host: str
    server_port: int
    server_device: str
    camera_info_mode: str
    home_root: Path
    workspace_root: Path
    deployment_root: Path
    internnav_root: Path
    navdp_root: Path
    isaaclab_root: Path
    checkpoints_root: Path
    internvla_navdp_model_path: Path
    navdp_checkpoint_path: Path
    navdp_cluttered_easy_usd: Path
    realsense_config: Path
    camera_pose_presets: Path
    conda_sh: Path

    def as_log_dict(self) -> dict[str, str]:
        return {
            "linux_user": self.linux_user,
            "workspace_root": str(self.workspace_root),
            "deployment_root": str(self.deployment_root),
            "internnav_root": str(self.internnav_root),
            "navdp_root": str(self.navdp_root),
            "isaaclab_root": str(self.isaaclab_root),
            "model_path": str(self.internvla_navdp_model_path),
            "navdp_checkpoint": str(self.navdp_checkpoint_path),
            "navdp_scene_usd": str(self.navdp_cluttered_easy_usd),
            "realsense_config": str(self.realsense_config),
            "camera_pose_presets": str(self.camera_pose_presets),
            "conda_sh": str(self.conda_sh),
            "server_host": self.server_host,
            "server_port": str(self.server_port),
            "server_device": self.server_device,
            "camera_info_mode": self.camera_info_mode,
        }


def load_workspace_paths(env_path: Path | None = None) -> WorkspacePaths:
    resolved_env_path = WORKSPACE_ENV if env_path is None else env_path.expanduser().resolve()
    values = _read_env_file(resolved_env_path)

    linux_user = _require_value(values, "NAVDP_LINUX_USER", resolved_env_path)
    workspace_name = _require_value(values, "NAVDP_WORKSPACE_NAME", resolved_env_path)
    conda_dir_name = _require_value(values, "NAVDP_CONDA_DIR_NAME", resolved_env_path)
    server_host = _require_value(values, "NAVDP_SERVER_HOST", resolved_env_path)
    server_port = _require_int(values, "NAVDP_SERVER_PORT", resolved_env_path)
    server_device = _require_value(values, "NAVDP_SERVER_DEVICE", resolved_env_path)
    camera_info_mode = _require_value(values, "NAVDP_CAMERA_INFO_MODE", resolved_env_path)
    if camera_info_mode not in {"effective", "real"}:
        raise ValueError(
            "NAVDP_CAMERA_INFO_MODE must be 'effective' or 'real' in "
            f"{resolved_env_path}: {camera_info_mode!r}"
        )

    home_root = Path("/home") / linux_user
    workspace_root = home_root / workspace_name
    deployment_root = workspace_root / "go2_internnav_navdp_deployment"
    internnav_root = workspace_root / "InternNav"
    navdp_root = workspace_root / "NavDP"
    isaaclab_root = workspace_root / "IsaacLab"
    checkpoints_root = workspace_root / "checkpoints"

    return WorkspacePaths(
        linux_user=linux_user,
        workspace_name=workspace_name,
        conda_dir_name=conda_dir_name,
        server_host=server_host,
        server_port=server_port,
        server_device=server_device,
        camera_info_mode=camera_info_mode,
        home_root=home_root,
        workspace_root=workspace_root,
        deployment_root=deployment_root,
        internnav_root=internnav_root,
        navdp_root=navdp_root,
        isaaclab_root=isaaclab_root,
        checkpoints_root=checkpoints_root,
        internvla_navdp_model_path=checkpoints_root / "InternVLA-N1-w-NavDP",
        # Easy4 Mixed V3 deployment candidates. Keep exactly one active.
        #navdp_checkpoint_path=(navdp_root/"outputs/go2w_easy4_mixed_v3_train/milestones/navdp_v3_step_004000.ckpt"),
        #navdp_checkpoint_path=(navdp_root/ "outputs/go2w_easy4_mixed_v3_train/milestones/navdp_v3_step_008000.ckpt" ),
        # navdp_checkpoint_path=(navdp_root / "outputs/go2w_easy4_mixed_v3_train/milestones/navdp_v3_step_012000.ckpt"),
        navdp_checkpoint_path=(navdp_root/ "outputs/go2w_easy4_mixed_v3_train/milestones/navdp_v3_step_016450.ckpt"),
        # navdp_checkpoint_path=(navdp_root / "outputs/go2w_random_circular_v2_train/navdp_v2_latest.ckpt"),
        # navdp_checkpoint_path=navdp_root / "navdp-cross-modal.ckpt",
        navdp_cluttered_easy_usd=navdp_root/ "assets/scenes/cluttered_easy/easy_0/cluttered-0.usd",
        #navdp_cluttered_easy_usd=navdp_root/ "assets/scenes/internscenes_commercial/scenes_commercial/MV5M25QKTKJZ2AABAAAAAAA8_usd/start_result_navigation.usd",
        
        realsense_config=deployment_root / "configs/realsense_d435i.json",
        camera_pose_presets=deployment_root / "configs/camera_pose_presets.yaml",
        conda_sh=home_root / conda_dir_name / "etc/profile.d/conda.sh",
    )


def require_path(path: Path, description: str) -> Path:
    resolved_path = path.expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"{description} does not exist: {resolved_path}")
    return resolved_path
