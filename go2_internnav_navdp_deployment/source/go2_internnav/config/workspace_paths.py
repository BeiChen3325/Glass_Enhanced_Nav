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
    navdp_root: Path
    isaaclab_root: Path
    navdp_checkpoint_path: Path
    x_navdp_checkpoint_path: Path
    dav2_checkpoint_path: Path
    realsense_config: Path
    conda_sh: Path

    def as_log_dict(self) -> dict[str, str]:
        return {
            "linux_user": self.linux_user,
            "workspace_root": str(self.workspace_root),
            "deployment_root": str(self.deployment_root),
            "navdp_root": str(self.navdp_root),
            "isaaclab_root": str(self.isaaclab_root),
            "navdp_checkpoint": str(self.navdp_checkpoint_path),
            "x_navdp_checkpoint": str(self.x_navdp_checkpoint_path),
            "dav2_checkpoint": str(self.dav2_checkpoint_path),
            "realsense_config": str(self.realsense_config),
            "conda_sh": str(self.conda_sh),
            "server_host": self.server_host,
            "server_port": str(self.server_port),
            "server_device": self.server_device,
            "camera_info_mode": self.camera_info_mode,
        }


def load_workspace_paths(env_path: Path | None = None) -> WorkspacePaths:
    resolved_env_path = WORKSPACE_ENV if env_path is None else env_path.expanduser().resolve()
    values = _read_env_file(resolved_env_path)

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

    # Anchor all repository paths to the deployment folder itself.  The same
    # checkout can therefore run under /home/unitree on Go2 or /home/robot on
    # the GPU machine without editing source code.
    deployment_root = DEPLOYMENT_ROOT
    workspace_root = deployment_root.parent
    home_root = workspace_root.parent
    linux_user = home_root.name
    workspace_name = workspace_root.name
    navdp_root = workspace_root / "NavDP"
    isaaclab_root = workspace_root / "IsaacLab"

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
        navdp_root=navdp_root,
        isaaclab_root=isaaclab_root,
        navdp_checkpoint_path=navdp_root / "model" / "navdp-cross-modal.ckpt",
        x_navdp_checkpoint_path=navdp_root / "model" / "x-navdp_posttrain.ckpt",
        dav2_checkpoint_path=navdp_root / "model" / "depth_anything_v2_vits.pth",
        realsense_config=deployment_root / "configs/realsense_d435i.json",
        conda_sh=home_root / conda_dir_name / "etc/profile.d/conda.sh",
    )


def require_path(path: Path, description: str) -> Path:
    resolved_path = path.expanduser().resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"{description} does not exist: {resolved_path}")
    return resolved_path
