"""Dependency-free helpers for the NavDP navigation status ROS topic."""

from __future__ import annotations

import json
import math
import time
from typing import Any

from .topic_names import NAVIGATION_STATUS_TOPIC

KNOWN_NAVIGATION_STATES = frozenset(
    {
        "starting",
        "waiting_for_plan",
        "running",
        "holding",
        "obstacle_blocked",
        "goal_reached",
        "model_stopped",
        "manual_override",
        "shutting_down",
        "unknown",
    }
)


def encode_navigation_status(
    state: str,
    reason: str,
    *,
    goal_mode: str | None = None,
    distance_to_goal_m: float | None = None,
    timestamp_unix_s: float | None = None,
) -> str:
    """Build the small JSON payload carried by ``NAVIGATION_STATUS_TOPIC``."""

    normalized_state = str(state).strip() or "unknown"
    normalized_reason = str(reason).strip() or "unspecified"
    payload: dict[str, Any] = {
        "state": normalized_state,
        "reason": normalized_reason,
        "timestamp_unix_s": (
            time.time() if timestamp_unix_s is None else float(timestamp_unix_s)
        ),
    }
    if goal_mode is not None:
        payload["goal_mode"] = str(goal_mode)
    if distance_to_goal_m is not None and math.isfinite(float(distance_to_goal_m)):
        payload["distance_to_goal_m"] = float(distance_to_goal_m)
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def decode_navigation_status(payload: str) -> dict[str, Any]:
    """Parse a status payload without allowing malformed data to break safety logs."""

    try:
        decoded = json.loads(payload)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {"state": "unknown", "reason": "malformed_status_payload"}
    if not isinstance(decoded, dict):
        return {"state": "unknown", "reason": "malformed_status_payload"}

    state = str(decoded.get("state", "unknown")).strip() or "unknown"
    reason = str(decoded.get("reason", "unspecified")).strip() or "unspecified"
    result: dict[str, Any] = {"state": state, "reason": reason}
    for key in ("goal_mode", "timestamp_unix_s", "distance_to_goal_m"):
        if key in decoded:
            result[key] = decoded[key]
    return result


def effective_stop_cause(
    *,
    manual_override_latched: bool,
    cmd_timed_out: bool,
    navigation_state: str,
    navigation_reason: str,
) -> str:
    """Explain why the bridge currently reports a physical StopMove action."""

    if manual_override_latched:
        return "manual_override"
    if navigation_state == "goal_reached":
        return navigation_reason or "pointgoal_reached"
    if navigation_state == "obstacle_blocked":
        return navigation_reason or "depth_obstacle"
    if navigation_state in {"holding", "model_stopped", "shutting_down"}:
        return navigation_reason or navigation_state
    if cmd_timed_out:
        return "cmd_timeout"
    return "none"
