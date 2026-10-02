"""Golden-run response recording and replay helpers."""

from __future__ import annotations

import copy
import json
import threading
import time
from pathlib import Path
from typing import Any


STOP_RESPONSE = {"discrete_action": [0]}


class GoldenRunFormatError(ValueError):
    """Raised when a golden-run JSONL file cannot be replayed safely."""


def validate_model_response(response: Any, *, context: str = "response") -> dict[str, Any]:
    if not isinstance(response, dict):
        raise GoldenRunFormatError(f"{context} must be a JSON object")
    if "trajectory" not in response and "discrete_action" not in response:
        raise GoldenRunFormatError(
            f"{context} must contain 'trajectory' or 'discrete_action'"
        )
    return copy.deepcopy(response)


def response_from_record(record: Any, *, line_number: int) -> dict[str, Any]:
    context = f"line {line_number}"
    if not isinstance(record, dict):
        raise GoldenRunFormatError(f"{context} must be a JSON object")

    if "response" in record:
        return validate_model_response(record["response"], context=f"{context} response")

    return validate_model_response(record, context=context)


def load_golden_run_responses(path: str | Path) -> list[dict[str, Any]]:
    resolved_path = Path(path).expanduser().resolve()
    if not resolved_path.is_file():
        raise FileNotFoundError(f"golden-run recording does not exist: {resolved_path}")

    responses: list[dict[str, Any]] = []
    for line_number, raw_line in enumerate(
        resolved_path.read_text(encoding="utf-8").splitlines(), 1
    ):
        line = raw_line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise GoldenRunFormatError(
                f"invalid JSON in {resolved_path}:{line_number}: {exc.msg}"
            ) from exc
        responses.append(response_from_record(record, line_number=line_number))

    if not responses:
        raise GoldenRunFormatError(f"golden-run recording is empty: {resolved_path}")
    return responses


class GoldenRunResponseRecorder:
    """Write one model response per JSONL record."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("w", encoding="utf-8")
        self._lock = threading.Lock()
        self._request_sequence = 0
        self._closed = False

    @property
    def request_sequence(self) -> int:
        return self._request_sequence

    def write(
        self,
        *,
        reset: bool,
        idx: int,
        response: dict[str, Any],
        elapsed_s: float,
        server_url: str,
        instruction: str | None = None,
        goal_mode: str | None = None,
        goal: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        validate_model_response(response)
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("record metadata must be a dictionary or None")
        with self._lock:
            if self._closed:
                return
            record = {
                "request_sequence": self._request_sequence,
                "timestamp_unix_s": time.time(),
                "elapsed_s": float(elapsed_s),
                "server_url": server_url,
                "reset": bool(reset),
                "idx": int(idx),
                "response": response,
            }
            if instruction is not None:
                record["instruction"] = instruction
            if goal_mode is not None:
                record["goal_mode"] = goal_mode
            if goal is not None:
                record["goal"] = copy.deepcopy(goal)
            if metadata is not None:
                reserved = set(record).intersection(metadata)
                if reserved:
                    names = ", ".join(sorted(reserved))
                    raise GoldenRunFormatError(
                        f"record metadata cannot overwrite core fields: {names}"
                    )
                # Keep inference_odom and evaluation fields at the record root:
                # the trajectory visualizer and replay diagnostics consume the
                # established flat JSONL schema.
                record.update(copy.deepcopy(metadata))
            self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
            self._file.flush()
            self._request_sequence += 1

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if not self._file.closed:
                self._file.close()

    def __enter__(self) -> "GoldenRunResponseRecorder":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class GoldenRunReplay:
    """Serve recorded model responses in request order."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self.responses = load_golden_run_responses(self.path)
        self._lock = threading.Lock()
        self._cursor = 0

    @property
    def cursor(self) -> int:
        with self._lock:
            return self._cursor

    def reset(self) -> None:
        with self._lock:
            self._cursor = 0

    def next_response(self) -> tuple[int | None, dict[str, Any], bool]:
        """Return (record_index, response, exhausted)."""
        with self._lock:
            if self._cursor >= len(self.responses):
                return None, copy.deepcopy(STOP_RESPONSE), True

            record_index = self._cursor
            response = copy.deepcopy(self.responses[record_index])
            self._cursor += 1
            return record_index, response, False
