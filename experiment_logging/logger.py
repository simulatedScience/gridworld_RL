"""Structured JSON logger for environment runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_timestamp() -> str:
    """Return current UTC timestamp in ISO-8601 format."""

    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True, slots=True)
class EpisodeSummary:
    """Compact summary for one completed episode."""

    episode_index: int
    steps: int
    total_return: float
    terminated: bool
    truncated: bool
    reached_goal: bool
    hazard_hit: bool
    slip_count: int
    final_position: tuple[int, int]


class JsonRunLogger:
    """Write structured JSON logs for a run.

    Logs are written as JSON Lines in ``experiments/logs/<run_id>.jsonl``.
    Each line is an independent event object.
    """

    def __init__(
        self,
        run_name: str,
        output_dir: str | Path = "experiments/logs",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Initialize logger and emit a ``run_started`` event.

        Args:
            run_name: Human-readable run label.
            output_dir: Directory where run logs are stored.
            metadata: Optional run metadata (e.g., config path, seed).
        """

        self.run_id: str = f"{run_name}_{uuid4().hex[:8]}"
        self.run_name: str = run_name
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._log_path = self._output_dir / f"{self.run_id}.jsonl"
        self._stream = self._log_path.open("w", encoding="utf-8")

        self.log_event(
            event_type="run_started",
            payload={
                "run_id": self.run_id,
                "run_name": self.run_name,
                "metadata": metadata or {},
            },
        )

    @property
    def log_path(self) -> Path:
        """Return the file path where logs are stored."""

        return self._log_path

    def log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Write one structured event.

        Args:
            event_type: Event category string.
            payload: JSON-serializable event data.
        """

        event = {
            "timestamp": utc_timestamp(),
            "event_type": event_type,
            "payload": payload,
        }
        self._stream.write(json.dumps(event) + "\n")
        self._stream.flush()

    def log_episode(self, summary: EpisodeSummary) -> None:
        """Log one completed episode summary."""

        self.log_event(
            event_type="episode_finished",
            payload={
                "episode_index": summary.episode_index,
                "steps": summary.steps,
                "total_return": summary.total_return,
                "terminated": summary.terminated,
                "truncated": summary.truncated,
                "reached_goal": summary.reached_goal,
                "hazard_hit": summary.hazard_hit,
                "slip_count": summary.slip_count,
                "final_position": list(summary.final_position),
            },
        )

    def close(self) -> None:
        """Emit ``run_finished`` and close the stream."""

        self.log_event(
            event_type="run_finished",
            payload={
                "run_id": self.run_id,
                "run_name": self.run_name,
            },
        )
        self._stream.close()
