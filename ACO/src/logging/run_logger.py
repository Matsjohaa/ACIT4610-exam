"""Lightweight CSV logger for iterative optimization runs."""

from __future__ import annotations

import csv
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, MutableMapping, Optional


@dataclass
class RunLogger:
    """Collect and persist iteration-level metrics with shared metadata."""

    base_dir: Path
    filename: Optional[str] = None
    metadata: Optional[Dict[str, object]] = None
    field_order: Optional[Iterable[str]] = None

    _records: List[MutableMapping[str, object]] = field(default_factory=list, init=False)
    _resolved_path: Optional[Path] = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.base_dir = Path(self.base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        if self.metadata is None:
            self.metadata = {}
        else:
            self.metadata = dict(self.metadata)

    def log_iteration(self, **metrics: object) -> None:
        """Buffer metrics for an iteration."""

        record: MutableMapping[str, object] = {
            'timestamp': dt.datetime.utcnow().isoformat(timespec='milliseconds') + 'Z'
        }
        if self.metadata:
            record.update(self.metadata)
        record.update(metrics)
        self._records.append(record)

    def update_metadata(self, **extra: object) -> None:
        """Merge additional metadata that all future rows will share."""

        if self.metadata is None:
            self.metadata = {}
        self.metadata.update(extra)

    def flush(self) -> Path:
        """Write buffered records to disk and return the file path."""

        if not self._records:
            raise RuntimeError("No records to write; did you call log_iteration()?" )

        path = self._resolve_path()
        fieldnames = self._determine_fieldnames()

        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for record in self._records:
                writer.writerow(record)

        return path

    def _resolve_path(self) -> Path:
        if self._resolved_path is None:
            filename = self.filename or f"run_{dt.datetime.utcnow().strftime('%Y%m%dT%H%M%S')}.csv"
            self._resolved_path = self.base_dir / filename
        return self._resolved_path

    def _determine_fieldnames(self) -> List[str]:
        if self.field_order:
            return list(self.field_order)

        keys: List[str] = []
        for record in self._records:
            for key in record.keys():
                if key not in keys:
                    keys.append(key)
        return keys

