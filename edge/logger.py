"""
ExperimentLogger: Records and persists experiment results for Edge-Cloud Demo scenarios.

Buffers LogRecord entries in memory, captures system metrics (CPU/RAM) at log time,
saves results to CSV with a SUMMARY row appended, and computes aggregate statistics.
"""

from __future__ import annotations

import dataclasses
import os
import sys
from datetime import datetime

import pandas as pd
import psutil

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from models import ExperimentSummary, LogRecord


class ExperimentLogger:
    """
    Logs per-image results for an experiment scenario and saves them to CSV.

    Usage:
        logger = ExperimentLogger("results/edge_only.csv", "edge_only")
        logger.log_image_result(record)
        ...
        summary = logger.compute_summary()
        logger.save()
    """

    def __init__(self, output_path: str, scenario: str) -> None:
        """
        Initialise the logger.

        Args:
            output_path: Destination CSV file path (e.g. "results/edge_only.csv").
            scenario: One of "cloud_only", "edge_only", or "edge_cloud".
        """
        self.output_path: str = output_path
        self.scenario: str = scenario
        self._records: list[dict] = []
        self.start_time: str = datetime.now().isoformat()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_image_result(self, record: LogRecord) -> None:
        """
        Append a LogRecord to the internal buffer.

        Captures current CPU and RAM usage via psutil and writes them
        into the record's ``cpu_percent`` and ``ram_mb`` fields before
        appending the record as a dict.

        Args:
            record: A LogRecord instance (cpu_percent / ram_mb will be overwritten).
        """
        record.cpu_percent = psutil.cpu_percent(interval=0.1)
        record.ram_mb = round(
            psutil.Process().memory_info().rss / (1024 * 1024), 2
        )
        self._records.append(dataclasses.asdict(record))

    def save(self) -> None:
        """
        Persist the buffer to a CSV file and append a SUMMARY row.

        Behaviour:
        - Records end_time at call time.
        - Creates the parent directory if it does not exist.
        - Converts the buffer to a pandas DataFrame and writes it with
          ``encoding="utf-8"`` and ``index=False``, overwriting any
          existing file.
        - Appends a SUMMARY row (plain text) at the end of the file.
        - On any error: prints the error to stderr and returns without
          raising an exception.
        """
        end_time = datetime.now().isoformat()

        try:
            # Ensure the output directory exists
            output_dir = os.path.dirname(self.output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            # Build DataFrame from buffer
            df = pd.DataFrame(self._records)

            # Write CSV (overwrite)
            df.to_csv(self.output_path, index=False, encoding="utf-8")

            # Compute summary for the SUMMARY row
            summary = self._compute_summary_at(end_time)

            summary_line = (
                f"SUMMARY,"
                f"{summary.avg_latency_ms:.2f},"
                f"{summary.total_uploaded_bytes},"
                f"{summary.throughput_img_per_sec:.4f},"
                f"{summary.avg_confidence:.4f},"
                f"{summary.avg_cpu_percent:.2f},"
                f"{summary.avg_ram_mb:.2f},"
                f"{summary.offload_ratio:.2f}%\n"
            )
            with open(self.output_path, "a", encoding="utf-8") as fh:
                fh.write(summary_line)

        except Exception as exc:  # noqa: BLE001
            print(
                f"[ExperimentLogger] Error saving CSV to '{self.output_path}': {exc}",
                file=sys.stderr,
            )

    def compute_summary(self) -> ExperimentSummary:
        """
        Compute aggregate statistics over all buffered LogRecords.

        Returns:
            An ExperimentSummary with zeros/empty strings when the buffer
            is empty.
        """
        end_time = datetime.now().isoformat()
        return self._compute_summary_at(end_time)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_summary_at(self, end_time: str) -> ExperimentSummary:
        """
        Internal helper that computes the summary using a given end_time string.

        Args:
            end_time: ISO 8601 string representing the end of the experiment.

        Returns:
            An ExperimentSummary instance.
        """
        total = len(self._records)

        if total == 0:
            return ExperimentSummary(
                scenario=self.scenario,
                total_images=0,
                avg_latency_ms=0.0,
                total_uploaded_bytes=0,
                throughput_img_per_sec=0.0,
                avg_confidence=0.0,
                avg_cpu_percent=0.0,
                avg_ram_mb=0.0,
                offload_ratio=0.0,
                start_time=self.start_time,
                end_time=end_time,
                total_duration_sec=0.0,
            )

        # Compute total duration
        try:
            start_dt = datetime.fromisoformat(self.start_time)
            end_dt = datetime.fromisoformat(end_time)
            total_duration_sec = (end_dt - start_dt).total_seconds()
        except (ValueError, TypeError):
            total_duration_sec = 0.0

        # Guard against zero duration for throughput
        throughput = total / total_duration_sec if total_duration_sec > 0 else 0.0

        avg_latency_ms = sum(r["latency_ms"] for r in self._records) / total
        total_uploaded_bytes = sum(r["uploaded_bytes"] for r in self._records)
        avg_confidence = sum(r["confidence"] for r in self._records) / total
        avg_cpu_percent = sum(r["cpu_percent"] for r in self._records) / total
        avg_ram_mb = sum(r["ram_mb"] for r in self._records) / total
        offload_count = sum(1 for r in self._records if r["offloaded"])
        offload_ratio = (offload_count / total) * 100.0 if total > 0 else 0.0

        return ExperimentSummary(
            scenario=self.scenario,
            total_images=total,
            avg_latency_ms=avg_latency_ms,
            total_uploaded_bytes=total_uploaded_bytes,
            throughput_img_per_sec=throughput,
            avg_confidence=avg_confidence,
            avg_cpu_percent=avg_cpu_percent,
            avg_ram_mb=avg_ram_mb,
            offload_ratio=offload_ratio,
            start_time=self.start_time,
            end_time=end_time,
            total_duration_sec=total_duration_sec,
        )
