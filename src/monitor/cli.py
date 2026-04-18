"""
CLI entry point for the Child Monitor Analyzer.

Provides a command-line interface to analyse Hebrew audio files and
produce a detection report (console summary + optional JSON export).

Usage:
    python -m monitor recording.wav
    python -m monitor recording.wav -o report.json --verbose
    python -m monitor recording.wav --no-ai
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .models import AnalysisReport
from .pipeline import AnalysisPipeline

log = logging.getLogger(__name__)

# ===========================
# ARGUMENT PARSING
# ===========================


def parse_arguments(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse and validate command-line arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:]).

    Returns:
        Parsed namespace with ``audio``, ``output``, ``no_ai``, ``verbose``.
    """
    parser = argparse.ArgumentParser(
        prog="monitor",
        description="Child Monitor Analyzer -- analyse Hebrew audio for profanity, "
                    "shouting, crying, and other events with precise timestamps.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
    monitor recording.wav
    monitor recording.wav -o report.json
    monitor recording.wav --no-ai --verbose
        """,
    )
    parser.add_argument("audio", type=Path, help="Path to the audio file to analyse")
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="Export JSON report to this file path",
    )
    parser.add_argument(
        "--no-ai", action="store_true",
        help="Disable AI profanity classification (word-list only)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    args = parser.parse_args(argv)

    # Post-parse validation.
    if not args.audio.exists():
        parser.error(f"Audio file not found: {args.audio}")

    return args


# ===========================
# REPORT DISPLAY
# ===========================


def _print_report(report: AnalysisReport) -> None:
    """Print a human-readable summary to the console.

    Uses ``print()`` here because this IS the program's intended output,
    not a status/diagnostic message (which goes through ``logging``).

    Args:
        report: The completed analysis report.
    """
    print()
    print("=" * 60)
    print(f"  File: {report.audio_path}")
    minutes, seconds = divmod(int(report.duration_seconds), 60)
    print(f"  Duration: {minutes:02d}:{seconds:02d}")
    print(f"  Speech segments: {len(report.segments)}")
    print(f"  Detections: {len(report.detections)}")
    print("=" * 60)

    if report.detections:
        print()
        # Table header -- Hebrew labels.
        print(f"  {'Time':<8}  {'Type':<15}  {'Confidence':<12}  Details")
        print("  " + "-" * 56)

        for detection in report.detections_sorted():
            details_str = _format_detection_details(detection)
            print(
                f"  {detection.time_display:<8}  {detection.label_he:<15}  "
                f"{detection.confidence:.0%}{'':>9}  {details_str}"
            )
    else:
        print("\n  No detections found.")

    if report.full_transcription:
        print()
        print("  Full transcription:")
        print("  " + "-" * 56)
        # Wrap long transcription text at ~70 characters per line.
        text = report.full_transcription
        while text:
            print(f"  {text[:70]}")
            text = text[70:]

    print()


def _format_detection_details(detection) -> str:
    """Format the details column for a single detection row.

    Args:
        detection: A Detection object.

    Returns:
        Human-readable details string.
    """
    if detection.profanity_words:
        return ", ".join(detection.profanity_words)
    if "audioset_class" in detection.details:
        return detection.details["audioset_class"]
    if "rms" in detection.details:
        return f"RMS={detection.details['rms']}"
    return ""


# ===========================
# MAIN ENTRY POINT
# ===========================


def main(argv: Optional[List[str]] = None) -> None:
    """Entry point for the ``monitor`` CLI command.

    Args:
        argv: Optional argument list for testing; defaults to sys.argv.
    """
    args = parse_arguments(argv)

    # Configure logging -- verbose flag sets DEBUG level.
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    log.info("Starting analysis of %s", args.audio)

    pipeline = AnalysisPipeline(use_ai_profanity=not args.no_ai)

    def _on_progress(pct: int, msg: str) -> None:
        # Progress goes to stderr so it does not mix with program output.
        print(f"\r  [{pct:3d}%] {msg}", end="", flush=True, file=sys.stderr)

    report = pipeline.analyze(args.audio, on_progress=_on_progress)
    print(file=sys.stderr)  # newline after progress

    _print_report(report)

    if args.output:
        args.output.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        log.info("JSON report saved to %s", args.output)


if __name__ == "__main__":
    main()
