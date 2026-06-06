"""
benchmark.py — MLOps Latency Benchmarking for DocumentScanner
==============================================================
Production-grade performance harness for the Automated Document Scanner &
Enhancer (Week 1, 12-Week CV Portfolio Sprint).

Features
--------
  ▸ Synthetic test-image generator (skewed, noisy, gradient-lit document)
  ▸ Per-stage mean / p95 / stddev / min / max latency over N iterations
  ▸ End-to-end SLA gate: PASS if p95 < target (default 200 ms)
  ▸ Multi-config comparison table (default / strict / fast presets)
  ▸ JSON report export for CI artefact storage
  ▸ System information header (CPU count, OpenCV build, Python version)
  ▸ Non-zero exit code on SLA breach (CI-friendly)
  ▸ Optional --image path to benchmark against a real photo

Usage
-----
    # 100 iterations on a synthetic 1080p image, default config
    python benchmark.py

    # 200 iterations, 720p synthetic image, fast preset
    python benchmark.py --n 200 --size 720 --config fast

    # Compare all three presets side-by-side
    python benchmark.py --compare

    # Use a real photo and export JSON report
    python benchmark.py --image photo.jpg --export report.json

    # Override SLA target
    python benchmark.py --sla 150
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from scanner import DocumentScanner, ScannerConfig, ScanResult

# ════════════════════════════════════════════════════════════════════════════
# 1.  SYNTHETIC DOCUMENT IMAGE GENERATOR
# ════════════════════════════════════════════════════════════════════════════

def make_synthetic_document(
    height: int = 1080,
    width: int = 810,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a realistic synthetic document photograph for deterministic
    benchmarking.

    Realism layers
    --------------
    1. Off-white paper background with a left→right luminance gradient
       (simulates a phone torch casting uneven light).
    2. A mildly skewed convex quadrilateral border — gives the contour
       detector a real 4-sided polygon to find.
    3. Horizontal ruled lines at variable opacity (text-line simulation).
    4. Sparse salt-and-pepper noise (camera sensor / JPEG artefact simulation).
    5. A subtle radial vignette darkening the corners.

    Parameters
    ----------
    height, width : int
        Output image dimensions in pixels.
    seed : int
        RNG seed for reproducibility across benchmark runs.

    Returns
    -------
    np.ndarray
        BGR uint8 image, shape (height, width, 3).
    """
    rng = np.random.default_rng(seed)

    # ── Base: off-white paper ────────────────────────────────────────────
    img = np.full((height, width, 3), 232, dtype=np.float32)

    # ── Gradient: bright left, dim right ────────────────────────────────
    gradient_h = np.linspace(1.0, 0.62, width, dtype=np.float32)
    gradient_v = np.linspace(0.95, 1.0,  height, dtype=np.float32)
    combined   = gradient_h[np.newaxis, :] * gradient_v[:, np.newaxis]
    img *= combined[:, :, np.newaxis]

    # ── Document quad (slightly skewed, not axis-aligned) ────────────────
    m, sk = 55, 38          # margin, skew amount
    doc_pts = np.array([
        [m + sk,        m          ],   # TL
        [width  - m,    m + sk // 2],   # TR
        [width  - m - sk, height - m],  # BR
        [m,             height - m - sk // 2],  # BL
    ], dtype=np.int32)

    doc_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(doc_mask, [doc_pts], 255)
    img[doc_mask == 255] = 246   # paper white inside quad
    cv2.polylines(img.astype(np.uint8), [doc_pts],
                  isClosed=True, color=(40, 40, 40), thickness=4)

    # ── Ruled lines (text simulation) ───────────────────────────────────
    line_ys = range(m + 90, height - m, 26)
    x1 = int(m + sk + 18)
    x2 = int(width - m - 18)
    for y in line_ys:
        alpha = float(rng.uniform(0.55, 0.90))
        color = int(165 * alpha)
        cv2.line(img.astype(np.uint8), (x1, y), (x2, y),
                 (color, color, color - 5), 1)

    # ── Salt-and-pepper noise ────────────────────────────────────────────
    noise = rng.integers(0, 100, (height, width), dtype=np.uint8)
    img_u8 = np.clip(img, 0, 255).astype(np.uint8)
    img_u8[noise < 2]  = 0
    img_u8[noise > 97] = 255

    # ── Radial vignette ──────────────────────────────────────────────────
    cx, cy = width / 2, height / 2
    Y, X   = np.ogrid[:height, :width]
    dist   = np.sqrt(((X - cx) / cx) ** 2 + ((Y - cy) / cy) ** 2)
    vignette = np.clip(1.0 - 0.35 * dist ** 2, 0.55, 1.0).astype(np.float32)
    img_u8 = np.clip(
        img_u8.astype(np.float32) * vignette[:, :, np.newaxis], 0, 255
    ).astype(np.uint8)

    return img_u8


# ════════════════════════════════════════════════════════════════════════════
# 2.  STATS HELPER
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class StageStats:
    """Descriptive statistics for one pipeline stage across N iterations."""
    stage_name: str
    mean_ms:    float
    p50_ms:     float
    p95_ms:     float
    p99_ms:     float
    stddev_ms:  float
    min_ms:     float
    max_ms:     float
    n:          int

    def as_dict(self) -> dict:
        return asdict(self)


def compute_stats(stage_name: str, latencies_ms: List[float]) -> StageStats:
    """Compute full descriptive statistics from a list of latency samples."""
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return StageStats(
        stage_name=stage_name,
        mean_ms=float(np.mean(arr)),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
        stddev_ms=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        min_ms=float(np.min(arr)),
        max_ms=float(np.max(arr)),
        n=len(latencies_ms),
    )


# ════════════════════════════════════════════════════════════════════════════
# 3.  BENCHMARK ENGINE
# ════════════════════════════════════════════════════════════════════════════

BenchmarkReport = Dict[str, StageStats]


def run_benchmark(
    scanner:      DocumentScanner,
    image:        np.ndarray,
    n_iterations: int = 100,
    warmup:       int = 5,
    verbose:      bool = True,
) -> BenchmarkReport:
    """
    Run the full DocumentScanner pipeline ``n_iterations`` times and return
    per-stage and end-to-end ``StageStats`` objects.

    Warm-up rationale
    -----------------
    The first few calls incur OS page-fault overhead, NumPy JIT paths, and
    Python's attribute-lookup caching.  We discard ``warmup`` iterations so
    they do not inflate the reported mean.

    Parameters
    ----------
    scanner      : DocumentScanner  — pre-configured scanner instance
    image        : np.ndarray       — BGR uint8 test image (same every run)
    n_iterations : int              — number of *measured* iterations
    warmup       : int              — number of *discarded* warm-up iterations
    verbose      : bool             — print progress dots to stdout

    Returns
    -------
    dict mapping stage_name → StageStats  (includes "END-TO-END" key)
    """
    # ── Warm-up ─────────────────────────────────────────────────────────
    if verbose:
        print(f"\n  Warm-up ({warmup} iterations) … ", end="", flush=True)
    for _ in range(warmup):
        scanner.scan(image)
    if verbose:
        print("done.")

    # ── Measurement loop ─────────────────────────────────────────────────
    stage_buckets: Dict[str, List[float]] = {}
    e2e_bucket:    List[float]            = []

    if verbose:
        print(f"  Measuring ({n_iterations} iterations):", flush=True)
        bar_width = 40

    for i in range(n_iterations):
        t0:    float     = time.perf_counter()
        result: ScanResult = scanner.scan(image)
        e2e_ms: float    = (time.perf_counter() - t0) * 1_000
        e2e_bucket.append(e2e_ms)

        for m in result.metrics:
            stage_buckets.setdefault(m.stage_name, []).append(m.latency_ms)

        if verbose:
            done = int(bar_width * (i + 1) / n_iterations)
            bar  = "█" * done + "░" * (bar_width - done)
            pct  = int(100 * (i + 1) / n_iterations)
            print(f"\r  [{bar}] {pct:3d}%  e2e={e2e_ms:6.1f}ms",
                  end="", flush=True)

    if verbose:
        print()   # newline after progress bar

    # ── Aggregate ────────────────────────────────────────────────────────
    report: BenchmarkReport = {}
    for name, lats in stage_buckets.items():
        report[name] = compute_stats(name, lats)
    report["END-TO-END"] = compute_stats("END-TO-END", e2e_bucket)
    return report


# ════════════════════════════════════════════════════════════════════════════
# 4.  SYSTEM INFORMATION
# ════════════════════════════════════════════════════════════════════════════

def collect_system_info() -> dict:
    """Collect host environment metadata for the report header."""
    cpu_count_logical  = os.cpu_count() or 1
    cv_build           = cv2.getBuildInformation()

    # Extract SIMD feature line from OpenCV build string
    simd_line = next(
        (ln.strip() for ln in cv_build.splitlines() if "CPU" in ln and "/" in ln),
        "N/A",
    )

    return {
        "python_version": sys.version.split()[0],
        "opencv_version": cv2.__version__,
        "numpy_version":  np.__version__,
        "platform":       platform.platform(),
        "cpu_count":      cpu_count_logical,
        "simd":           simd_line,
    }


# ════════════════════════════════════════════════════════════════════════════
# 5.  REPORT RENDERING
# ════════════════════════════════════════════════════════════════════════════

_W   = 82          # total table width
_SEP = "═" * _W
_DIV = "─" * _W


def _center(text: str) -> str:
    return text.center(_W)


def print_system_info(info: dict) -> None:
    print(_SEP)
    print(_center("DocumentScanner — MLOps Latency Benchmark"))
    print(_center("Automated Document Scanner & Enhancer · Week 1 Portfolio Sprint"))
    print(_SEP)
    print(f"  Python  : {info['python_version']}          "
          f"Platform : {info['platform']}")
    print(f"  OpenCV  : {info['opencv_version']}          "
          f"CPU cores: {info['cpu_count']}")
    print(f"  NumPy   : {info['numpy_version']}")
    print(_DIV)


def print_report(
    report:     BenchmarkReport,
    config_name: str = "default",
    sla_ms:     float = 200.0,
) -> bool:
    """
    Render the benchmark report table.

    Returns
    -------
    bool — True if the SLA is met (p95 end-to-end < sla_ms).
    """
    col_name = 36
    print(f"\n  Config: [{config_name}]")
    print(_DIV)
    print(
        f"  {'Stage':<{col_name}}"
        f"{'Mean':>8}  {'p50':>8}  {'p95':>8}  {'p99':>8}"
        f"  {'Std':>7}  {'Min':>7}  {'Max':>7}"
    )
    print(_DIV)

    for name, s in report.items():
        if name == "END-TO-END":
            print(_SEP)
        flag = ""
        if name == "END-TO-END":
            flag = " ✅" if s.p95_ms < sla_ms else " ❌"
        print(
            f"  {name:<{col_name}}"
            f"{s.mean_ms:>7.1f}ms"
            f"  {s.p50_ms:>7.1f}ms"
            f"  {s.p95_ms:>7.1f}ms"
            f"  {s.p99_ms:>7.1f}ms"
            f"  {s.stddev_ms:>6.2f}ms"
            f"  {s.min_ms:>6.1f}ms"
            f"  {s.max_ms:>6.1f}ms"
            f"{flag}"
        )

    e2e  = report["END-TO-END"]
    pass_ = e2e.p95_ms < sla_ms
    verdict = (
        f"✅  PASS  —  p95 {e2e.p95_ms:.1f} ms  <  SLA {sla_ms:.0f} ms"
        if pass_
        else
        f"❌  FAIL  —  p95 {e2e.p95_ms:.1f} ms  >=  SLA {sla_ms:.0f} ms"
    )
    print(_SEP)
    print(f"\n  SLA target : p95 < {sla_ms:.0f} ms")
    print(f"  Verdict    : {verdict}")
    print(_SEP)
    return pass_


def print_comparison_table(
    all_reports: Dict[str, BenchmarkReport],
    sla_ms:      float = 200.0,
) -> None:
    """Render a compact multi-config comparison table (end-to-end only)."""
    print(f"\n{'─'*60}")
    print("  Multi-Config Comparison (END-TO-END)")
    print(f"{'─'*60}")
    print(f"  {'Config':<12} {'Mean':>9}  {'p95':>9}  {'p99':>9}  SLA")
    print(f"{'─'*60}")
    for cfg_name, report in all_reports.items():
        s    = report["END-TO-END"]
        flag = "✅" if s.p95_ms < sla_ms else "❌"
        print(
            f"  {cfg_name:<12}"
            f"  {s.mean_ms:>7.1f}ms"
            f"  {s.p95_ms:>7.1f}ms"
            f"  {s.p99_ms:>7.1f}ms"
            f"  {flag}"
        )
    print(f"{'─'*60}\n")


# ════════════════════════════════════════════════════════════════════════════
# 6.  JSON EXPORT
# ════════════════════════════════════════════════════════════════════════════

def export_json(
    all_reports: Dict[str, BenchmarkReport],
    sys_info:    dict,
    path:        str,
    sla_ms:      float,
) -> None:
    """Serialise all benchmark results to a JSON file for CI artefact storage."""
    payload = {
        "meta": {
            "sla_ms":    sla_ms,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "system":    sys_info,
        },
        "results": {
            cfg: {name: s.as_dict() for name, s in rep.items()}
            for cfg, rep in all_reports.items()
        },
    }
    Path(path).write_text(json.dumps(payload, indent=2))
    print(f"\n  JSON report saved → {path}")


# ════════════════════════════════════════════════════════════════════════════
# 7.  CONFIG PRESETS
# ════════════════════════════════════════════════════════════════════════════

PRESETS: Dict[str, ScannerConfig] = {
    "default": ScannerConfig(
        # Balanced quality / speed for typical smartphone photos
        bilateral_d=9,
        bilateral_sigma_color=75.0,
        bilateral_sigma_space=75.0,
        canny_auto=True,
        clahe_clip_limit=2.0,
        clahe_tile_grid_size=(8, 8),
        output_width=1240,
        output_height=1754,
        adaptive_threshold_fallback=True,
        adaptive_block_size=21,
        adaptive_c=10,
    ),
    "strict": ScannerConfig(
        # Maximum quality — tighter filtering, finer CLAHE grid
        # Best for archival scans; slower bilateral pass
        bilateral_d=11,
        bilateral_sigma_color=90.0,
        bilateral_sigma_space=90.0,
        canny_auto=True,
        clahe_clip_limit=1.5,
        clahe_tile_grid_size=(16, 16),
        output_width=1240,
        output_height=1754,
        adaptive_threshold_fallback=True,
        adaptive_block_size=25,
        adaptive_c=12,
    ),
    "fast": ScannerConfig(
        # Optimised for speed — smaller bilateral kernel, half-res output
        # Suitable for live preview / mobile back-ends
        bilateral_d=5,
        bilateral_sigma_color=40.0,
        bilateral_sigma_space=40.0,
        canny_auto=True,
        clahe_clip_limit=2.5,
        clahe_tile_grid_size=(4, 4),
        output_width=620,
        output_height=877,
        adaptive_threshold_fallback=True,
        adaptive_block_size=15,
        adaptive_c=8,
    ),
}


# ════════════════════════════════════════════════════════════════════════════
# 8.  CLI
# ════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="benchmark.py",
        description="DocumentScanner MLOps latency benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--n", type=int, default=100, metavar="ITER",
        help="Number of measured iterations (after warm-up)",
    )
    p.add_argument(
        "--warmup", type=int, default=5, metavar="N",
        help="Discarded warm-up iterations",
    )
    p.add_argument(
        "--size", type=int, default=1080, metavar="PX",
        help="Synthetic image height in pixels (width = height * 0.75)",
    )
    p.add_argument(
        "--config", type=str, default="default",
        choices=list(PRESETS.keys()),
        help="Scanner configuration preset to benchmark",
    )
    p.add_argument(
        "--compare", action="store_true",
        help="Benchmark ALL presets and print a comparison table",
    )
    p.add_argument(
        "--image", type=str, default=None, metavar="PATH",
        help="Path to a real image file (overrides synthetic generator)",
    )
    p.add_argument(
        "--sla", type=float, default=200.0, metavar="MS",
        help="p95 end-to-end SLA target in milliseconds",
    )
    p.add_argument(
        "--export", type=str, default=None, metavar="PATH",
        help="Export results as JSON to this file path",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress progress bar (useful in CI logs)",
    )
    return p


def load_image(path: Optional[str], height: int) -> np.ndarray:
    """Load a real image or generate a synthetic one."""
    if path:
        img = cv2.imread(path)
        if img is None:
            print(f"ERROR: cannot read image at '{path}'", file=sys.stderr)
            sys.exit(1)
        print(f"  Using real image: {path}  ({img.shape[1]}×{img.shape[0]} px)")
        return img
    width = int(height * 0.75)
    print(f"  Using synthetic image: {width}×{height} px")
    return make_synthetic_document(height=height, width=width)


def main() -> None:
    args    = build_parser().parse_args()
    verbose = not args.quiet

    sys_info = collect_system_info()
    print_system_info(sys_info)

    image = load_image(args.image, args.size)

    # ── Which configs to run ─────────────────────────────────────────────
    configs_to_run: Dict[str, ScannerConfig] = (
        PRESETS if args.compare else {args.config: PRESETS[args.config]}
    )

    all_reports: Dict[str, BenchmarkReport] = {}
    all_pass = True

    for cfg_name, cfg in configs_to_run.items():
        scanner = DocumentScanner(cfg)
        report  = run_benchmark(
            scanner,
            image,
            n_iterations=args.n,
            warmup=args.warmup,
            verbose=verbose,
        )
        all_reports[cfg_name] = report
        passed = print_report(report, config_name=cfg_name, sla_ms=args.sla)
        if not passed:
            all_pass = False

    # ── Comparison table (only shown when --compare) ─────────────────────
    if args.compare:
        print_comparison_table(all_reports, sla_ms=args.sla)

    # ── JSON export ───────────────────────────────────────────────────────
    if args.export:
        export_json(all_reports, sys_info, args.export, args.sla)

    # ── CI exit code ──────────────────────────────────────────────────────
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
