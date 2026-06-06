"""
scanner.py — Automated Document Scanner & Enhancer
====================================================
Senior CV/MLOps implementation for the 12-Week Portfolio Sprint (Week 1).

Pipeline stages
---------------
1. preprocess_image      — grayscale → bilateral filter → Canny edges
2. find_document_contours — contour extraction → 4-sided polygon detection
3. order_corner_points   — TL / TR / BR / BL ordering for homography
4. apply_perspective_warp — getPerspectiveTransform + warpPerspective
5. enhance_contrast      — CLAHE on luminance channel → final monochrome

Design choices
--------------
- Bilateral filter instead of Gaussian: preserves edge sharpness while
  smoothing in-region noise. Controlled by `sigma_color`/`sigma_space`.
- Otsu-derived automatic Canny thresholds: uses the Otsu threshold on a
  blurred grayscale as the upper bound; lower = 0.5 * upper (Canny's
  recommended 1:2 ratio). Eliminates magic-number tuning per image.
- CLAHE on the L-channel of LAB (not raw grayscale): avoids hue shift
  artefacts when the caller wants a colour-preserving path; when the
  output is monochrome the L-channel IS the final image.
- All public methods return (result, latency_seconds) tuples so the
  caller can do per-stage MLOps benchmarking without instrumentation
  wrappers.
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass, field
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------

@dataclass
class ScannerConfig:
    """
    Centralised, serialisable configuration for every tunable parameter in
    the pipeline.  Pass a custom instance to DocumentScanner to override
    defaults without subclassing.
    """

    # --- pre-processing ---
    bilateral_d: int = 9
    """Diameter of each pixel neighbourhood used during bilateral filtering."""
    bilateral_sigma_color: float = 75.0
    """Filter sigma in the colour space (higher → more colour mixing)."""
    bilateral_sigma_space: float = 75.0
    """Filter sigma in coordinate space (higher → farther pixels mix)."""

    # --- edge detection ---
    canny_auto: bool = True
    """
    When True, derive Canny thresholds automatically via Otsu's method on a
    Gaussian-blurred grayscale (robust across exposure levels).
    When False, `canny_low` / `canny_high` are used directly.
    """
    canny_low: float = 50.0
    canny_high: float = 150.0
    canny_aperture: int = 3
    """Sobel aperture size for Canny (3, 5, or 7)."""

    # --- contour / polygon detection ---
    contour_min_area_ratio: float = 0.05
    """
    Minimum contour area as a fraction of the full image area.  Filters out
    tiny noise polygons before the 4-sided check.
    """
    max_contour_candidates: int = 10
    """How many largest contours to inspect before giving up."""
    poly_epsilon_ratio: float = 0.02
    """
    Douglas-Peucker epsilon as a fraction of the contour's arc-length.
    Controls how aggressively curves are approximated to straight lines.
    """

    # --- perspective warp ---
    output_width: int = 1240
    """Output document width in pixels (≈ A4 at 150 dpi)."""
    output_height: int = 1754
    """Output document height in pixels (≈ A4 at 150 dpi)."""

    # --- CLAHE enhancement ---
    clahe_clip_limit: float = 2.0
    """
    Threshold for contrast limiting.  Higher values → more aggressive
    enhancement but risk noise amplification.
    """
    clahe_tile_grid_size: Tuple[int, int] = (8, 8)
    """
    Number of grid tiles in x and y.  Finer grids improve local adaptation
    but increase computation.
    """

    # --- output ---
    adaptive_threshold_fallback: bool = True
    """
    When True, apply adaptive Gaussian thresholding after CLAHE to produce
    a clean binary document.  Set False to keep greyscale output.
    """
    adaptive_block_size: int = 21
    """Block size for adaptive thresholding (must be odd)."""
    adaptive_c: int = 10
    """Constant subtracted from the mean in adaptive thresholding."""


# ---------------------------------------------------------------------------
# Typed result containers
# ---------------------------------------------------------------------------

@dataclass
class ContourResult:
    """Result of document contour detection."""
    corners: Optional[np.ndarray]
    """4×2 array of (x, y) corner coordinates, or None if not found."""
    found: bool
    all_candidates: list = field(default_factory=list)
    """All 4-sided polygons inspected, largest-first (useful for debugging)."""


@dataclass
class StageMetrics:
    """Latency metrics for a single pipeline stage."""
    stage_name: str
    latency_s: float

    @property
    def latency_ms(self) -> float:
        return self.latency_s * 1_000

    def __str__(self) -> str:
        return f"[{self.stage_name}] {self.latency_ms:.2f} ms"


@dataclass
class ScanResult:
    """
    Full output of DocumentScanner.scan().
    Contains the final image plus per-stage latency breakdowns.
    """
    original: np.ndarray
    preprocessed: np.ndarray
    warped: Optional[np.ndarray]
    enhanced: np.ndarray
    corners: Optional[np.ndarray]
    document_found: bool
    metrics: list[StageMetrics]

    @property
    def total_latency_ms(self) -> float:
        return sum(m.latency_ms for m in self.metrics)

    def print_metrics(self) -> None:
        print("\n── Pipeline Latency Report ──────────────────────────")
        for m in self.metrics:
            print(f"  {m}")
        print(f"  [TOTAL]      {self.total_latency_ms:.2f} ms")
        print("─────────────────────────────────────────────────────\n")


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class DocumentScanner:
    """
    Production-grade, fully modular document scanner.

    Usage
    -----
    >>> scanner = DocumentScanner()
    >>> result  = scanner.scan(image_bgr)
    >>> cv2.imwrite("out.png", result.enhanced)
    >>> result.print_metrics()

    Each public stage method returns ``(output_image, latency_seconds)`` so
    you can unit-test or benchmark stages independently.
    """

    def __init__(self, config: Optional[ScannerConfig] = None) -> None:
        self.cfg = config or ScannerConfig()
        self._clahe = cv2.createCLAHE(
            clipLimit=self.cfg.clahe_clip_limit,
            tileGridSize=self.cfg.clahe_tile_grid_size,
        )

    # ------------------------------------------------------------------
    # Stage 1 — Preprocessing
    # ------------------------------------------------------------------

    def preprocess_image(
        self, image_bgr: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Convert to grayscale using photometric-correct luminance weights,
        apply bilateral filtering, then run Canny edge detection.

        Photometric weights (BT.601):
            Y = 0.299·R + 0.587·G + 0.114·B
        OpenCV's ``cvtColor(BGR2GRAY)`` applies these weights internally.
        We do NOT use a simple channel average because the human visual
        system is far more sensitive to green than to red or blue.

        Parameters
        ----------
        image_bgr : np.ndarray
            Input image in BGR uint8 format (as returned by cv2.imread).

        Returns
        -------
        edges : np.ndarray
            Binary edge map (uint8, values 0 or 255).
        latency_s : float
            Wall-clock time for this stage in seconds.
        """
        t0 = time.perf_counter()

        # Photometric grayscale (BT.601 weights via OpenCV)
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Bilateral filter — edge-preserving noise reduction.
        # Unlike Gaussian blur, bilateral weights neighbours by BOTH spatial
        # proximity AND intensity similarity, so edges stay sharp while
        # flat regions are smoothed.
        filtered = cv2.bilateralFilter(
            gray,
            d=self.cfg.bilateral_d,
            sigmaColor=self.cfg.bilateral_sigma_color,
            sigmaSpace=self.cfg.bilateral_sigma_space,
        )

        # Canny with automatic or manual thresholds
        low, high = self._compute_canny_thresholds(filtered)
        edges = cv2.Canny(
            filtered,
            threshold1=low,
            threshold2=high,
            apertureSize=self.cfg.canny_aperture,
            L2gradient=True,   # more accurate gradient magnitude
        )

        latency_s = time.perf_counter() - t0
        logger.debug("preprocess_image: %.2f ms", latency_s * 1000)
        return edges, latency_s

    def _compute_canny_thresholds(
        self, gray: np.ndarray
    ) -> Tuple[float, float]:
        """
        Derive Canny hysteresis thresholds.

        Auto mode (recommended): blur the image lightly, then run Otsu's
        global binarisation.  The Otsu threshold is a statistically optimal
        split between foreground and background intensity distributions.
        We use it as the *upper* Canny threshold and set the lower to half
        (the 1:2 ratio recommended in the original Canny paper).

        This makes the scanner robust to dim scans, high-contrast receipts,
        and over-exposed paper alike — without per-image tuning.
        """
        if not self.cfg.canny_auto:
            return self.cfg.canny_low, self.cfg.canny_high

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        otsu_high, _ = cv2.threshold(
            blurred, 0, 255,
            cv2.THRESH_BINARY | cv2.THRESH_OTSU
        )
        otsu_low = 0.5 * otsu_high
        logger.debug("Otsu thresholds → low=%.1f high=%.1f", otsu_low, otsu_high)
        return otsu_low, otsu_high

    # ------------------------------------------------------------------
    # Stage 2 — Contour / polygon detection
    # ------------------------------------------------------------------

    def find_document_contours(
        self, edges: np.ndarray, image_shape: Tuple[int, ...]
    ) -> Tuple[ContourResult, float]:
        """
        Extract contours from the edge map and find the largest 4-sided
        convex polygon — the document boundary.

        Algorithm
        ---------
        1. Find all external contours (RETR_EXTERNAL avoids nested noise).
        2. Sort by area descending — the document is almost always the
           largest connected region in the frame.
        3. For each of the top-N candidates, approximate with
           Douglas-Peucker and check for exactly 4 vertices.
        4. Return the first (largest) 4-sided match.

        Parameters
        ----------
        edges : np.ndarray
            Binary edge map from ``preprocess_image``.
        image_shape : tuple
            Shape of the *original* image (H, W[, C]) used for area gating.

        Returns
        -------
        ContourResult
            ``found=True`` and ``corners`` set if a document quad was found.
        latency_s : float
        """
        t0 = time.perf_counter()

        h, w = image_shape[:2]
        image_area = h * w
        min_area = self.cfg.contour_min_area_ratio * image_area

        contours, _ = cv2.findContours(
            edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Sort largest first
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        candidates = []
        result_corners: Optional[np.ndarray] = None

        for contour in contours[: self.cfg.max_contour_candidates]:
            area = cv2.contourArea(contour)
            if area < min_area:
                break  # remaining contours are even smaller

            arc_len = cv2.arcLength(contour, closed=True)
            epsilon = self.cfg.poly_epsilon_ratio * arc_len
            poly = cv2.approxPolyDP(contour, epsilon, closed=True)

            if len(poly) == 4 and cv2.isContourConvex(poly):
                corners = poly.reshape(4, 2).astype(np.float32)
                candidates.append(corners)
                if result_corners is None:
                    result_corners = corners

        found = result_corners is not None
        if not found:
            logger.warning(
                "No 4-sided convex polygon found; "
                "will fall back to full-image bounding box."
            )

        latency_s = time.perf_counter() - t0
        return (
            ContourResult(corners=result_corners, found=found, all_candidates=candidates),
            latency_s,
        )

    # ------------------------------------------------------------------
    # Stage 3 — Corner ordering
    # ------------------------------------------------------------------

    @staticmethod
    def order_corner_points(pts: np.ndarray) -> np.ndarray:
        """
        Return the 4 corner points in a canonical (TL, TR, BR, BL) order
        required by ``cv2.getPerspectiveTransform``.

        Method
        ------
        - Top-Left     → smallest  (x + y)  sum
        - Bottom-Right → largest   (x + y)  sum
        - Top-Right    → smallest  (y - x)  difference
        - Bottom-Left  → largest   (y - x)  difference

        This is robust to arbitrary rotations because the sum/difference
        criteria are invariant to axis-aligned reflections that swap
        "left" and "right" on a tilted image.

        Parameters
        ----------
        pts : np.ndarray
            Shape (4, 2) array of (x, y) corner coordinates.

        Returns
        -------
        np.ndarray
            Shape (4, 2) float32 array ordered [TL, TR, BR, BL].
        """
        rect = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).ravel()

        rect[0] = pts[np.argmin(s)]    # TL
        rect[2] = pts[np.argmax(s)]    # BR
        rect[1] = pts[np.argmin(diff)] # TR
        rect[3] = pts[np.argmax(diff)] # BL
        return rect

    # ------------------------------------------------------------------
    # Stage 4 — Perspective warp
    # ------------------------------------------------------------------

    def apply_perspective_warp(
        self,
        image_bgr: np.ndarray,
        corners: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, float]:
        """
        Compute a homography from the 4 detected corners to a rectangular
        output canvas and warp the image.

        If ``corners`` is None (document not detected), the original image
        is returned unchanged — the pipeline degrades gracefully.

        The output canvas dimensions are determined by
        ``ScannerConfig.output_width`` / ``output_height`` (default: A4 at
        150 dpi).  You can override these for letter-size or custom formats.

        Parameters
        ----------
        image_bgr : np.ndarray
            Original colour image.
        corners : np.ndarray or None
            4×2 float32 corner array from ``find_document_contours``.

        Returns
        -------
        warped : np.ndarray
            Perspective-corrected image (BGR uint8).
        latency_s : float
        """
        t0 = time.perf_counter()

        if corners is None:
            logger.info("No corners provided; skipping warp (identity path).")
            return image_bgr.copy(), time.perf_counter() - t0

        ordered = self.order_corner_points(corners)
        W, H = self.cfg.output_width, self.cfg.output_height

        dst = np.array(
            [[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]],
            dtype=np.float32,
        )

        M = cv2.getPerspectiveTransform(ordered, dst)
        warped = cv2.warpPerspective(
            image_bgr, M, (W, H),
            flags=cv2.INTER_LANCZOS4,   # Lanczos = best quality for upscaling
            borderMode=cv2.BORDER_REPLICATE,
        )

        latency_s = time.perf_counter() - t0
        logger.debug("apply_perspective_warp: %.2f ms", latency_s * 1000)
        return warped, latency_s

    # ------------------------------------------------------------------
    # Stage 5 — Contrast enhancement (CLAHE + optional binarisation)
    # ------------------------------------------------------------------

    def enhance_contrast(
        self, image_bgr: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        Enhance contrast using CLAHE on the L-channel of CIE L*a*b* space,
        then optionally binarise with adaptive Gaussian thresholding to
        produce a print-ready monochrome document.

        Why L*a*b*?
        -----------
        CLAHE on raw grayscale is fine for monochrome output, but applying
        it to the perceptual-lightness L-channel in L*a*b* is more correct:
        it equalises only perceived brightness, not raw pixel intensity,
        which matches how the human visual system perceives contrast.

        Why adaptive thresholding (not global)?
        ----------------------------------------
        Documents often have gradual brightness gradients (edge shadows,
        paper curl, uneven phone lighting).  Adaptive Gaussian thresholding
        computes a local mean in each ``block_size × block_size`` window and
        thresholds each pixel relative to its neighbourhood — making text
        legible even under very uneven illumination.

        Parameters
        ----------
        image_bgr : np.ndarray
            Perspective-corrected image (BGR uint8).

        Returns
        -------
        enhanced : np.ndarray
            Single-channel uint8 enhanced image.
        latency_s : float
        """
        t0 = time.perf_counter()

        # Convert to L*a*b* and apply CLAHE to L-channel
        lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_eq = self._clahe.apply(l_ch)

        # Merge back and convert to grayscale for monochrome output
        lab_eq = cv2.merge([l_eq, a_ch, b_ch])
        enhanced_bgr = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
        gray = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2GRAY)

        if self.cfg.adaptive_threshold_fallback:
            block = self.cfg.adaptive_block_size
            # Ensure block size is odd (requirement of OpenCV)
            if block % 2 == 0:
                block += 1
            enhanced = cv2.adaptiveThreshold(
                gray,
                maxValue=255,
                adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                thresholdType=cv2.THRESH_BINARY,
                blockSize=block,
                C=self.cfg.adaptive_c,
            )
        else:
            enhanced = gray

        latency_s = time.perf_counter() - t0
        logger.debug("enhance_contrast: %.2f ms", latency_s * 1000)
        return enhanced, latency_s

    # ------------------------------------------------------------------
    # High-level convenience method
    # ------------------------------------------------------------------

    def scan(self, image_bgr: np.ndarray) -> ScanResult:
        """
        Run the complete pipeline end-to-end and return a ``ScanResult``
        containing all intermediate artefacts and per-stage latencies.

        Parameters
        ----------
        image_bgr : np.ndarray
            Raw camera image in BGR uint8 (as read by ``cv2.imread``).

        Returns
        -------
        ScanResult
        """
        metrics: list[StageMetrics] = []

        # Stage 1 — preprocess
        edges, lat = self.preprocess_image(image_bgr)
        metrics.append(StageMetrics("Preprocessing / Edge Detection", lat))

        # Stage 2 — contour detection
        contour_result, lat = self.find_document_contours(edges, image_bgr.shape)
        metrics.append(StageMetrics("Contour Detection", lat))

        # Stage 3+4 — warp (corner ordering is internal to warp stage)
        warped, lat = self.apply_perspective_warp(image_bgr, contour_result.corners)
        metrics.append(StageMetrics("Perspective Warp", lat))

        # Stage 5 — enhance
        enhanced, lat = self.enhance_contrast(warped)
        metrics.append(StageMetrics("CLAHE + Enhancement", lat))

        return ScanResult(
            original=image_bgr,
            preprocessed=edges,
            warped=warped,
            enhanced=enhanced,
            corners=contour_result.corners,
            document_found=contour_result.found,
            metrics=metrics,
        )

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    def draw_debug_overlay(
        self, image_bgr: np.ndarray, corners: Optional[np.ndarray]
    ) -> np.ndarray:
        """
        Draw the detected document polygon on a copy of the image for
        visual debugging.  Returns the annotated image (BGR).
        """
        out = image_bgr.copy()
        if corners is not None:
            ordered = self.order_corner_points(corners)
            pts = ordered.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(out, [pts], isClosed=True, color=(0, 255, 0), thickness=3)
            labels = ["TL", "TR", "BR", "BL"]
            colours = [(255, 0, 0), (0, 200, 255), (0, 0, 255), (255, 100, 0)]
            for (x, y), label, colour in zip(ordered, labels, colours):
                cv2.circle(out, (int(x), int(y)), 8, colour, -1)
                cv2.putText(
                    out, label, (int(x) + 10, int(y) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, colour, 2,
                )
        return out
