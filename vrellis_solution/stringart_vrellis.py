"""
Жадный string art в духе Petros Vrellis / grvlbit/stringart.
Алгоритм: гвозди по окружности, на каждом шаге — хорда с максимальной
«темнотой» по целевому изображению, вычитание с остатка.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from skimage.draw import line as draw_line


def load_grayscale_square(path: str | Path) -> np.ndarray:
    image = Image.open(path)
    if image.mode != "L":
        raise SystemExit("Нужно одноканальное изображение в оттенках серого: режим L.")
    if image.width != image.height:
        raise SystemExit("Нужно квадратное изображение.")
    return np.asarray(image, dtype=float) / 255.0


def preprocess_portrait(image: np.ndarray) -> np.ndarray:
    """Как в grvlbit/stringart: инверсия, контраст, усиление границ."""
    pil = Image.fromarray((image * 255).astype(np.uint8))
    pil = ImageOps.grayscale(pil)
    pil = ImageOps.invert(pil)
    pil = pil.filter(ImageFilter.EDGE_ENHANCE_MORE)
    pil = ImageEnhance.Contrast(pil).enhance(1.4)
    data = np.asarray(pil, dtype=float) / 255.0
    return np.clip(data, 0.0, 1.0)


def circle_mask(size: int) -> np.ndarray:
    center = (size - 1) / 2
    y, x = np.ogrid[:size, :size]
    return (x - center) ** 2 + (y - center) ** 2 <= center ** 2


def nail_positions_circle(n_nails: int, size: int) -> list[tuple[float, float]]:
    center = (size - 1) / 2
    radius = center
    thetas = np.linspace(0.0, 2.0 * np.pi, n_nails, endpoint=False)
    nails = []
    for theta in thetas:
        col = center + radius * np.cos(theta)
        row = center + radius * np.sin(theta)
        nails.append((col, row))
    return nails


def bresenham_pixels(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    size: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = draw_line(
        int(round(y0)),
        int(round(x0)),
        int(round(y1)),
        int(round(x1)),
    )
    valid = (rows >= 0) & (rows < size) & (cols >= 0) & (cols < size)
    return rows[valid], cols[valid]


def build_path_cache(nails: list[tuple[float, float]], size: int, min_gap: int):
    n = len(nails)
    cache: list[list[tuple[np.ndarray, np.ndarray] | None]] = [
        [None] * n for _ in range(n)
    ]

    for i in range(n):
        for gap in range(min_gap, n // 2):
            j = (i + gap) % n
            x0, y0 = nails[i]
            x1, y1 = nails[j]
            rows, cols = bresenham_pixels(x0, y0, x1, y1, size)
            if rows.size > 0:
                cache[i][j] = (rows, cols)

    return cache


def choose_darkest_path(
    residual: np.ndarray,
    start_nail: int,
    path_cache,
    min_gap: int,
) -> tuple[int, np.ndarray, float]:
    n = len(path_cache)
    best_nail = start_nail
    best_mask = None
    best_score = -1.0

    for j in range(min_gap, n // 2):
        target = (start_nail + j) % n
        entry = path_cache[start_nail][target]
        if entry is None:
            continue

        rows, cols = entry
        score = float(residual[rows, cols].sum())

        if score > best_score:
            best_score = score
            best_nail = target
            mask = np.zeros_like(residual)
            mask[rows, cols] = 1.0
            best_mask = mask

    if best_mask is None:
        return start_nail, np.zeros_like(residual), 0.0

    return best_nail, best_mask, best_score


def generate_string_art(
    target: np.ndarray,
    n_nails: int = 240,
    max_iterations: int = 5000,
    line_weight: float = 0.08,
    min_nail_gap: int = 12,
    seed_nail: int = 0,
) -> tuple[list[tuple[int, int]], np.ndarray]:
    size = target.shape[0]
    target = target * circle_mask(size)
    residual = target.copy()
    nails = nail_positions_circle(n_nails, size)
    path_cache = build_path_cache(nails, size, min_nail_gap)

    segments: list[tuple[int, int]] = []
    current = seed_nail % n_nails

    for _ in range(max_iterations):
        next_nail, path_mask, score = choose_darkest_path(
            residual,
            current,
            path_cache,
            min_nail_gap,
        )

        if score <= 1e-6:
            break

        segments.append((current, next_nail))
        residual = np.clip(residual - line_weight * path_mask, 0.0, 1.0)
        current = next_nail

        if residual.sum() <= 1e-3:
            break

    return segments, residual


def draw_segments(
    segments: list[tuple[int, int]],
    nails: list[tuple[float, float]],
    size: int,
    render_size: int | None = None,
    line_weight: float = 1.0,
) -> np.ndarray:
    render_size = render_size or size
    scale = render_size / size
    canvas = np.zeros((render_size, render_size), dtype=float)
    center = (size - 1) / 2
    center_r = (render_size - 1) / 2

    for i, j in segments:
        x0, y0 = nails[i]
        x1, y1 = nails[j]
        rows, cols = bresenham_pixels(x0, y0, x1, y1, size)
        rr = np.rint(center_r + (rows - center) * scale).astype(int)
        cc = np.rint(center_r + (cols - center) * scale).astype(int)
        valid = (
            (rr >= 0)
            & (rr < render_size)
            & (cc >= 0)
            & (cc < render_size)
        )
        canvas[rr[valid], cc[valid]] += line_weight

    canvas *= circle_mask(render_size)
    return canvas


def save_preview_png(
    segments,
    nails,
    size: int,
    path: Path,
    preview_scale: float = 3.0,
    display_gain: float = 4.0,
):
    render_size = int(round(size * preview_scale))
    canvas = draw_segments(
        segments,
        nails,
        size,
        render_size=render_size,
        line_weight=1.0,
    )
    canvas = np.clip(canvas * display_gain, 0.0, None)
    positive = canvas[canvas > 0]
    limit = np.quantile(positive, 0.99) if positive.size else canvas.max()
    image = np.clip(canvas, 0, limit if limit > 0 else 1)
    if image.max() > 0:
        image /= image.max()
    Image.fromarray((image * 255).astype(np.uint8)).save(path)


def save_preview_svg(
    segments,
    nails,
    size: int,
    path: Path,
    preview_scale: float = 3.0,
    stroke_opacity: float = 0.45,
):
    render_size = int(round(size * preview_scale))
    scale = render_size / size
    center = (size - 1) / 2
    center_r = (render_size - 1) / 2

    lines_svg = []
    for i, j in segments:
        x0 = center_r + (nails[i][0] - center) * scale
        y0 = center_r + (nails[i][1] - center) * scale
        x1 = center_r + (nails[j][0] - center) * scale
        y1 = center_r + (nails[j][1] - center) * scale
        lines_svg.append(
            f'  <line x1="{x0:.2f}" y1="{y0:.2f}" x2="{x1:.2f}" y2="{y1:.2f}" '
            f'stroke="#ffffff" stroke-width="0.35" stroke-opacity="{stroke_opacity}" '
            f'stroke-linecap="round"/>'
        )

    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{render_size}" height="{render_size}" '
        f'viewBox="0 0 {render_size} {render_size}">\n'
        f'  <rect width="100%" height="100%" fill="#000000"/>\n'
        f'  <clipPath id="clip"><circle cx="{center_r:.2f}" cy="{center_r:.2f}" r="{center_r:.2f}"/></clipPath>\n'
        f'  <g clip-path="url(#clip)">\n'
        + "\n".join(lines_svg)
        + "\n  </g>\n</svg>\n"
    )
    Path(path).write_text(svg, encoding="utf-8")


def save_schema_csv(segments: list[tuple[int, int]], path: Path):
    import csv

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["nail_a", "nail_b"])
        writer.writerows(segments)
