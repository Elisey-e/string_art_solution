from pathlib import Path
import csv

import numpy as np
from PIL import Image


def serpentine_dither_1d(projection, seed=None):
    """Serpentine dithering for 1D signal (e.g., sinogram row)."""
    rng = np.random.default_rng(seed)
    output = np.zeros_like(projection, dtype=np.uint8)

    forward = True
    for i in range(len(projection)):
        idx = i if forward else len(projection) - 1 - i
        value = projection[idx]

        random = rng.random()
        if random < value:
            output[idx] = 1

        forward = not forward

    return output


def sinogram_error_diffusion(sinogram, seed=None):
    """
    Apply simple error diffusion to sinogram.
    Propagates error to the next offset and next angle.
    """
    rng = np.random.default_rng(seed)
    output = np.zeros_like(sinogram, dtype=np.uint8)
    sinogram_copy = sinogram.copy()

    for angle_idx in range(sinogram.shape[0]):
        forward = angle_idx % 2 == 0

        indices = (
            range(sinogram.shape[1])
            if forward
            else range(sinogram.shape[1] - 1, -1, -1)
        )

        for offset_idx in indices:
            val = sinogram_copy[angle_idx, offset_idx]

            if rng.random() < val:
                output[angle_idx, offset_idx] = 1
                error = val - 1.0
            else:
                output[angle_idx, offset_idx] = 0
                error = val

            if offset_idx < sinogram.shape[1] - 1:
                sinogram_copy[angle_idx, offset_idx + 1] += error * 0.5

            if angle_idx < sinogram.shape[0] - 1:
                sinogram_copy[angle_idx + 1, offset_idx] += error * 0.5

    return output


def load_grayscale_square(path):
    image = Image.open(path)

    if image.mode != "L":
        raise SystemExit("Нужно одноканальное изображение в оттенках серого: режим L.")

    if image.width != image.height:
        raise SystemExit("Нужно квадратное изображение.")

    return np.asarray(image, dtype=float) / 255.0


def circle_mask(size):
    center = (size - 1) / 2
    y, x = np.ogrid[:size, :size]
    return (x - center) ** 2 + (y - center) ** 2 <= center ** 2


def bilinear_sample(image, rows, cols):
    size = image.shape[0]

    row0 = np.floor(rows).astype(int)
    col0 = np.floor(cols).astype(int)

    valid = (0 <= row0) & (row0 < size - 1) & (0 <= col0) & (col0 < size - 1)

    row0 = np.clip(row0, 0, size - 2)
    col0 = np.clip(col0, 0, size - 2)
    row1 = row0 + 1
    col1 = col0 + 1

    dr = rows - row0
    dc = cols - col0

    values = (
        image[row0, col0] * (1 - dr) * (1 - dc)
        + image[row0, col1] * (1 - dr) * dc
        + image[row1, col0] * dr * (1 - dc)
        + image[row1, col1] * dr * dc
    )
    values[~valid] = 0.0
    return values


def radon_transform(image, angle_count):
    size = image.shape[0]
    center = (size - 1) / 2
    image = image * circle_mask(size)

    offsets = np.arange(size) - center
    line_points = np.linspace(-center, center, size)
    angles = np.linspace(0.0, 180.0, angle_count, endpoint=False)
    sinogram = np.zeros((angle_count, size), dtype=float)

    for angle_index, angle in enumerate(angles):
        phi = np.deg2rad(angle)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)

        x = offsets[:, None] * cos_phi - line_points[None, :] * sin_phi
        y = offsets[:, None] * sin_phi + line_points[None, :] * cos_phi

        rows = center + y
        cols = center + x
        sinogram[angle_index] = bilinear_sample(image, rows, cols).sum(axis=1)

    return sinogram, angles


def ramp_kernel(size):
    offsets = np.arange(-(size // 2), size - size // 2)
    kernel = np.zeros(size, dtype=float)

    kernel[offsets == 0] = 1 / 4
    odd = (offsets % 2 != 0)
    kernel[odd] = -1 / (np.pi * offsets[odd]) ** 2

    return kernel


def filter_sinogram(sinogram):
    kernel = ramp_kernel(sinogram.shape[1])
    filtered = np.zeros_like(sinogram)

    for angle_index, projection in enumerate(sinogram):
        filtered[angle_index] = np.convolve(projection, kernel, mode="same")

    return filtered


def keep_brightest_per_angle(sinogram, thread_count):
    kept = np.zeros_like(sinogram)
    thread_count = min(thread_count, sinogram.shape[1])

    for angle_index, projection in enumerate(sinogram):
        indices = np.argpartition(projection, -thread_count)[-thread_count:]
        kept[angle_index, indices] = projection[indices]

    return kept


def normalize_nonnegative(sinogram):
    normalized = sinogram.copy()
    normalized[normalized < 0] = 0

    maximum = normalized.max()
    if maximum > 0:
        normalized /= maximum

    return normalized


def binary_dither(probabilities, seed):
    random = np.random.default_rng(seed).random(probabilities.shape)
    return (random < probabilities).astype(np.uint8)


def binary_sinogram_to_lines(binary_sinogram, angles):
    center = (binary_sinogram.shape[1] - 1) / 2
    lines = []

    for angle_index, offset_index in np.argwhere(binary_sinogram > 0):
        angle = float(angles[angle_index])
        offset = float(offset_index - center)
        lines.append((angle, offset))

    return lines


def save_schema(lines, path):
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["angle_deg", "rho_px"])
        writer.writerows(lines)


def read_schema(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return [(float(row["angle_deg"]), float(row["rho_px"])) for row in reader]


def save_float_image(array, path):
    image = array.astype(float).copy()
    image -= image.min()

    maximum = image.max()
    if maximum > 0:
        image /= maximum

    Image.fromarray((image * 255).astype(np.uint8)).save(path)


def _render_scale(size, render_size):
    if render_size is None or render_size == size:
        return size, 1.0
    return render_size, render_size / size


def hough_line_segment(size, angle_deg, rho_px, render_size=None):
    render_size, scale = _render_scale(size, render_size)
    center = (size - 1) / 2
    radius = center

    if abs(rho_px) > radius:
        return None

    phi = np.deg2rad(angle_deg)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    half_length = np.sqrt(radius ** 2 - rho_px ** 2)
    t0, t1 = -half_length, half_length
    x0 = rho_px * cos_phi - t0 * sin_phi
    y0 = rho_px * sin_phi + t0 * cos_phi
    x1 = rho_px * cos_phi - t1 * sin_phi
    y1 = rho_px * sin_phi + t1 * cos_phi

    center_r = (render_size - 1) / 2
    return (
        center_r + x0 * scale,
        center_r + y0 * scale,
        center_r + x1 * scale,
        center_r + y1 * scale,
    )


def iter_thread_segments(lines, size, nail_count=0, render_size=None):
    render_size, scale = _render_scale(size, render_size)

    if nail_count > 0:
        rows, cols = nail_coordinates(nail_count, size)
        for entry in lines:
            if len(entry) >= 4 and entry[2] is not None:
                nail_a, nail_b = int(entry[2]), int(entry[3])
            else:
                continue
            center_r = (render_size - 1) / 2
            center = (size - 1) / 2
            yield (
                center_r + (cols[nail_a] - center) * scale,
                center_r + (rows[nail_a] - center) * scale,
                center_r + (cols[nail_b] - center) * scale,
                center_r + (rows[nail_b] - center) * scale,
            )
        return

    for entry in lines:
        angle = entry[0] if isinstance(entry, (tuple, list)) else entry
        rho = entry[1] if isinstance(entry, (tuple, list)) else 0.0
        segment = hough_line_segment(size, float(angle), float(rho), render_size)
        if segment is not None:
            yield segment


def draw_threads(lines, size, render_size=None):
    render_size, scale = _render_scale(size, render_size)
    canvas = np.zeros((render_size, render_size), dtype=float)
    center = (size - 1) / 2
    radius = center
    center_r = (render_size - 1) / 2
    sample_count = max(int(size * 2 * scale), size * 2)

    for entry in lines:
        angle = float(entry[0])
        offset = float(entry[1])
        if abs(offset) > radius:
            continue

        phi = np.deg2rad(angle)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        half_length = np.sqrt(radius ** 2 - offset ** 2)
        line_points = np.linspace(-half_length, half_length, sample_count)

        x = offset * cos_phi - line_points * sin_phi
        y = offset * sin_phi + line_points * cos_phi

        rows = np.rint(center_r + y * scale).astype(int)
        cols = np.rint(center_r + x * scale).astype(int)
        valid = (
            (0 <= rows)
            & (rows < render_size)
            & (0 <= cols)
            & (cols < render_size)
        )

        np.add.at(canvas, (rows[valid], cols[valid]), 1.0)

    mask = circle_mask(render_size)
    canvas *= mask
    return canvas


def contrast_threads(canvas):
    positive = canvas[canvas > 0]

    if positive.size == 0:
        return canvas

    limit = np.quantile(positive, 0.995)
    image = np.clip(canvas, 0, limit)
    image /= image.max()
    return image


def save_threads_image(lines, size, path):
    canvas = draw_threads(lines, size)
    image = contrast_threads(canvas)
    Image.fromarray((image * 255).astype(np.uint8)).save(path)


def save_threads_svg(
    lines,
    size,
    path,
    nail_count=0,
    render_size=None,
    stroke_width=0.4,
    stroke_opacity=0.35,
):
    render_size, _ = _render_scale(size, render_size)
    segments = list(iter_thread_segments(lines, size, nail_count, render_size))

    lines_svg = []
    for x1, y1, x2, y2 in segments:
        lines_svg.append(
            f'  <line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" '
            f'stroke="#ffffff" stroke-width="{stroke_width}" '
            f'stroke-opacity="{stroke_opacity}" stroke-linecap="round"/>'
        )

    center_r = (render_size - 1) / 2
    radius_r = center_r
    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{render_size}" height="{render_size}" '
        f'viewBox="0 0 {render_size} {render_size}">\n'
        f'  <rect width="100%" height="100%" fill="#000000"/>\n'
        f'  <clipPath id="circleClip">\n'
        f'    <circle cx="{center_r:.3f}" cy="{center_r:.3f}" r="{radius_r:.3f}"/>\n'
        f"  </clipPath>\n"
        f'  <g clip-path="url(#circleClip)">\n'
        + "\n".join(lines_svg)
        + "\n  </g>\n</svg>\n"
    )

    Path(path).write_text(svg, encoding="utf-8")


def _raster_preview_from_canvas(canvas):
    image = canvas.copy()
    limit = (
        np.quantile(image[image > 0], 0.995) if (image > 0).sum() > 0 else image.max()
    )
    image = np.clip(image, 0, limit)
    if image.max() > 0:
        image /= image.max()
    return image


def save_threads_image_black_background(lines, size, path, preview_scale=3):
    render_size = int(round(size * preview_scale))
    canvas = draw_threads(lines, size, render_size=render_size)
    image = _raster_preview_from_canvas(canvas)
    Image.fromarray((image * 255).astype(np.uint8)).save(path)


def save_preview_bundle(
    lines,
    size,
    output_dir,
    nail_count=0,
    preview_scale=3,
    png_name="preview.png",
    svg_name="preview.svg",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    render_size = int(round(size * preview_scale))

    save_threads_image_black_background(
        lines,
        size,
        output_dir / png_name,
        preview_scale=preview_scale,
    )
    save_threads_svg(
        lines,
        size,
        output_dir / svg_name,
        nail_count=nail_count,
        render_size=render_size,
    )
