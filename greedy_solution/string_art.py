from pathlib import Path
import csv

import numpy as np
from PIL import Image
from scipy.sparse import csr_matrix
from skimage.draw import line as draw_line


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


def prepare_target(image, gamma=0.75):
    """Светлый фон -> мало нитей, тёмные детали -> больше нитей."""
    size = image.shape[0]
    mask = circle_mask(size)
    target = (1.0 - image) * mask
    values = target[mask]

    if values.size == 0:
        return target

    target[mask] = (values - values.min()) / (values.max() - values.min() + 1e-9)
    target[mask] = target[mask] ** gamma
    return target


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


def build_candidates_from_sinogram(target, angle_count, threads_per_angle):
    sinogram, angles = radon_transform(target, angle_count)
    size = target.shape[0]
    center = (size - 1) / 2
    catalog = []
    seen = set()

    for angle_index, projection in enumerate(sinogram):
        count = min(threads_per_angle, projection.size)
        indices = np.argpartition(projection, -count)[-count:]

        for offset_index in indices:
            angle = float(angles[angle_index])
            rho = float(offset_index - center)
            key = (angle_index, offset_index)

            if key in seen:
                continue

            seen.add(key)
            pixels = line_pixel_indices(size, angle, rho)

            if pixels.size > 0:
                catalog.append((angle, rho, pixels, None, None))

    return catalog


def line_pixel_indices(size, angle_deg, rho):
    center = (size - 1) / 2
    radius = center

    if abs(rho) > radius:
        return np.array([], dtype=np.int32)

    phi = np.deg2rad(angle_deg)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    half_length = np.sqrt(radius ** 2 - rho ** 2)
    line_points = np.linspace(-half_length, half_length, size * 2)

    x = rho * cos_phi - line_points * sin_phi
    y = rho * sin_phi + line_points * cos_phi

    rows = np.rint(center + y).astype(np.int32)
    cols = np.rint(center + x).astype(np.int32)
    valid = (0 <= rows) & (rows < size) & (0 <= cols) & (cols < size)

    return np.unique(rows[valid] * size + cols[valid])


def nail_coordinates(n_nails, size):
    center = (size - 1) / 2
    radius = center
    thetas = np.linspace(0.0, 2.0 * np.pi, n_nails, endpoint=False)
    cols = center + radius * np.cos(thetas)
    rows = center + radius * np.sin(thetas)
    return rows, cols


def bresenham_pixels(row0, col0, row1, col1, size):
    rows, cols = draw_line(
        int(round(row0)),
        int(round(col0)),
        int(round(row1)),
        int(round(col1)),
    )
    valid = (rows >= 0) & (rows < size) & (cols >= 0) & (cols < size)
    return np.unique(rows[valid] * size + cols[valid])


def chord_to_angle_rho(y0, x0, y1, x1, center):
    dy = y1 - y0
    dx = x1 - x0
    length = np.hypot(dy, dx)

    if length < 1e-9:
        return 0.0, 0.0

    angle = float(np.rad2deg(np.arctan2(dy, dx)) % 180.0)
    normal_x = -dy / length
    normal_y = dx / length
    rho = float(normal_x * x0 + normal_y * y0)
    return angle, rho


def build_nail_catalog(n_nails, size, min_nail_gap):
    rows, cols = nail_coordinates(n_nails, size)
    center = (size - 1) / 2
    catalog = []

    for nail_a in range(n_nails):
        for gap in range(min_nail_gap, n_nails // 2):
            nail_b = (nail_a + gap) % n_nails
            pixels = bresenham_pixels(
                rows[nail_a],
                cols[nail_a],
                rows[nail_b],
                cols[nail_b],
                size,
            )

            if pixels.size == 0:
                continue

            angle, rho = chord_to_angle_rho(
                rows[nail_a],
                cols[nail_a],
                rows[nail_b],
                cols[nail_b],
                center,
            )
            catalog.append((angle, rho, pixels, nail_a, nail_b))

    return catalog


def build_line_catalog(size, angle_count):
    center = (size - 1) / 2
    angles = np.linspace(0.0, 180.0, angle_count, endpoint=False)
    rhos = np.arange(size, dtype=float) - center

    catalog = []
    for angle in angles:
        for rho in rhos:
            pixels = line_pixel_indices(size, float(angle), float(rho))
            if pixels.size > 0:
                catalog.append((float(angle), float(rho), pixels, None, None))

    return catalog


def build_line_matrix(catalog, pixel_count):
    row_indices = []
    col_indices = []

    for line_index, entry in enumerate(catalog):
        pixels = entry[2]
        row_indices.extend([line_index] * pixels.size)
        col_indices.extend(pixels.tolist())

    data = np.ones(len(col_indices), dtype=np.float32)
    return csr_matrix((data, (row_indices, col_indices)), shape=(len(catalog), pixel_count))


def greedy_string_art(
    target,
    angle_count=180,
    max_lines=4000,
    darkness=0.03,
    min_score=1e-4,
    threads_per_angle=40,
    use_full_catalog=False,
    progress_every=500,
    nail_count=0,
    min_nail_gap=12,
):
    size = target.shape[0]
    mask = circle_mask(size)
    target = (target * mask).astype(np.float32)

    if nail_count > 0:
        catalog = build_nail_catalog(nail_count, size, min_nail_gap)
    elif use_full_catalog:
        catalog = build_line_catalog(size, angle_count)
    else:
        catalog = build_candidates_from_sinogram(
            target,
            angle_count,
            threads_per_angle,
        )

    if not catalog:
        return []

    print(f"Кандидатов линий: {len(catalog)}", flush=True)
    line_matrix = build_line_matrix(catalog, size * size)
    line_lengths = np.array([entry[2].size for entry in catalog], dtype=np.float32)
    line_lengths = np.maximum(line_lengths, 1.0)

    canvas = np.zeros(size * size, dtype=np.float32)
    residual = target.reshape(-1)
    lines = []
    best_lines = []
    best_rmse = float("inf")
    stall_steps = 0
    patience = 400

    for step in range(max_lines):
        scores = (line_matrix @ residual) / line_lengths
        best_index = int(np.argmax(scores))
        best_score = float(scores[best_index])

        if best_score < min_score:
            print(f"Остановка на шаге {step}: вклад {best_score:.6f}", flush=True)
            break

        angle, rho, pixels, nail_a, nail_b = catalog[best_index]
        canvas[pixels] += darkness
        residual = np.clip(target.reshape(-1) - canvas, 0.0, 1.0)
        lines.append((angle, rho, nail_a, nail_b))

        rmse = float(np.sqrt(np.mean((canvas.reshape(size, size) - target) ** 2)))
        if rmse < best_rmse - 1e-5:
            best_rmse = rmse
            best_lines = list(lines)
            stall_steps = 0
        else:
            stall_steps += 1
            if stall_steps >= patience and best_lines:
                print(
                    f"Ранняя остановка на шаге {step}: RMSE не улучшался {patience} шагов",
                    flush=True,
                )
                lines = best_lines
                break

        if progress_every and step and step % progress_every == 0:
            print(f"  шаг {step}, нитей {len(lines)}, RMSE {rmse:.4f}", flush=True)

    return lines


def reconstruction_metrics(target, lines, size, darkness=0.07, nail_count=0):
    mask = circle_mask(size)
    canvas = draw_threads(
        lines,
        size,
        darkness=darkness,
        nail_count=nail_count,
    )
    inside = mask > 0

    error = canvas[inside] - target[inside]
    mse = float(np.mean(error ** 2))
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(mse))

    return {"mse": mse, "mae": mae, "rmse": rmse}


def save_schema(lines, path, use_nails=False):
    with Path(path).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if use_nails:
            writer.writerow(["nail_a", "nail_b", "angle_deg", "rho_px"])
            writer.writerows(
                (nail_a, nail_b, angle, rho)
                for angle, rho, nail_a, nail_b in lines
            )
        else:
            writer.writerow(["angle_deg", "rho_px"])
            writer.writerows((angle, rho) for angle, rho, _, _ in lines)


def read_schema(path):
    with Path(path).open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)

    if "nail_a" in reader.fieldnames:
        return [
            (
                float(row["angle_deg"]),
                float(row["rho_px"]),
                int(row["nail_a"]),
                int(row["nail_b"]),
            )
            for row in rows
        ]

    return [
        (float(row["angle_deg"]), float(row["rho_px"]), None, None)
        for row in rows
    ]


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
        center = (size - 1) / 2
        center_r = (render_size - 1) / 2
        for entry in lines:
            if len(entry) >= 4 and entry[2] is not None:
                nail_a, nail_b = int(entry[2]), int(entry[3])
            else:
                continue
            yield (
                center_r + (cols[nail_a] - center) * scale,
                center_r + (rows[nail_a] - center) * scale,
                center_r + (cols[nail_b] - center) * scale,
                center_r + (rows[nail_b] - center) * scale,
            )
        return

    for entry in lines:
        segment = hough_line_segment(size, float(entry[0]), float(entry[1]), render_size)
        if segment is not None:
            yield segment


def draw_threads(lines, size, darkness=1.0, nail_count=0, render_size=None):
    render_size, scale = _render_scale(size, render_size)
    canvas = np.zeros((render_size, render_size), dtype=float)
    flat = canvas.reshape(-1)

    if nail_count > 0:
        rows, cols = nail_coordinates(nail_count, size)
        for _, _, nail_a, nail_b in lines:
            pixels = bresenham_pixels(
                rows[nail_a],
                cols[nail_a],
                rows[nail_b],
                cols[nail_b],
                size,
            )
            if pixels.size > 0:
                src_rows = pixels // size
                src_cols = pixels % size
                center = (size - 1) / 2
                center_r = (render_size - 1) / 2
                dst_rows = np.rint(center_r + (src_rows - center) * scale).astype(int)
                dst_cols = np.rint(center_r + (src_cols - center) * scale).astype(int)
                valid = (
                    (0 <= dst_rows)
                    & (dst_rows < render_size)
                    & (0 <= dst_cols)
                    & (dst_cols < render_size)
                )
                dst = dst_rows[valid] * render_size + dst_cols[valid]
                flat[dst] += darkness
    else:
        center = (size - 1) / 2
        radius = center
        center_r = (render_size - 1) / 2
        sample_count = max(int(size * 2 * scale), size * 2)

        for angle, offset, _, _ in lines:
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
            flat[rows[valid] * render_size + cols[valid]] += darkness

    canvas = flat.reshape(render_size, render_size)
    mask = circle_mask(render_size)
    canvas *= mask
    return canvas


def save_threads_svg(
    lines,
    size,
    path,
    nail_count=0,
    render_size=None,
    stroke_width=0.35,
    stroke_opacity=0.25,
    dark_on_light=False,
):
    render_size, _ = _render_scale(size, render_size)
    segments = list(iter_thread_segments(lines, size, nail_count, render_size))
    stroke = "#111111" if dark_on_light else "#ffffff"
    background = "#ffffff" if dark_on_light else "#000000"

    lines_svg = []
    for x1, y1, x2, y2 in segments:
        lines_svg.append(
            f'  <line x1="{x1:.3f}" y1="{y1:.3f}" x2="{x2:.3f}" y2="{y2:.3f}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}" '
            f'stroke-opacity="{stroke_opacity}" stroke-linecap="round"/>'
        )

    center_r = (render_size - 1) / 2
    svg = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{render_size}" height="{render_size}" '
        f'viewBox="0 0 {render_size} {render_size}">\n'
        f'  <rect width="100%" height="100%" fill="{background}"/>\n'
        f'  <clipPath id="circleClip">\n'
        f'    <circle cx="{center_r:.3f}" cy="{center_r:.3f}" r="{center_r:.3f}"/>\n'
        f"  </clipPath>\n"
        f'  <g clip-path="url(#circleClip)">\n'
        + "\n".join(lines_svg)
        + "\n  </g>\n</svg>\n"
    )
    Path(path).write_text(svg, encoding="utf-8")


def save_threads_image(lines, size, path, darkness=0.07, nail_count=0, preview_scale=1.0):
    render_size = int(round(size * preview_scale)) if preview_scale != 1.0 else size
    canvas = draw_threads(
        lines,
        size,
        darkness=darkness,
        nail_count=nail_count,
        render_size=render_size if preview_scale != 1.0 else None,
    )
    preview = 1.0 - np.clip(canvas, 0.0, 1.0)
    Image.fromarray((preview * 255).astype(np.uint8)).save(path)


def save_preview_bundle(
    lines,
    size,
    output_dir,
    nail_count=0,
    darkness=0.07,
    preview_scale=3.0,
    png_name="preview.png",
    svg_name="preview.svg",
    dark_on_light=True,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    render_size = int(round(size * preview_scale))

    save_threads_image(
        lines,
        size,
        output_dir / png_name,
        darkness=darkness,
        nail_count=nail_count,
        preview_scale=preview_scale,
    )
    save_threads_svg(
        lines,
        size,
        output_dir / svg_name,
        nail_count=nail_count,
        render_size=render_size,
        dark_on_light=dark_on_light,
    )


def save_comparison_image(
    target,
    lines,
    size,
    path,
    darkness=0.07,
    nail_count=0,
    preview_scale=3.0,
):
    render_size = int(round(size * preview_scale))
    canvas = draw_threads(
        lines,
        size,
        darkness=darkness,
        nail_count=nail_count,
        render_size=render_size,
    )
    preview = 1.0 - np.clip(canvas, 0.0, 1.0)
    original = 1.0 - target
    original_hr = np.array(
        Image.fromarray((original * 255).astype(np.uint8)).resize(
            (render_size, render_size),
            Image.Resampling.LANCZOS,
        )
    ) / 255.0

    stacked = np.concatenate([original_hr, preview], axis=1)
    Image.fromarray((stacked * 255).astype(np.uint8)).save(path)


def compute_line_coverage(lines, size, nail_count=0):
    """Compute how many lines pass through each pixel."""
    canvas = draw_threads(lines, size, darkness=1.0, nail_count=nail_count)
    return canvas


def compute_error_map(target, reconstruction):
    """Compute per-pixel error map."""
    error = np.abs(reconstruction - target)
    return error


def save_error_maps(target, lines, size, path_prefix, darkness=0.07, nail_count=0):
    """Save error visualization maps."""
    canvas = draw_threads(lines, size, darkness=darkness, nail_count=nail_count)
    error_map = compute_error_map(target, canvas)

    max_error = error_map.max()
    if max_error > 0:
        error_normalized = error_map / max_error
    else:
        error_normalized = error_map

    Image.fromarray((error_normalized * 255).astype(np.uint8)).save(
        f"{path_prefix}_error.png"
    )

    heatmap = np.clip(error_map * 255 * 3, 0, 255).astype(np.uint8)
    Image.fromarray(heatmap).save(f"{path_prefix}_heatmap.png")
