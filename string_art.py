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


def draw_threads(lines, size):
    canvas = np.zeros((size, size), dtype=float)
    center = (size - 1) / 2
    radius = center

    for angle, offset in lines:
        phi = np.deg2rad(angle)
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        half_length = np.sqrt(radius ** 2 - offset ** 2)
        line_points = np.linspace(-half_length, half_length, size * 2)

        x = offset * cos_phi - line_points * sin_phi
        y = offset * sin_phi + line_points * cos_phi

        rows = np.rint(center + y).astype(int)
        cols = np.rint(center + x).astype(int)
        valid = (0 <= rows) & (rows < size) & (0 <= cols) & (cols < size)

        np.add.at(canvas, (rows[valid], cols[valid]), 1.0)

    canvas *= circle_mask(size)
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


def save_threads_image_black_background(lines, size, path):
    canvas = draw_threads(lines, size)
    image = canvas
    limit = np.quantile(image[image > 0], 0.995) if (image > 0).sum() > 0 else image.max()
    image = np.clip(image, 0, limit)
    if image.max() > 0:
        image /= image.max()
    Image.fromarray((image * 255).astype(np.uint8)).save(path)
