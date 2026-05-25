import numpy as np
from scipy.ndimage import gaussian_filter


def ssim(img1, img2, window_size=11, sigma=1.5):
    """
    Structural Similarity Index (SSIM).
    Returns value in [-1, 1], higher is better.
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = gaussian_filter(img1, sigma=sigma)
    mu2 = gaussian_filter(img2, sigma=sigma)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = gaussian_filter(img1 ** 2, sigma=sigma) - mu1_sq
    sigma2_sq = gaussian_filter(img2 ** 2, sigma=sigma) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sigma=sigma) - mu1_mu2

    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / (denominator + 1e-10)
    return float(np.mean(ssim_map))


def psnr(img1, img2, max_val=1.0):
    """Peak Signal-to-Noise Ratio (dB)."""
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return float(20 * np.log10(max_val / np.sqrt(mse)))


def pearson_correlation(img1, img2):
    """Pearson correlation coefficient."""
    mean1, mean2 = img1.mean(), img2.mean()
    std1, std2 = img1.std(), img2.std()

    if std1 == 0 or std2 == 0:
        return 0.0

    numerator = np.mean((img1 - mean1) * (img2 - mean2))
    denominator = std1 * std2

    return float(numerator / denominator)


def gradient_error(img1, img2):
    """Gradient magnitude error."""
    grad1_y, grad1_x = np.gradient(img1)
    grad2_y, grad2_x = np.gradient(img2)

    grad1_mag = np.sqrt(grad1_x ** 2 + grad1_y ** 2)
    grad2_mag = np.sqrt(grad2_x ** 2 + grad2_y ** 2)

    error = np.abs(grad1_mag - grad2_mag)
    return float(np.mean(error))


def compute_comprehensive_metrics(target, reconstruction, mask=None):
    """
    Compute comprehensive metrics for string art quality evaluation.

    Args:
        target: Ground truth image (H, W), values in [0, 1]
        reconstruction: Reconstructed image (H, W), values in [0, 1]
        mask: Optional mask for region of interest (H, W)

    Returns:
        Dict of metric names to values
    """
    if mask is None:
        mask = np.ones_like(target, dtype=bool)

    target_masked = target[mask]
    recon_masked = reconstruction[mask]

    error = recon_masked - target_masked

    metrics = {
        "mse": float(np.mean(error ** 2)),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "mae": float(np.mean(np.abs(error))),
        "psnr": psnr(target, reconstruction, max_val=1.0),
        "pearson": pearson_correlation(target_masked, recon_masked),
        "gradient_error": gradient_error(target, reconstruction),
        "max_error": float(np.max(np.abs(error))),
        "median_error": float(np.median(np.abs(error))),
        "target_mean": float(target_masked.mean()),
        "recon_mean": float(recon_masked.mean()),
        "target_std": float(target_masked.std()),
        "recon_std": float(recon_masked.std()),
        "pixel_count": int(mask.sum()),
    }

    try:
        metrics["ssim"] = ssim(target, reconstruction)
    except Exception:
        metrics["ssim"] = -1.0

    return metrics


def compute_zone_metrics(target, reconstruction, center_size=0.3):
    """
    Compute metrics for different zones of the image.

    Zones:
    - center: Central square/rectangle (usually face)
    - ring: Middle ring
    - outer: Outer ring (edges/background)

    Args:
        target: Ground truth (H, W)
        reconstruction: Reconstruction (H, W)
        center_size: Fraction of image size for center zone

    Returns:
        Dict with metrics for each zone
    """
    h, w = target.shape
    center_h = int(h * center_size)
    center_w = int(w * center_size)

    y0, y1 = (h - center_h) // 2, (h + center_h) // 2
    x0, x1 = (w - center_w) // 2, (w + center_w) // 2

    center_mask = np.zeros_like(target, dtype=bool)
    center_mask[y0:y1, x0:x1] = True

    ring_mask = ~center_mask
    metrics = {
        "center": compute_comprehensive_metrics(target, reconstruction, center_mask),
        "ring": compute_comprehensive_metrics(target, reconstruction, ring_mask),
        "full": compute_comprehensive_metrics(target, reconstruction),
    }

    return metrics


def format_metrics_report(metrics):
    """Format metrics as a readable report."""
    lines = ["Quality Metrics Report", "=" * 40]

    def add_section(title, section_metrics):
        lines.append(f"\n{title}:")
        for key, value in section_metrics.items():
            if isinstance(value, float):
                lines.append(f"  {key:20s}: {value:8.4f}")
            else:
                lines.append(f"  {key:20s}: {value}")

    if "full" in metrics:
        for zone in ["center", "ring", "full"]:
            add_section(f"Zone: {zone.upper()}", metrics[zone])
    else:
        add_section("Overall", metrics)

    return "\n".join(lines)