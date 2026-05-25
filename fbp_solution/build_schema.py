from pathlib import Path
import argparse
import json
import numpy as np

from string_art import (
    binary_dither,
    binary_sinogram_to_lines,
    circle_mask,
    filter_sinogram,
    keep_brightest_per_angle,
    load_grayscale_square,
    normalize_nonnegative,
    radon_transform,
    save_float_image,
    save_schema,
    save_threads_image,
    save_preview_bundle,
    sinogram_error_diffusion,
)


def simple_reconstruction_metrics(target, reconstruction):
    mask = circle_mask(target.shape[0])
    error = reconstruction[mask] - target[mask]
    mse = float(np.mean(error ** 2))
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(mse))
    return {"rmse": rmse, "mae": mae}


def main():
    parser = argparse.ArgumentParser(
        description="Построение схемы ниткографии методом FBP."
    )
    parser.add_argument("image", help="квадратное изображение в режиме L")
    parser.add_argument("--angles", type=int, default=180, help="число углов")
    parser.add_argument("--threads", type=int, default=20, help="максимум нитей на один угол")
    parser.add_argument("--seed", type=int, default=1, help="зерно случайного распыления")
    parser.add_argument(
        "--dither-mode",
        choices=["random", "error-diffusion"],
        default="random",
        help="режим дизеринга sinogram",
    )
    parser.add_argument("--out", default="out", help="папка результата")
    args = parser.parse_args()

    if args.angles <= 0:
        raise SystemExit("Число углов должно быть положительным.")

    if args.threads <= 0:
        raise SystemExit("Число нитей на угол должно быть положительным.")

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = load_grayscale_square(args.image)
    image = (1.0 - image) * circle_mask(image.shape[0])

    sinogram, angles = radon_transform(image, args.angles)
    filtered = filter_sinogram(sinogram)
    burned = keep_brightest_per_angle(filtered, args.threads)
    probabilities = normalize_nonnegative(burned)

    if args.dither_mode == "error-diffusion":
        binary = sinogram_error_diffusion(probabilities, args.seed)
    else:
        binary = binary_dither(probabilities, args.seed)

    lines = binary_sinogram_to_lines(binary, angles)

    save_float_image(sinogram, output_dir / "01_sinogram.png")
    save_float_image(filtered, output_dir / "02_filtered.png")
    save_float_image(probabilities, output_dir / "03_probabilities.png")
    save_float_image(binary, output_dir / "04_binary_sinogram.png")
    save_schema(lines, output_dir / "schema.csv")
    save_preview_bundle(lines, image.shape[0], output_dir, preview_scale=3)

    recon_canvas = None
    try:
        from string_art import draw_threads

        recon_canvas = draw_threads(lines, image.shape[0])
        recon_canvas = np.clip(recon_canvas, 0.0, 1.0)
    except Exception:
        pass

    metrics = {
        "image_size": int(image.shape[0]),
        "angle_count": args.angles,
        "threads_per_angle": args.threads,
        "thread_count": len(lines),
        "algorithm": "fbp_radon",
    }

    if recon_canvas is not None:
        rm = simple_reconstruction_metrics(image, recon_canvas)
        metrics["rmse"] = rm["rmse"]
        metrics["mae"] = rm["mae"]

    (output_dir / "metadata.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(f"Готово: {output_dir / 'schema.csv'}")
    print(f"Нитей: {len(lines)}")
    if "rmse" in metrics:
        print(f"RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}")
    print(f"PNG: {output_dir / 'preview.png'} ({image.shape[0] * 3}×{image.shape[0] * 3})")
    print(f"SVG: {output_dir / 'preview.svg'}")


if __name__ == "__main__":
    main()