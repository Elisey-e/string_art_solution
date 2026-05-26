from pathlib import Path
import argparse
import json
import numpy as np

from string_art import (
    binary_dither,
    binary_sinogram_to_lines,
    circle_mask,
    draw_threads,
    filter_sinogram,
    keep_brightest_per_angle,
    load_grayscale_square,
    prepare_fbp_image,
    normalize_sinogram_probabilities,
    radon_transform,
    save_float_image,
    save_preview_bundle,
    save_schema,
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
        description="Построение схемы ниткографии методом FBP (task.md)."
    )
    parser.add_argument("image", help="квадратное изображение в режиме L")
    parser.add_argument("--angles", type=int, default=360, help="число углов")
    parser.add_argument("--threads", type=int, default=40, help="максимум нитей на один угол")
    parser.add_argument("--seed", type=int, default=1, help="зерно случайного распыления")
    parser.add_argument(
        "--brightness-lift",
        type=float,
        default=0.2,
        help="подъём яркости исходника перед Radon (см. task.md)",
    )
    parser.add_argument(
        "--prob-gamma",
        type=float,
        default=0.85,
        help="гамма вероятностей sinogram перед dither",
    )
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
    target = prepare_fbp_image(image, brightness_lift=args.brightness_lift)

    sinogram, angles = radon_transform(target, args.angles)
    filtered = filter_sinogram(sinogram)
    burned = keep_brightest_per_angle(filtered, args.threads)
    probabilities = normalize_sinogram_probabilities(burned, gamma=args.prob_gamma)

    if args.dither_mode == "error-diffusion":
        binary = sinogram_error_diffusion(probabilities, args.seed)
    else:
        binary = binary_dither(probabilities, args.seed)

    lines, weights = binary_sinogram_to_lines(binary, angles, probabilities)

    save_float_image(sinogram, output_dir / "01_sinogram.png")
    save_float_image(filtered, output_dir / "02_filtered.png")
    save_float_image(probabilities, output_dir / "03_probabilities.png")
    save_float_image(binary, output_dir / "04_binary_sinogram.png")
    save_schema(lines, output_dir / "schema.csv")
    save_preview_bundle(
        lines,
        image.shape[0],
        output_dir,
        preview_scale=3,
        weights=weights,
    )

    recon_canvas = np.clip(draw_threads(lines, image.shape[0], weights=weights), 0.0, 1.0)
    rm = simple_reconstruction_metrics(target, recon_canvas)

    metadata = {
        "image_size": int(image.shape[0]),
        "angle_count": args.angles,
        "threads_per_angle": args.threads,
        "thread_count": len(lines),
        "brightness_lift": args.brightness_lift,
        "prob_gamma": args.prob_gamma,
        "dither_mode": args.dither_mode,
        "algorithm": "fbp_radon",
        "rmse": rm["rmse"],
        "mae": rm["mae"],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    hr = int(image.shape[0] * 3)
    print(f"Готово: {output_dir / 'schema.csv'}")
    print(f"Нитей: {len(lines)}")
    print(f"RMSE: {rm['rmse']:.4f}, MAE: {rm['mae']:.4f}")
    print(f"PNG: {output_dir / 'preview.png'} ({hr}×{hr})")
    print(f"SVG: {output_dir / 'preview.svg'}")


if __name__ == "__main__":
    main()
