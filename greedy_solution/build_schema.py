from pathlib import Path
import argparse
import json
import numpy as np

from metrics import compute_zone_metrics, format_metrics_report

from string_art import (
    greedy_string_art,
    load_grayscale_square,
    prepare_target,
    reconstruction_metrics,
    save_comparison_image,
    save_error_maps,
    save_preview_bundle,
    save_schema,
    sinogram_error_diffusion,
)


def main():
    parser = argparse.ArgumentParser(
        description="Построение схемы ниткографии жадным методом по остатку."
    )
    parser.add_argument("image", help="квадратное изображение в режиме L")
    parser.add_argument("--angles", type=int, default=180, help="число углов кандидатов")
    parser.add_argument(
        "--threads-per-angle",
        type=int,
        default=40,
        help="кандидатов на угол из sinogram цели",
    )
    parser.add_argument("--lines", type=int, default=4000, help="максимум нитей")
    parser.add_argument(
        "--full-catalog",
        action="store_true",
        help="все линии (медленно, ~80k кандидатов)",
    )
    parser.add_argument(
        "--nails",
        type=int,
        default=256,
        help="число гвоздей по окружности (0 = режим sinogram)",
    )
    parser.add_argument(
        "--min-nail-gap",
        type=int,
        default=14,
        help="минимальная дистанция между гвоздями по дуге",
    )
    parser.add_argument(
        "--dither-mode",
        choices=["random", "error-diffusion", "serpentine"],
        default="random",
        help="режим дизеринга sinogram (для sinogram-режима)",
    )
    parser.add_argument(
        "--darkness",
        type=float,
        default=0.03,
        help="вклад одной нити в накопленную темноту [0..1]",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=1e-4,
        help="остановка, если лучшая нить даёт меньший вклад",
    )
    parser.add_argument("--out", default="out", help="папка результата")
    args = parser.parse_args()

    if args.angles <= 0:
        raise SystemExit("Число углов должно быть положительным.")

    if args.lines <= 0:
        raise SystemExit("Число нитей должно быть положительным.")

    if args.darkness <= 0:
        raise SystemExit("Параметр darkness должен быть положительным.")

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = load_grayscale_square(args.image)
    target = prepare_target(image)

    print("Строю каталог линий и жадно подбираю нити...")
    lines = greedy_string_art(
        target,
        angle_count=args.angles,
        max_lines=args.lines,
        darkness=args.darkness,
        min_score=args.min_score,
        threads_per_angle=args.threads_per_angle,
        use_full_catalog=args.full_catalog,
        nail_count=args.nails,
        min_nail_gap=args.min_nail_gap,
    )

    use_nails = args.nails > 0
    recon = reconstruction_metrics(
        target,
        lines,
        image.shape[0],
        darkness=args.darkness,
        nail_count=args.nails if use_nails else 0,
    )

    canvas = (
        np.zeros((image.shape[0], image.shape[0]))
        if use_nails
        else None
    )
    if use_nails:
        from string_art import draw_threads

        canvas = draw_threads(
            lines,
            image.shape[0],
            darkness=args.darkness,
            nail_count=args.nails,
        )
        canvas = np.clip(canvas, 0.0, 1.0)

    zone_metrics = compute_zone_metrics(target, canvas if use_nails else canvas)

    save_schema(lines, output_dir / "schema.csv", use_nails=use_nails)
    save_preview_bundle(
        lines,
        image.shape[0],
        output_dir,
        nail_count=args.nails if use_nails else 0,
        darkness=args.darkness,
        preview_scale=3.0,
    )
    save_comparison_image(
        target,
        lines,
        image.shape[0],
        output_dir / "compare.png",
        darkness=args.darkness,
        nail_count=args.nails if use_nails else 0,
        preview_scale=3.0,
    )

    if use_nails and canvas is not None:
        save_error_maps(
            target,
            lines,
            image.shape[0],
            output_dir / "maps",
            darkness=args.darkness,
            nail_count=args.nails,
        )

    metadata = {
        "image_size": int(image.shape[0]),
        "angle_count": args.angles,
        "threads_per_angle": args.threads_per_angle,
        "full_catalog": args.full_catalog,
        "nail_count": args.nails,
        "min_nail_gap": args.min_nail_gap,
        "max_lines": args.lines,
        "thread_count": len(lines),
        "darkness": args.darkness,
        "min_score": args.min_score,
        "reconstruction_rmse": recon["rmse"],
        "reconstruction_mae": recon["mae"],
        "zone_center_rmse": zone_metrics["center"]["rmse"],
        "zone_ring_rmse": zone_metrics["ring"]["rmse"],
        "zone_center_ssim": zone_metrics["center"]["ssim"],
        "zone_ring_ssim": zone_metrics["ring"]["ssim"],
        "zone_full_ssim": zone_metrics["full"]["ssim"],
        "algorithm": "greedy_residual",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Готово: {output_dir / 'schema.csv'}")
    print(f"Нитей: {len(lines)}")
    print(f"Полный RMSE: {recon['rmse']:.4f}, MAE: {recon['mae']:.4f}")
    print(f"Центр RMSE: {zone_metrics['center']['rmse']:.4f}, SSIM: {zone_metrics['center']['ssim']:.4f}")
    print(f"Кольцо RMSE: {zone_metrics['ring']['rmse']:.4f}, SSIM: {zone_metrics['ring']['ssim']:.4f}")
    hr = int(image.shape[0] * 3)
    print(f"PNG: {output_dir / 'preview.png'} ({hr}×{hr})")
    print(f"SVG: {output_dir / 'preview.svg'}")
    print(f"Сравнение: {output_dir / 'compare.png'} ({hr}×{hr * 2})")

    if use_nails:
        report = format_metrics_report(zone_metrics)
        (output_dir / "metrics_report.txt").write_text(report, encoding="utf-8")
        print(f"\nМетрики: {output_dir / 'metrics_report.txt'}")


if __name__ == "__main__":
    main()
