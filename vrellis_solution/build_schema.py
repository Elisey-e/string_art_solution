from pathlib import Path
import argparse
import json

from stringart_vrellis import (
    generate_string_art,
    load_grayscale_square,
    nail_positions_circle,
    preprocess_portrait,
    save_preview_png,
    save_preview_svg,
    save_schema_csv,
)


def main():
    parser = argparse.ArgumentParser(
        description="String art (Vrellis / grvlbit greedy на гвоздях)."
    )
    parser.add_argument("image", help="квадратное изображение L")
    parser.add_argument("--nails", type=int, default=240, help="число гвоздей")
    parser.add_argument("--lines", type=int, default=5000, help="макс. итераций (нитей)")
    parser.add_argument("--weight", type=float, default=0.08, help="вклад одной нити")
    parser.add_argument("--min-gap", type=int, default=12, help="мин. шаг по дуге")
    parser.add_argument("--seed-nail", type=int, default=0, help="стартовый гвоздь")
    parser.add_argument("--out", default="out", help="папка результата")
    args = parser.parse_args()

    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = load_grayscale_square(args.image)
    target = preprocess_portrait(image)

    print(
        f"Генерация: {args.nails} гвоздей, до {args.lines} нитей, weight={args.weight}...",
        flush=True,
    )
    segments, residual = generate_string_art(
        target,
        n_nails=args.nails,
        max_iterations=args.lines,
        line_weight=args.weight,
        min_nail_gap=args.min_gap,
        seed_nail=args.seed_nail,
    )

    size = image.shape[0]
    nails = nail_positions_circle(args.nails, size)

    save_schema_csv(segments, output_dir / "schema.csv")
    save_preview_png(segments, nails, size, output_dir / "preview.png", preview_scale=3.0)
    save_preview_svg(segments, nails, size, output_dir / "preview.svg", preview_scale=3.0)

    rmse = float((residual ** 2).mean() ** 0.5)

    metadata = {
        "image_size": size,
        "nail_count": args.nails,
        "thread_count": len(segments),
        "line_weight": args.weight,
        "min_nail_gap": args.min_gap,
        "residual_rmse": rmse,
        "algorithm": "vrellis_greedy",
        "reference": "https://github.com/grvlbit/stringart (Petros Vrellis)",
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    hr = size * 3
    print(f"Готово: {output_dir / 'schema.csv'}")
    print(f"Нитей: {len(segments)}")
    print(f"Остаток RMSE: {rmse:.4f}")
    print(f"PNG: {output_dir / 'preview.png'} ({hr}×{hr})")
    print(f"SVG: {output_dir / 'preview.svg'}")


if __name__ == "__main__":
    main()
