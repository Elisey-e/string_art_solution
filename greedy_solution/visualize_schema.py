import argparse
from pathlib import Path

from string_art import read_schema, save_preview_bundle, save_threads_image


def main():
    parser = argparse.ArgumentParser(description="Визуализация схемы нитей.")
    parser.add_argument("schema", help="CSV-файл (angle_deg,rho_px или nail_a,nail_b,...)")
    parser.add_argument("--size", type=int, required=True, help="размер квадратного изображения")
    parser.add_argument(
        "--nails",
        type=int,
        default=0,
        help="число гвоздей, если schema.csv содержит nail_a,nail_b",
    )
    parser.add_argument("--out", default="preview.png", help="выходной PNG-файл")
    parser.add_argument(
        "--preview-scale",
        type=float,
        default=3.0,
        help="масштаб PNG-превью относительно размера схемы",
    )
    parser.add_argument(
        "--darkness",
        type=float,
        default=0.03,
        help="вклад одной нити при растеризации",
    )
    parser.add_argument(
        "--svg",
        action="store_true",
        help="дополнительно сохранить preview.svg рядом с PNG",
    )
    args = parser.parse_args()

    lines = read_schema(args.schema)
    nail_count = args.nails
    if nail_count == 0 and lines and lines[0][2] is not None:
        nail_count = max(max(line[2], line[3]) for line in lines) + 1

    out_path = Path(args.out)

    if args.svg:
        save_preview_bundle(
            lines,
            args.size,
            out_path.parent,
            nail_count=nail_count,
            darkness=args.darkness,
            preview_scale=args.preview_scale,
            png_name=out_path.name,
            svg_name=out_path.with_suffix(".svg").name,
        )
        print(f"Готово: {out_path}")
        print(f"SVG: {out_path.with_suffix('.svg')}")
    else:
        save_threads_image(
            lines,
            args.size,
            out_path,
            darkness=args.darkness,
            nail_count=nail_count,
            preview_scale=args.preview_scale,
        )
        print(f"Готово: {out_path}")


if __name__ == "__main__":
    main()
