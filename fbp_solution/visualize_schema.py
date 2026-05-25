import argparse

from pathlib import Path

from string_art import read_schema, save_preview_bundle


def main():
    parser = argparse.ArgumentParser(description="Визуализация схемы нитей.")
    parser.add_argument("schema", help="CSV-файл с колонками angle_deg,rho_px")
    parser.add_argument("--size", type=int, required=True, help="размер квадратного изображения")
    parser.add_argument("--out", default="preview.png", help="выходной PNG-файл")
    parser.add_argument(
        "--preview-scale",
        type=float,
        default=3.0,
        help="масштаб PNG-превью относительно размера схемы",
    )
    parser.add_argument(
        "--svg",
        action="store_true",
        help="дополнительно сохранить preview.svg рядом с PNG",
    )
    args = parser.parse_args()

    lines = read_schema(args.schema)
    out_path = Path(args.out)

    if args.svg:
        save_preview_bundle(
            lines,
            args.size,
            out_path.parent,
            preview_scale=args.preview_scale,
            png_name=out_path.name,
            svg_name=out_path.with_suffix(".svg").name,
        )
        print(f"Готово: {out_path}")
        print(f"SVG: {out_path.with_suffix('.svg')}")
    else:
        from string_art import save_threads_image_black_background

        save_threads_image_black_background(
            lines,
            args.size,
            out_path,
            preview_scale=args.preview_scale,
        )
        print(f"Готово: {out_path}")


if __name__ == "__main__":
    main()
