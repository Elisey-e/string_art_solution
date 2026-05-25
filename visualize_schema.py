import argparse

from string_art import read_schema, save_threads_image


def main():
    parser = argparse.ArgumentParser(description="Визуализация схемы нитей.")
    parser.add_argument("schema", help="CSV-файл с колонками angle_deg,rho_px")
    parser.add_argument("--size", type=int, required=True, help="размер квадратного изображения")
    parser.add_argument("--out", default="preview.png", help="выходной PNG-файл")
    args = parser.parse_args()

    lines = read_schema(args.schema)
    save_threads_image(lines, args.size, args.out)
    print(f"Готово: {args.out}")


if __name__ == "__main__":
    main()
