from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="YOLO inference for image, video, or directory with image sequence.",
    )
    parser.add_argument("--source_path", help="Path to an image, video, or directory with frames.")
    parser.add_argument("--model_config_path", help="Path to JSON/YAML model config.")
    parser.add_argument("--output_path", help="Path to output file or output directory.")
    parser.add_argument("--draw_config_path", help="Path to JSON/YAML draw config.")
    return parser


def main() -> int:
    args = build_parser().parse_args()

    from core.pipeline import run_inference

    result = run_inference(
        source_path=args.source_path,
        model_config_path=args.model_config_path,
        output_path=args.output_path,
        draw_config_path=args.draw_config_path,
    )
    print(
        f"Processed {result.processed_items} item(s). "
        f"Source type: {result.source_type}. Output: {result.output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
