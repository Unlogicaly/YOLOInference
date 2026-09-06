from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency for yaml configs
    yaml = None


@dataclass(slots=True)
class ModelConfig:
    weights_path: Path
    conf: float = 0.25
    iou: float = 0.45
    agnostic_nms: bool = False
    imgsz: int | tuple[int, int] | list[int] | None = None
    device: str | int | None = None
    verbose: bool = False


@dataclass(slots=True)
class CornerStyleConfig:
    min_line_length: int = 3
    max_line_length: int = 25
    min_box_size: int = 10
    max_box_size: int = 100
    line_width: int = 2


@dataclass(slots=True)
class DrawConfig:
    thickness: int = 2
    class_colors: dict[int, tuple[int, int, int]] = field(default_factory=dict)
    default_color: tuple[int, int, int] = (0, 255, 255)
    box_mode: str = "corners"
    corner_style: CornerStyleConfig = field(default_factory=CornerStyleConfig)
    draw_class_name: bool = True
    draw_confidence: bool = True
    font_scale: float = 0.5
    font_thickness: int = 1
    text_padding: int = 4
    text_background: bool = True


def load_model_config(config_path: str | Path) -> ModelConfig:
    data = _load_config_file(config_path)
    weights_path = Path(data["weights_path"])
    imgsz = _parse_imgsz(data.get("imgsz"))
    return ModelConfig(
        weights_path=weights_path,
        conf=float(data.get("conf", 0.25)),
        iou=float(data.get("iou", 0.45)),
        agnostic_nms=bool(data.get("agnostic_nms", False)),
        imgsz=imgsz,
        device=data.get("device"),
        verbose=bool(data.get("verbose", False)),
    )


def load_draw_config(config_path: str | Path) -> DrawConfig:
    data = _load_config_file(config_path)
    corner_style_data = data.get("corner_style", {})
    class_colors_raw = data.get("class_colors", {})
    class_colors = {
        int(class_id): _parse_color(color)
        for class_id, color in class_colors_raw.items()
    }
    return DrawConfig(
        thickness=int(data.get("thickness", 2)),
        class_colors=class_colors,
        default_color=_parse_color(data.get("default_color", [0, 255, 255])),
        box_mode=str(data.get("box_mode", "corners")),
        corner_style=CornerStyleConfig(
            min_line_length=int(corner_style_data.get("min_line_length", 3)),
            max_line_length=int(corner_style_data.get("max_line_length", 25)),
            min_box_size=int(corner_style_data.get("min_box_size", 10)),
            max_box_size=int(corner_style_data.get("max_box_size", 100)),
            line_width=int(corner_style_data.get("line_width", 2)),
        ),
        draw_class_name=bool(data.get("draw_class_name", True)),
        draw_confidence=bool(data.get("draw_confidence", True)),
        font_scale=float(data.get("font_scale", 0.5)),
        font_thickness=int(data.get("font_thickness", 1)),
        text_padding=int(data.get("text_padding", 4)),
        text_background=bool(data.get("text_background", True)),
    )


def _load_config_file(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    raw_text = path.read_text(encoding="utf-8")
    if suffix == ".json":
        data = json.loads(raw_text)
    elif suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to read YAML configs.")
        data = yaml.safe_load(raw_text)
    else:
        raise ValueError(f"Unsupported config format: {path.suffix}")

    if not isinstance(data, dict):
        raise ValueError(f"Config must contain an object at the top level: {path}")
    return data


def _parse_color(color: Any) -> tuple[int, int, int] | None:
    if (not isinstance(color, (list, tuple)) or len(color) != 3) and color is not None:
        raise ValueError(f"Color must be a 3-item list or tuple, got: {color}")

    if color is None:
        return None

    return int(color[0]), int(color[1]), int(color[2])


def _parse_imgsz(imgsz: Any) -> int | tuple[int, int] | list[int] | None:
    if imgsz is None:
        return None
    if isinstance(imgsz, int):
        return imgsz
    if isinstance(imgsz, (list, tuple)):
        if len(imgsz) != 2:
            raise ValueError(f"imgsz list/tuple must contain exactly 2 items, got: {imgsz}")
        return int(imgsz[0]), int(imgsz[1])
    raise ValueError(f"Unsupported imgsz value: {imgsz}")
