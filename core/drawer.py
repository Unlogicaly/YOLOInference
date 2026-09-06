from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import CornerStyleConfig, DrawConfig


@dataclass(slots=True)
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int
    class_id: int
    class_name: str
    confidence: float


def draw_bounding_box(
    image,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    thickness: int = 2,
    corners_only: bool = True,
    corner_style: CornerStyleConfig | None = None,
):
    corner_style = corner_style or CornerStyleConfig()

    if corners_only:
        box_width = x2 - x1
        box_height = y2 - y1

        coef_x = np.clip(
            (box_width - corner_style.min_box_size)
            / max(corner_style.max_box_size - corner_style.min_box_size, 1),
            0.0,
            1.0,
        )
        coef_y = np.clip(
            (box_height - corner_style.min_box_size)
            / max(corner_style.max_box_size - corner_style.min_box_size, 1),
            0.0,
            1.0,
        )

        line_length_x = int(
            corner_style.min_line_length
            + coef_x * (corner_style.max_line_length - corner_style.min_line_length)
        )
        line_length_y = int(
            corner_style.min_line_length
            + coef_y * (corner_style.max_line_length - corner_style.min_line_length)
        )

        line_width = corner_style.line_width

        cv2.line(image, (x1, y1), (x1 + line_length_x, y1), color, line_width)
        cv2.line(image, (x1, y1), (x1, y1 + line_length_y), color, line_width)
        cv2.line(image, (x1, y2), (x1 + line_length_x, y2), color, line_width)
        cv2.line(image, (x1, y2), (x1, y2 - line_length_y), color, line_width)

        cv2.line(image, (x2, y1), (x2 - line_length_x, y1), color, line_width)
        cv2.line(image, (x2, y1), (x2, y1 + line_length_y), color, line_width)
        cv2.line(image, (x2, y2), (x2 - line_length_x, y2), color, line_width)
        cv2.line(image, (x2, y2), (x2, y2 - line_length_y), color, line_width)
    else:
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)

    return image


class BBoxDrawer:
    def __init__(self, config: DrawConfig):
        self.config = config

    def draw(self, frame, detections: list[Detection]):
        if frame is None:
            return None

        for detection in detections:
            color = self.config.class_colors.get(detection.class_id, self.config.default_color)
            if color is None:
                continue
            draw_bounding_box(
                frame,
                detection.x1,
                detection.y1,
                detection.x2,
                detection.y2,
                color,
                thickness=self.config.thickness,
                corners_only=self.config.box_mode == "corners",
                corner_style=self.config.corner_style,
            )

            text = self._build_label(detection)
            if text:
                self._draw_label(frame, detection.x1, detection.y1, text, color)

        return frame

    def _build_label(self, detection: Detection) -> str:
        parts: list[str] = []
        if self.config.draw_class_name:
            parts.append(detection.class_name)
        if self.config.draw_confidence:
            parts.append(f"{detection.confidence:.2f}")
        return " | ".join(parts)

    def _draw_label(self, frame, x1: int, y1: int, text: str, color: tuple[int, int, int]):
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size, baseline = cv2.getTextSize(
            text,
            font,
            self.config.font_scale,
            self.config.font_thickness,
        )
        text_width, text_height = text_size
        padding = self.config.text_padding

        text_x = max(x1, 0)
        text_y = max(y1 - 8, text_height + padding)

        if self.config.text_background:
            top_left = (text_x - padding, text_y - text_height - padding)
            bottom_right = (text_x + text_width + padding, text_y + baseline)
            cv2.rectangle(frame, top_left, bottom_right, color, thickness=-1)
            text_color = (0, 0, 0)
        else:
            text_color = color

        cv2.putText(
            frame,
            text,
            (text_x, text_y),
            font,
            self.config.font_scale,
            text_color,
            self.config.font_thickness,
            lineType=cv2.LINE_AA,
        )
