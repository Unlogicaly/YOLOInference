from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
from tqdm import tqdm
from ultralytics import YOLO

from .config import DrawConfig, ModelConfig
from .drawer import BBoxDrawer, Detection

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg", ".wmv", ".m4v"}


@dataclass(slots=True)
class InferenceResult:
    source_type: str
    source_path: Path
    output_path: Path
    processed_items: int


class InferencePipeline:
    def __init__(self, model_config: ModelConfig, draw_config: DrawConfig):
        self.model_config = model_config
        self.draw_config = draw_config
        self.model = YOLO(str(model_config.weights_path))
        self.model_stride = _model_stride(self.model)
        self.drawer = BBoxDrawer(draw_config)

    def run(self, source_path: str | Path, output_path: str | Path) -> InferenceResult:
        source = Path(source_path)
        target = Path(output_path)

        if source.is_dir():
            return self._run_image_sequence(source, target)
        if _is_image_file(source):
            return self._run_single_image(source, target)
        if _is_video_file(source):
            return self._run_video(source, target)

        raise ValueError(f"Unsupported source type: {source}")

    def _run_single_image(self, source: Path, output: Path) -> InferenceResult:
        frame = cv2.imread(str(source))
        if frame is None:
            raise RuntimeError(f"Failed to read image: {source}")

        annotated = self._process_frame(frame)
        if output.exists() and output.is_dir():
            output = output / source.name
        elif output.suffix == "":
            output = output.with_suffix(source.suffix)
        _prepare_parent(output)
        if not cv2.imwrite(str(output), annotated):
            raise RuntimeError(f"Failed to write image: {output}")

        return InferenceResult("image", source, output, 1)

    def _run_image_sequence(self, source: Path, output: Path) -> InferenceResult:
        frame_paths = _collect_image_files(source)
        if not frame_paths:
            raise ValueError(f"No image files found in directory: {source}")

        output.mkdir(parents=True, exist_ok=True)
        for frame_path in tqdm(frame_paths, desc=f"Processing {source.name}", unit="frame"):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Failed to read image: {frame_path}")
            annotated = self._process_frame(frame)
            destination = output / frame_path.name
            if not cv2.imwrite(str(destination), annotated):
                raise RuntimeError(f"Failed to write image: {destination}")

        return InferenceResult("image_sequence", source, output, len(frame_paths))

    def _run_video(self, source: Path, output: Path) -> InferenceResult:
        if output.exists() and output.is_dir():
            output = output / f"{source.stem}_annotated.mp4"
        elif output.suffix == "":
            output = output.with_suffix(".mp4")

        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {source}")

        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

        _prepare_parent(output)
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"Failed to open output video writer: {output}")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        progress_total = total_frames if total_frames > 0 else None
        processed_frames = 0
        try:
            with tqdm(total=progress_total, desc=f"Processing {source.name}", unit="frame") as progress:
                while True:
                    has_frame, frame = capture.read()
                    if not has_frame:
                        break
                    annotated = self._process_frame(frame)
                    writer.write(annotated)
                    processed_frames += 1
                    progress.update(1)
        finally:
            capture.release()
            writer.release()

        return InferenceResult("video", source, output, processed_frames)

    def _process_frame(self, frame):
        predict_kwargs = {
            "conf": self.model_config.conf,
            "iou": self.model_config.iou,
            "agnostic_nms": self.model_config.agnostic_nms,
            "device": self.model_config.device,
            "verbose": self.model_config.verbose,
        }
        if self.model_config.imgsz is not None:
            predict_kwargs["imgsz"] = _align_imgsz(self.model_config.imgsz, self.model_stride)
        else:
            predict_kwargs["imgsz"] = _align_imgsz(_frame_imgsz(frame), self.model_stride)

        results = self.model(frame, **predict_kwargs)

        detections = self._extract_detections(results[0])
        return self.drawer.draw(frame.copy(), detections)

    def _extract_detections(self, result) -> list[Detection]:
        detections: list[Detection] = []
        boxes = result.boxes
        if boxes is None:
            return detections

        names = result.names or {}
        for box in boxes:
            coords = box.xyxy[0].tolist()
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            class_name = str(names.get(class_id, class_id))
            detections.append(
                Detection(
                    x1=int(coords[0]),
                    y1=int(coords[1]),
                    x2=int(coords[2]),
                    y2=int(coords[3]),
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                )
            )
        return detections


def run_inference(
    source_path: str | Path,
    model_config_path: str | Path,
    output_path: str | Path,
    draw_config_path: str | Path,
) -> InferenceResult:
    from .config import load_draw_config, load_model_config

    model_config = load_model_config(model_config_path)
    draw_config = load_draw_config(draw_config_path)
    pipeline = InferencePipeline(model_config, draw_config)
    return pipeline.run(source_path, output_path)


def _collect_image_files(directory: Path) -> list[Path]:
    return sorted(
        [path for path in directory.iterdir() if path.is_file() and _is_image_file(path)],
        key=lambda path: path.name,
    )


def _prepare_parent(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)


def _is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def _is_video_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES


def _frame_imgsz(frame) -> tuple[int, int]:
    height, width = frame.shape[:2]
    return height, width


def _align_imgsz(imgsz: int | tuple[int, int] | list[int], stride: int):
    if isinstance(imgsz, int):
        return _round_up_to_multiple(imgsz, stride)

    height, width = int(imgsz[0]), int(imgsz[1])
    return _round_up_to_multiple(height, stride), _round_up_to_multiple(width, stride)


def _round_up_to_multiple(value: int, multiple: int) -> int:
    if multiple <= 0:
        return value
    return ((value + multiple - 1) // multiple) * multiple


def _model_stride(model) -> int:
    stride = getattr(model, "stride", 32)
    if isinstance(stride, int):
        return stride
    if hasattr(stride, "max"):
        return int(stride.max())
    if isinstance(stride, (list, tuple)):
        return int(max(stride))
    return 32
