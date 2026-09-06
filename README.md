# YOLO Inference

Проект для инференса моделей `ultralytics YOLO` на:

- изображениях
- видео
- последовательностях изображений в папке

Результат сохраняется с отрисованными bounding boxes.

## Структура

- `main.py` - CLI-вход с 4 аргументами
- `core/config.py` - загрузка конфигов модели и отрисовки
- `core/pipeline.py` - основной reusable pipeline инференса
- `core/drawer.py` - логика отрисовки боксов и подписей
- `BBoxDrawer.py` - совместимый экспорт старой логики
- `configs/` - примеры конфигураций

## Установка

Рекомендуется ставить зависимости в виртуальное окружение.

Зависимости проекта:

- `ultralytics`
- `opencv-python`
- `PyYAML`
- `tqdm`

```bash
pip install -r requirements.txt
```

Порядок установки важен:

1. Установить `ultralytics`.
2. После этого переустановить `torch` и `torchvision` с CUDA-версией, подходящей под вашу систему.

Пример для CUDA-сборки PyTorch:

```bash
pip install ultralytics
pip install --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Если у вас другая версия CUDA, замените `cu121` на нужную.

## Запуск

CLI принимает 4 аргумента:

```bash
python main.py <source_path> <model_config_path> <output_path> <draw_config_path>
```

### Примеры

Изображение:

```bash
python main.py data/image.jpg configs/model_config.json results/image_out.jpg configs/draw_config.json
```

Видео:

```bash
python main.py data/video.mp4 configs/model_config.json results/video_out.mp4 configs/draw_config.json
```

Папка с кадрами:

```bash
python main.py data/frames configs/model_config.json results/frames_out configs/draw_config.json
```

## Конфиг модели

Поддерживаемые поля:

- `weights_path`
- `conf`
- `iou`
- `agnostic_nms`
- `imgsz`
- `device`
- `verbose`

Если `imgsz = null`, значение не передается в `YOLO.predict`, и используется поведение по умолчанию.

## Конфиг отрисовки

Поддерживаемые поля:

- `thickness`
- `class_colors`
- `default_color`
- `box_mode` - `corners` или `full`
- `corner_style`
  - `min_line_length`
  - `max_line_length`
  - `min_box_size`
  - `max_box_size`
  - `line_width`
- `draw_class_name`
- `draw_confidence`
- `font_scale`
- `font_thickness`
- `text_padding`
- `text_background`

## Переиспользование

Core-логика вынесена в `core.pipeline`.

Можно использовать напрямую:

```python
from core.pipeline import run_inference

result = run_inference(
  source_path="data/image.jpg",
  model_config_path="configs/model_config.json",
  output_path="results/image_out.jpg",
  draw_config_path="configs/draw_config.json",
)
```
