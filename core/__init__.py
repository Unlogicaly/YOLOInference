from .config import DrawConfig, ModelConfig, load_draw_config, load_model_config

__all__ = [
    "DrawConfig",
    "InferencePipeline",
    "ModelConfig",
    "load_draw_config",
    "load_model_config",
    "run_inference",
]


def __getattr__(name: str):
    if name in {"InferencePipeline", "run_inference"}:
        from .pipeline import InferencePipeline, run_inference

        exports = {
            "InferencePipeline": InferencePipeline,
            "run_inference": run_inference,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
