"""Lightweight helpers for resolving local model paths."""

from __future__ import annotations

from pathlib import Path


def resolve_model_path(model_name_or_path: str, cache_dir: str | None = None) -> str:
    """Resolve ``model_name_or_path`` against the workspace and an optional
    ``cache_dir``. Falls back to the original string if nothing matches locally,
    which allows HuggingFace Hub identifiers to pass through untouched.
    """
    input_path = Path(model_name_or_path).expanduser()
    project_root = Path(__file__).resolve().parent.parent

    candidate_paths: list[Path] = []
    if input_path.is_absolute():
        candidate_paths.append(input_path)
    else:
        candidate_paths.append((Path.cwd() / input_path).resolve())
        candidate_paths.append((project_root / input_path).resolve())

    if cache_dir is not None:
        cache_path = Path(cache_dir).expanduser()
        if not cache_path.is_absolute():
            candidate_paths.append((Path.cwd() / cache_path / input_path.name).resolve())
            candidate_paths.append((project_root / cache_path / input_path.name).resolve())
        else:
            candidate_paths.append((cache_path / input_path.name).resolve())

    seen: set[str] = set()
    for candidate_path in candidate_paths:
        candidate_str = str(candidate_path)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        if candidate_path.exists():
            return candidate_str

    looks_like_local_path = (
        any(sep in model_name_or_path for sep in ("/", "\\"))
        or model_name_or_path.startswith(".")
    )
    if looks_like_local_path:
        searched_paths = "\n".join(f"- {path}" for path in seen)
        raise FileNotFoundError(
            "Local model path was not found. Please check --model_name_or_path.\n"
            f"Input: {model_name_or_path}\nSearched:\n{searched_paths}"
        )

    # assume it is a HuggingFace Hub repo id
    return model_name_or_path
