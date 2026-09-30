"""Canonical filesystem locations used by ComfyUI Local GGUF LLM.

ComfyUI does not currently ship a core LLM model category, but the established
ComfyUI llama/LLM-node convention is the ``LLM`` model-folder key backed by
``models/LLM``.  Keep every package-owned LLM asset below that same root so
case-sensitive filesystems do not end up with parallel ``llm`` and ``LLM``
trees.
"""

from __future__ import annotations

from pathlib import Path

import folder_paths


LLM_FOLDER_KEY = "LLM"
LLM_MODEL_DIR = Path(folder_paths.models_dir) / LLM_FOLDER_KEY
PRESET_ROOT_DIR = LLM_MODEL_DIR / "local_LLM_presets"


def register_llm_model_folder() -> None:
    """Register the canonical GGUF directory with ComfyUI.

    ``add_model_folder_path`` composes with another custom node that already
    registered the ``LLM`` category instead of replacing its paths.  Marking
    this path as default puts the conventional ``models/LLM`` location first.
    """

    LLM_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    folder_paths.add_model_folder_path(LLM_FOLDER_KEY, str(LLM_MODEL_DIR), is_default=True)

    # ComfyUI initializes a newly-added model category with an empty extension
    # set.  Existing third-party registrations may use either a mutable set or
    # another collection, so normalize only when necessary.
    paths, extensions = folder_paths.folder_names_and_paths[LLM_FOLDER_KEY]
    if isinstance(extensions, set):
        extensions.add(".gguf")
    elif ".gguf" not in extensions:
        folder_paths.folder_names_and_paths[LLM_FOLDER_KEY] = (paths, set(extensions) | {".gguf"})
