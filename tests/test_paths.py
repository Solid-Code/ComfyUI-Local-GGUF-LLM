from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest

ROOT = Path(__file__).resolve().parents[1]
PATHS_FILE = ROOT / "paths.py"


def load_paths_module(models_dir: Path, initial=None):
    fake = types.ModuleType("folder_paths")
    fake.models_dir = str(models_dir)
    fake.folder_names_and_paths = dict(initial or {})

    def add_model_folder_path(folder_name: str, full_folder_path: str, is_default: bool = False):
        if folder_name in fake.folder_names_and_paths:
            paths, exts = fake.folder_names_and_paths[folder_name]
            if full_folder_path in paths:
                if is_default and paths[0] != full_folder_path:
                    paths.remove(full_folder_path)
                    paths.insert(0, full_folder_path)
            elif is_default:
                paths.insert(0, full_folder_path)
            else:
                paths.append(full_folder_path)
        else:
            fake.folder_names_and_paths[folder_name] = ([full_folder_path], set())

    fake.add_model_folder_path = add_model_folder_path

    prior = sys.modules.get("folder_paths")
    sys.modules["folder_paths"] = fake
    try:
        name = f"_local_llm_paths_test_{id(fake)}"
        spec = importlib.util.spec_from_file_location(name, PATHS_FILE)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module, fake
    finally:
        if prior is None:
            sys.modules.pop("folder_paths", None)
        else:
            sys.modules["folder_paths"] = prior


class PathContractTests(unittest.TestCase):
    def test_canonical_model_and_preset_roots_use_same_uppercase_llm_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            module, _ = load_paths_module(Path(temp))
            self.assertEqual(module.LLM_FOLDER_KEY, "LLM")
            self.assertEqual(module.LLM_MODEL_DIR, Path(temp) / "LLM")
            self.assertEqual(module.PRESET_ROOT_DIR, Path(temp) / "LLM" / "local_LLM_presets")

    def test_registration_composes_with_existing_llm_category_and_adds_gguf(self):
        with tempfile.TemporaryDirectory() as temp:
            extra = str(Path(temp) / "external_llm")
            module, fake = load_paths_module(
                Path(temp),
                initial={"LLM": ([extra], [".bin"])},
            )
            module.register_llm_model_folder()

            paths, extensions = fake.folder_names_and_paths["LLM"]
            self.assertEqual(paths[0], str(Path(temp) / "LLM"))
            self.assertIn(extra, paths)
            self.assertIn(".gguf", extensions)
            self.assertIn(".bin", extensions)
            self.assertTrue((Path(temp) / "LLM").is_dir())


if __name__ == "__main__":
    unittest.main()
