"""Zero-dependency test runner for hermes-laya-tools."""

import inspect
import sys
import traceback
from pathlib import Path

# Add project root to sys.path
root = Path(__file__).parent
sys.path.insert(0, str(root))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


class MonkeyPatch:
    def __init__(self):
        self._undo = []

    def setenv(self, key, value):
        import os
        old = os.environ.get(key)
        os.environ[key] = str(value)
        self._undo.append((key, old))

    def delenv(self, key, raising=True):
        import os
        if key in os.environ:
            old = os.environ.pop(key)
            self._undo.append((key, old))

    def finish(self):
        import os
        for k, old in reversed(self._undo):
            if old is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old


def run_all():
    import tempfile
    import shutil
    import os

    # Ensure test suite runs in isolation from ambient host config (~/.hermes/config.yaml)
    empty_cfg = tempfile.NamedTemporaryFile(delete=False, suffix="_hermes_cfg.yaml")
    empty_cfg.write(b"{}\n")
    empty_cfg.close()
    old_cfg_env = os.environ.get("HERMES_CONFIG_PATH")
    os.environ["HERMES_CONFIG_PATH"] = empty_cfg.name

    test_files = sorted(root.glob("tests/test_*.py"))
    passed = 0
    failed = 0
    total = 0

    print(f"Running hermes-laya-tools tests across {len(test_files)} files...\n")

    try:
        for tf in test_files:
            mod_name = tf.stem
            # Load module
            import importlib.util
            spec = importlib.util.spec_from_file_location(mod_name, str(tf))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)

            test_funcs = [
                (name, getattr(mod, name))
                for name in dir(mod)
                if name.startswith("test_") and callable(getattr(mod, name))
            ]

            print(f"[{tf.name}] ({len(test_funcs)} tests)")
            for name, fn in test_funcs:
                total += 1
                sig = inspect.signature(fn)
                kwargs = {}
                temp_dir = None
                mp = None

                if "tmp_path" in sig.parameters:
                    temp_dir = Path(tempfile.mkdtemp(prefix="test_laya_"))
                    kwargs["tmp_path"] = temp_dir

                if "monkeypatch" in sig.parameters:
                    mp = MonkeyPatch()
                    kwargs["monkeypatch"] = mp

                try:
                    fn(**kwargs)
                    passed += 1
                    print(f"  [PASS] {name}")
                except Exception as e:
                    failed += 1
                    print(f"  [FAIL] {name}: {e}")
                    traceback.print_exc()
                finally:
                    if mp:
                        mp.finish()
                    if temp_dir and temp_dir.is_dir():
                        shutil.rmtree(temp_dir, ignore_errors=True)

            print()

    finally:
        if old_cfg_env is not None:
            os.environ["HERMES_CONFIG_PATH"] = old_cfg_env
        else:
            os.environ.pop("HERMES_CONFIG_PATH", None)
        try:
            os.unlink(empty_cfg.name)
        except OSError:
            pass

    print(f"{'='*50}")
    print(f"Total: {total} | Passed: {passed} | Failed: {failed}")
    if failed > 0:
        sys.exit(1)
    else:
        print("ALL TESTS PASSED!")


if __name__ == "__main__":
    run_all()
