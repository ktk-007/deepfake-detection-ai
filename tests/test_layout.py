from unmask.common import PROJECT_ROOT


def test_package_layout():
    for sub in ["data", "models", "train", "eval", "explain", "app"]:
        assert (PROJECT_ROOT / "unmask" / sub / "__init__.py").exists()
