import argparse
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))


def test_resolve_views_defaults_to_singleview():
    from experiments.run_local_hr_benchmark import DEFAULT_MULTIVIEW_VIEWS, resolve_views

    args = argparse.Namespace(use_multiview=False, views=DEFAULT_MULTIVIEW_VIEWS)
    assert resolve_views(args, argv=[]) == ["identity"]


def test_resolve_views_uses_multiview_only_with_flag():
    from experiments.run_local_hr_benchmark import resolve_views

    args = argparse.Namespace(use_multiview=True, views=["identity", "gaussian"])
    assert resolve_views(args, argv=["--use-multiview", "--views", "identity", "gaussian"]) == [
        "identity",
        "gaussian",
    ]


def test_resolve_views_ignores_views_without_flag(capsys):
    from experiments.run_local_hr_benchmark import resolve_views

    args = argparse.Namespace(use_multiview=False, views=["identity", "gaussian"])
    assert resolve_views(args, argv=["--views", "identity", "gaussian"]) == ["identity"]
    assert "--views ignored" in capsys.readouterr().out
