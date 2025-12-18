import io
from pathlib import Path
import tempfile

import pandas as pd

import app as appmod


def test_global_search_matches_any_column():
    df = pd.DataFrame(
        {
            "A": ["foo", "bar"],
            "B": ["baz", "qux"],
            "C": [1, 2],
        }
    )
    out = appmod.global_search(df, "ba")
    # rows containing 'ba' in any column -> 'bar' and 'baz' both present on row index 1 and 0
    assert set(out.index.tolist()) == {0, 1}


def test_apply_filters_categorical_and_numeric():
    df = pd.DataFrame(
        {
            "cat": ["x", "y", "x", "z"],
            "val": [1, 5, 10, 20],
        }
    )
    filters = {
        "cat": ("categorical", {"x", "z"}),
        "val": ("numeric", (5, 20)),
    }
    out = appmod.apply_filters(df, filters)
    # cat in {x,z} and val between 5 and 20 -> rows index 2 (x,10) and 3 (z,20)
    assert out.reset_index(drop=True).to_dict(orient="list") == {
        "cat": ["x", "z"],
        "val": [10, 20],
    }


def test_apply_filters_datetime_range():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-10", "2024-02-01"]).date,
            "v": [1, 2, 3],
        }
    )
    filters = {
        "date": (
            "datetime",
            (pd.to_datetime("2024-01-05").date(), pd.to_datetime("2024-01-31").date()),
        ),
    }
    out = appmod.apply_filters(df, filters)
    # only 2024-01-10 remains
    assert out["v"].tolist() == [2]


def test_save_attachments_uses_row_key_directory(tmp_path: Path):
    # point uploads dir to a temp directory
    original = appmod.UPLOADS_DIR
    appmod.UPLOADS_DIR = tmp_path
    try:
        row_key = "ROW123"
        files = [io.BytesIO(b"hello"), io.BytesIO(b"world")]
        names = ["a.txt", "b.txt"]
        appmod.save_attachments(row_key, files, names)
        key_dir = tmp_path / row_key
        assert key_dir.exists()
        assert sorted(p.name for p in key_dir.iterdir()) == ["a.txt", "b.txt"]
        assert (key_dir / "a.txt").read_bytes() == b"hello"
    finally:
        appmod.UPLOADS_DIR = original


def test_load_excel_roundtrip(tmp_path: Path):
    df = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
    xlsx = tmp_path / "sample.xlsx"
    df.to_excel(xlsx, index=False)
    out = appmod.load_excel(xlsx)
    assert out.shape == (2, 2)
    assert out["id"].tolist() == [1, 2]
