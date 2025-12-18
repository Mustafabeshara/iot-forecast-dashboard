import io
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

# --- Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
UPLOADS_DIR = BASE_DIR / "uploads"


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_excel(upload) -> Path:
    """Save an uploaded Excel file into data/ and return the path."""
    target = DATA_DIR / upload.name
    with open(target, "wb") as f:
        f.write(upload.getbuffer())
    return target


@st.cache_data(show_spinner=False)
def load_excel(path: Path) -> pd.DataFrame:
    """Load an Excel file with pandas (openpyxl engine)."""
    df = pd.read_excel(path, engine="openpyxl")
    # Try basic dtype normalization: parse datetimes where possible
    for col in df.columns:
        if df[col].dtype == object:
            # Attempt datetime parsing on object columns; ignore errors
            parsed = pd.to_datetime(df[col], errors="ignore")
            df[col] = parsed
    return df


def list_data_files() -> List[Path]:
    return sorted([p for p in DATA_DIR.glob("*.xlsx") if p.is_file()])


def global_search(df: pd.DataFrame, query: str) -> pd.DataFrame:
    if not query:
        return df
    query_lower = query.lower()
    # Build a boolean mask over all columns by stringifying
    mask = pd.Series(False, index=df.index)
    for col in df.columns:
        col_vals = df[col].astype(str).str.lower()
        mask |= col_vals.str.contains(query_lower, na=False)
    return df[mask]


def build_filters(df: pd.DataFrame) -> Dict[str, Tuple[str, object]]:
    """Return filter configuration collected from the UI.
    Returns a dict mapping column -> (type, value) where type in {"categorical", "numeric", "datetime"}.
    """
    filters: Dict[str, Tuple[str, object]] = {}

    with st.expander("Column Filters", expanded=False):
        for col in df.columns:
            series = df[col]
            if pd.api.types.is_categorical_dtype(series) or pd.api.types.is_object_dtype(series):
                # Treat as categorical
                options = sorted(series.dropna().astype(str).unique().tolist())
                selected = st.multiselect(f"{col} (categorical)", options)
                if selected:
                    filters[col] = ("categorical", set(selected))
            elif pd.api.types.is_numeric_dtype(series):
                min_v = float(series.min()) if not series.dropna().empty else 0.0
                max_v = float(series.max()) if not series.dropna().empty else 0.0
                if min_v > max_v:
                    min_v, max_v = max_v, min_v
                sel_min, sel_max = st.slider(
                    f"{col} (numeric range)", min_value=min_v, max_value=max_v, value=(min_v, max_v)
                )
                filters[col] = ("numeric", (sel_min, sel_max))
            elif pd.api.types.is_datetime64_any_dtype(series):
                # Date range filter
                # Convert to date for UI
                sdates = pd.to_datetime(series, errors="coerce").dt.date
                min_d = sdates.min() if not sdates.dropna().empty else None
                max_d = sdates.max() if not sdates.dropna().empty else None
                if min_d and max_d:
                    sel = st.date_input(f"{col} (date range)", value=(min_d, max_d))
                    if isinstance(sel, tuple) and len(sel) == 2:
                        filters[col] = ("datetime", sel)
    return filters


def apply_filters(df: pd.DataFrame, filters: Dict[str, Tuple[str, object]]) -> pd.DataFrame:
    out = df.copy()
    for col, (ftype, val) in filters.items():
        if ftype == "categorical":
            # Compare stringified values
            out = out[out[col].astype(str).isin(val)]
        elif ftype == "numeric":
            low, high = val
            out = out[out[col].astype(float).between(float(low), float(high), inclusive="both")]
        elif ftype == "datetime":
            start, end = val
            series = pd.to_datetime(out[col], errors="coerce")
            out = out[(series.dt.date >= start) & (series.dt.date <= end)]
    return out


def get_key_column(df: pd.DataFrame) -> str:
    key_col = st.selectbox("Key Column (unique row identifier)", options=df.columns.tolist())
    return key_col


def select_row_key(df: pd.DataFrame, key_col: str) -> str:
    key_values = df[key_col].astype(str).unique().tolist()
    selected = st.selectbox("Select Row Key", options=key_values)
    return selected


def list_attachments(row_key: str) -> List[Path]:
    key_dir = UPLOADS_DIR / str(row_key)
    if not key_dir.exists():
        return []
    return sorted([p for p in key_dir.iterdir() if p.is_file()])


def save_attachments(row_key: str, files: List[io.BytesIO], names: List[str]) -> None:
    key_dir = UPLOADS_DIR / str(row_key)
    key_dir.mkdir(parents=True, exist_ok=True)
    for fobj, name in zip(files, names):
        target = key_dir / name
        with open(target, "wb") as f:
            f.write(fobj.getbuffer())


def main() -> None:
    ensure_dirs()

    st.set_page_config(page_title="Excel Search & Attachments", layout="wide")
    st.title("Excel Dashboard: Search, Filter, and Attach Documents")

    # Sidebar: pick or upload Excel
    with st.sidebar:
        st.header("Data Source")
        data_files = list_data_files()
        selected_name = None

        if data_files:
            selected_name = st.selectbox("Select Excel file", options=[p.name for p in data_files])
        uploaded = st.file_uploader(
            "Or upload Excel (.xlsx)", type=["xlsx"], accept_multiple_files=False
        )
        if uploaded is not None:
            saved = save_uploaded_excel(uploaded)
            st.success(f"Uploaded and saved: {saved.name}")
            selected_name = saved.name

    if not selected_name:
        st.info("Select or upload an Excel file from the sidebar.")
        return

    excel_path = DATA_DIR / selected_name
    try:
        df = load_excel(excel_path)
    except Exception as e:
        st.error(f"Failed to load Excel: {e}")
        return

    st.subheader("Search & Filters")
    query = st.text_input("Global search (matches any column)")
    df_q = global_search(df, query)

    filters = build_filters(df_q)
    df_f = apply_filters(df_q, filters)

    # Show table and export
    st.subheader("Results")
    st.caption(f"Rows: {len(df_f)} of {len(df)}")
    st.dataframe(df_f, use_container_width=True)

    # Export filtered CSV
    csv_bytes = df_f.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download filtered CSV", csv_bytes, file_name="filtered.csv", mime="text/csv"
    )

    # Attachments section
    st.subheader("Row Attachments")
    key_col = get_key_column(df_f if not df_f.empty else df)
    if df_f.empty:
        st.info("No rows after filters; you can still pick a row key from the original data.")
    row_key = select_row_key(df_f if not df_f.empty else df, key_col)

    st.write(f"Selected key: **{row_key}**")

    # Upload files to selected row
    up_files = st.file_uploader(
        "Upload documents (PDF, images, DOCX, etc.)", accept_multiple_files=True
    )
    if up_files:
        save_attachments(row_key, up_files, [f.name for f in up_files])
        st.success(f"Uploaded {len(up_files)} file(s) for key '{row_key}'.")

    # List existing attachments
    existing = list_attachments(row_key)
    if existing:
        st.write("Existing attachments:")
        for p in existing:
            with open(p, "rb") as f:
                st.download_button(label=f"Download {p.name}", data=f.read(), file_name=p.name)
    else:
        st.info("No attachments found for this row.")


if __name__ == "__main__":
    main()
