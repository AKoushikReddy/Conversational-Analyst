# tools_core.py — clean MCP-friendly data tools (no heuristics, JSON-safe returns)

import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import warnings

# Silence pandas/plotly warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# === Global runtime context ===
STATE: Dict[str, Any] = {
    "df": None,       # active DataFrame
    "name": None,     # dataset name
    "schema": None,   # LLM-provided schema
}

# ---- Data I/O ----
def load_data(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load a CSV file. If no path is provided, it looks for a default file named 'dataset.csv'
    in the current working directory.
    """
    if path is None:
        default = Path("dataset.csv")
        if not default.exists():
            return {"ok": False, "error": "no_path_provided_and_dataset.csv_not_found"}
        path = str(default)

    try:
        df = pd.read_csv(path)
        STATE["df"] = df
        STATE["name"] = Path(path).name
        return {
            "ok": True,
            "dataset": STATE["name"],
            "rows": len(df),
            "cols": df.shape[1],
            "columns": list(df.columns),
        }
    except Exception as e:
        return {"ok": False, "error": f"load_failed:{type(e).__name__}:{e}"}


def set_schema(schema: Dict[str, List[str]]) -> Dict[str, Any]:
    """Store the schema provided by the LLM."""
    STATE["schema"] = schema
    return {"ok": True, "keys": list(schema.keys())}


def get_schema() -> Dict[str, Any]:
    """Return the current schema."""
    return {"ok": STATE["schema"] is not None, "schema": STATE["schema"]}


# ---- Summaries ----
def summary(numeric_only: bool = True) -> Dict[str, Any]:
    """Return dataset size and describe() summary."""
    df = STATE["df"]
    if df is None:
        return {"ok": False, "error": "no_dataset_loaded"}

    try:
        if numeric_only:
            desc = df.describe().T
        else:
            desc = df.describe(include="all").T
        return {
            "ok": True,
            "rows": len(df),
            "cols": df.shape[1],
            "describe": desc.to_dict(),  # ✅ convert DataFrame → dict
        }
    except Exception as e:
        return {"ok": False, "error": f"summary_failed:{type(e).__name__}:{e}"}


def head(n: int = 5) -> Dict[str, Any]:
    """Return first n rows as JSON."""
    df = STATE["df"]
    if df is None:
        return {"ok": False, "error": "no_dataset_loaded"}
    return {"ok": True, "rows": df.head(n).to_dict(orient="records")}


# ---- Categorical (LLM decides which columns) ----
def top_categories(columns: List[str], top_n: int = 10) -> Dict[str, Any]:
    df = STATE["df"]
    if df is None:
        return {"ok": False, "error": "no_dataset_loaded"}

    out = {}
    for c in columns:
        try:
            counts = df[c].astype(str).value_counts(dropna=True).head(top_n)
            out[c] = counts.to_dict()
            fig = px.bar(
                x=list(out[c].keys()),
                y=list(out[c].values()),
                title=f"Top {top_n}: {c}",
            )
            fig.update_layout(xaxis_tickangle=-30)
            fig.show()
        except Exception as e:
            out[c] = f"error:{e}"

    return {"ok": True, "top_categories": out}


# ---- Correlations (LLM supplies the exact columns or pairs) ----
def correlations(columns: Optional[List[str]] = None, method: str = "pearson") -> Dict[str, Any]:
    df = STATE["df"]
    if df is None:
        return {"ok": False, "error": "no_dataset_loaded"}

    try:
        if columns is None:
            corr = df.corr(numeric_only=True, method=method)
        else:
            corr = df[columns].corr(method=method)
        return {"ok": True, "method": method, "corr": corr.to_dict()}
    except Exception as e:
        return {"ok": False, "error": f"correlation_failed:{e}"}


def scatter_pairs(pairs: List[Tuple[str, str]]) -> Dict[str, Any]:
    df = STATE["df"]
    if df is None:
        return {"ok": False, "error": "no_dataset_loaded"}

    rendered = []
    for x, y in pairs:
        try:
            fig = px.scatter(df, x=x, y=y, trendline="ols", title=f"{x} vs {y}")
            fig.show()
            rendered.append(f"{x}_vs_{y}")
        except Exception as e:
            rendered.append(f"{x}_vs_{y}_error:{e}")

    return {"ok": True, "pairs_rendered": rendered}


# ---- Outliers (LLM chooses columns & threshold) ----
def outliers(columns: List[str], z: float = 3.0) -> Dict[str, Any]:
    df = STATE["df"]
    if df is None:
        return {"ok": False, "error": "no_dataset_loaded"}

    try:
        sub = df[columns].select_dtypes(include=np.number)
        zscores = (sub - sub.mean()) / (sub.std(ddof=0) + 1e-9)
        mask = (zscores.abs() > z).any(axis=1)
        indices = df.index[mask].tolist()
        return {"ok": True, "count": len(indices), "indices": indices}
    except Exception as e:
        return {"ok": False, "error": f"outlier_failed:{e}"}


# ---- Missing values (LLM supplies threshold) ----
def missing(threshold: float = 0.20) -> Dict[str, Any]:
    df = STATE["df"]
    if df is None:
        return {"ok": False, "error": "no_dataset_loaded"}

    miss = df.isna().mean()
    flagged = miss[miss > threshold].sort_values(ascending=False)
    return {
        "ok": True,
        "threshold": threshold,
        "missing": flagged.to_dict(),
    }


# ---- Distributions / XY plots (LLM specifies what to plot) ----
def plot_hist(column: str, nbins: int = 30) -> Dict[str, Any]:
    df = STATE["df"]
    if df is None:
        return {"ok": False, "error": "no_dataset_loaded"}

    try:
        fig = px.histogram(df, x=column, nbins=nbins, title=f"Distribution: {column}")
        fig.show()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": f"plot_hist_failed:{e}"}


def plot_xy(x: str, y: str) -> Dict[str, Any]:
    df = STATE["df"]
    if df is None:
        return {"ok": False, "error": "no_dataset_loaded"}

    try:
        fig = px.scatter(df, x=x, y=y, trendline="ols", title=f"{x} vs {y}")
        fig.show()
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": f"plot_xy_failed:{e}"}


# ---- Time trends ----
def time_trend(column: str, freq: str = "M") -> Dict[str, Any]:
    df = STATE["df"]
    if df is None:
        return {"ok": False, "error": "no_dataset_loaded"}

    s = df[column]

    try:
        if pd.api.types.is_datetime64_any_dtype(s):
            grp = s.dropna().dt.to_period(freq).value_counts().sort_index()
            result = grp.to_dict()
            fig = px.line(
                x=list(result.keys()),
                y=list(result.values()),
                labels={"x": column, "y": "count"},
                title=f"Count by {freq}: {column}",
            )
            fig.show()
            return {"ok": True, "type": "datetime", "trend": result}

        ser = pd.to_numeric(s, errors="coerce").dropna().astype(int)
        grp = ser.value_counts().sort_index()
        result = grp.to_dict()
        fig = px.line(
            x=list(result.keys()),
            y=list(result.values()),
            labels={"x": column, "y": "count"},
            title=f"Count by Year: {column}",
        )
        fig.show()
        return {"ok": True, "type": "year-int", "trend": result}
    except Exception as e:
        return {"ok": False, "error": f"time_trend_failed:{e}"}
