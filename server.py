# server.py — simple MCP server exposing your tools with plain functions

from mcp.server.fastmcp import FastMCP
from typing import Dict, List, Tuple, Any, Optional

# Import YOUR normal functions (unchanged) from tools_core.py
from tools_core import (
    load_data,
    set_schema,
    get_schema,
    summary,
    head,
    top_categories,
    correlations,
    scatter_pairs,
    outliers,
    missing,
    plot_hist,
    plot_xy,
    time_trend,
)

# Create the MCP server
mcp = FastMCP("conversational-data-analyst")

# ---- Plain wrapper functions (normal def). The decorator only EXPOSES them to MCP. ----
@mcp.tool()
def tool_load_data(path):
    """Load a CSV by path and set global STATE."""
    return load_data(path)

@mcp.tool()
def tool_set_schema(schema: Dict[str, List[str]]):
    """Set LLM-provided schema buckets."""
    return set_schema(schema)

@mcp.tool()
def tool_get_schema():
    """Return the current schema stored in STATE."""
    return get_schema()

@mcp.tool()
def tool_summary(numeric_only=True):
    """Return pandas describe() for dataset."""
    return summary(numeric_only)

@mcp.tool()
def tool_head(n=5):
    """Return first n rows."""
    return head(n)

@mcp.tool()
def tool_top_categories(columns: List[str], top_n: int = 10):
    """Top-N category counts + bar charts."""
    return top_categories(columns, top_n)

@mcp.tool()
def tool_correlations(columns: Optional[List[str]] = None, method: str = "pearson"):
    """Correlation matrix."""
    return correlations(columns, method)

@mcp.tool()
def tool_scatter_pairs(pairs: List[Tuple[str, str]]):
    """Scatter plots with trendlines for each (x,y) pair."""
    return scatter_pairs(pairs)

@mcp.tool()
def tool_outliers(columns: List[str], z: float = 3.0):
    """Row indices where |z| > threshold in provided columns."""
    return outliers(columns, z)

@mcp.tool()
def tool_missing(threshold: float = 0.20):
    """Columns exceeding missing threshold."""
    return missing(threshold)

@mcp.tool()
def tool_plot_hist(column: str, nbins: int = 30):
    """Histogram for a column."""
    return plot_hist(column, nbins)

@mcp.tool()
def tool_plot_xy(x: str, y: str):
    """Scatter x vs y with trendline."""
    return plot_xy(x, y)

@mcp.tool()
def tool_time_trend(column: str, freq: str = "M"):
    """Time trend from datetime or year-like column."""
    return time_trend(column, freq)

if __name__ == "__main__":
    # Start the MCP server over stdio so any MCP client can connect
    mcp.run(transport="stdio")
