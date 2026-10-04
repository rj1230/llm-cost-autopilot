"""
LLM Cost Autopilot — complete single-file Streamlit UI.

Run:
    streamlit run src/ui.py
"""

from __future__ import annotations

import html
import json
import os
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests as http_requests
import streamlit as st

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parents[1]
DB_PATH = ROOT_DIR / "data" / "requests.db"
BENCHMARK_PATH = ROOT_DIR / "benchmark" / "results" / "latest.json"

sys.path.insert(0, str(ROOT_DIR))

DEFAULT_QUALITY_THRESHOLD = 0.80
REFRESH_INTERVAL_SECONDS = 10
MAX_AUDIT_ROWS = 5_000

API_BASE_URL = os.getenv(
    "AUTOPILOT_API_BASE_URL",
    "http://localhost:8000",
).rstrip("/")

API_TIMEOUT_SECONDS = 120


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


CLOUD_MODE = env_flag("AUTOPILOT_CLOUD_MODE", False)

CHART_COLORWAY = [
    "#f2a93c",
    "#4ea8de",
    "#34d399",
    "#f2545b",
    "#838ca0",
]


# ---------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------

st.set_page_config(
    page_title="LLM Cost Autopilot",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------

THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

:root {
    --bg: #080b11;
    --surface: #10141d;
    --surface-raised: #151a25;
    --border: rgba(148, 163, 184, 0.16);
    --border-strong: rgba(148, 163, 184, 0.30);
    --text: #e9edf4;
    --text-muted: #8b95a7;
    --text-faint: #6d7788;
    --accent: #f2a93c;
    --success: #34d399;
    --warning: #f2a93c;
    --danger: #f2545b;
    --info: #4ea8de;
    --radius-sm: 8px;
    --radius-md: 10px;
    --radius-lg: 14px;
    --shadow: 0 18px 45px rgba(0, 0, 0, 0.24);
}

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    color: var(--text);
}

code, pre, .mono, .metric-card-value, .readout-value,
.hero-meta strong, .status-pill {
    font-family: 'IBM Plex Mono', 'SFMono-Regular', Consolas, monospace;
    font-variant-numeric: tabular-nums;
}

.stApp {
    background:
        radial-gradient(circle at 10% -8%, rgba(242, 169, 60, 0.10), transparent 35%),
        var(--bg);
    color: var(--text);
}

.block-container {
    max-width: 1480px;
    padding: 2.4rem 2.75rem 3rem;
}

h1, h2, h3, h4 {
    letter-spacing: -0.025em;
    color: var(--text);
}

h1 { font-size: 2rem; font-weight: 700; }
h2 { font-size: 1.45rem; font-weight: 650; }
h3 { font-size: 1.15rem; font-weight: 650; }
h4 { font-size: 0.98rem; font-weight: 600; }

p {
    color: var(--text-muted);
    line-height: 1.65;
}

hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 1.75rem 0;
}

#MainMenu, footer, header {
    visibility: hidden;
}

[data-testid="stSidebar"] {
    background: #0a0e15;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1.35rem 2rem;
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 1rem;
}

.sidebar-logo {
    display: grid;
    place-items: center;
    width: 40px;
    height: 40px;
    border-radius: 11px;
    background: var(--accent);
    color: #1b1300;
    font-size: 1.15rem;
    font-weight: 700;
    box-shadow: 0 8px 22px rgba(242, 169, 60, 0.22);
}

.sidebar-title {
    color: #f8fafc;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 0.09em;
}

.sidebar-subtitle {
    margin-top: 0.16rem;
    color: var(--text-muted);
    font-size: 0.71rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.sidebar-description {
    color: var(--text-muted);
    font-size: 0.79rem;
    line-height: 1.55;
}

.sidebar-section {
    margin: 1.65rem 0 0.6rem;
    color: var(--accent);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.13em;
    text-transform: uppercase;
}

.hero {
    padding: 1.75rem 2rem;
    border: 1px solid rgba(242, 169, 60, 0.24);
    border-radius: var(--radius-lg);
    background:
        linear-gradient(135deg, rgba(242, 169, 60, 0.10), rgba(16, 20, 29, 0.97) 45%),
        var(--surface);
    box-shadow: var(--shadow);
    margin-bottom: 2rem;
}

.hero-topline {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 1.65rem;
}

.eyebrow {
    color: var(--accent);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.13em;
    text-transform: uppercase;
}

.hero-content {
    display: flex;
    align-items: flex-end;
    justify-content: space-between;
    gap: 2rem;
}

.hero-brand {
    display: flex;
    align-items: center;
    gap: 1.05rem;
}

.hero-icon {
    display: grid;
    place-items: center;
    width: 54px;
    height: 54px;
    border-radius: 14px;
    background: var(--accent);
    color: #1b1300;
    font-size: 1.55rem;
    box-shadow: 0 10px 28px rgba(242, 169, 60, 0.24);
}

.hero h1 {
    margin: 0;
    color: #f8fafc;
    font-size: 2.15rem;
    font-weight: 700;
}

.hero p {
    margin: 0.45rem 0 0;
    max-width: 720px;
    color: #9aa4b5;
    font-size: 0.96rem;
}

.hero-meta {
    display: flex;
    gap: 1.75rem;
}

.hero-meta div {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    min-width: 115px;
}

.hero-meta span {
    color: var(--text-faint);
    font-size: 0.66rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.hero-meta strong {
    color: var(--text);
    font-size: 0.76rem;
    font-weight: 500;
}

.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    padding: 0.32rem 0.75rem;
    border: 1px solid var(--border-strong);
    border-radius: 999px;
    background: rgba(0, 0, 0, 0.22);
    color: var(--text-muted);
    font-size: 0.66rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    white-space: nowrap;
}

.status-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--text-muted);
}

.status-pill--positive .status-dot {
    background: var(--success);
    animation: pulse-dot 2.4s ease-in-out infinite;
}

.status-pill--warning .status-dot { background: var(--warning); }
.status-pill--danger .status-dot { background: var(--danger); }
.status-pill--info .status-dot { background: var(--info); }

@keyframes pulse-dot {
    0% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0.5); }
    70% { box-shadow: 0 0 0 6px rgba(52, 211, 153, 0); }
    100% { box-shadow: 0 0 0 0 rgba(52, 211, 153, 0); }
}

.section-label {
    margin: 1.7rem 0 0.85rem;
    color: var(--text-muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.section-label::before {
    content: "";
    display: inline-block;
    width: 10px;
    height: 2px;
    margin-right: 0.55rem;
    vertical-align: middle;
    background: var(--accent);
}

.card {
    padding: 1.15rem 1.2rem;
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    background: var(--surface);
}

.card p { color: var(--text-muted); }
.card b { color: var(--text); }

.metric-card {
    position: relative;
    min-height: 118px;
    padding: 1.05rem 1.1rem;
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    background:
        linear-gradient(145deg, rgba(255, 255, 255, 0.035), rgba(255, 255, 255, 0.012)),
        var(--surface);
    overflow: hidden;
}

.metric-card::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: #4b5563;
}

.metric-card--primary::before { background: var(--accent); }
.metric-card--success::before { background: var(--success); }
.metric-card--warning::before { background: var(--warning); }
.metric-card--danger::before { background: var(--danger); }
.metric-card--info::before { background: var(--info); }

.metric-card-label {
    color: var(--text-muted);
    font-size: 0.67rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.metric-card-value {
    margin-top: 0.65rem;
    color: #f8fafc;
    font-size: 1.55rem;
    font-weight: 600;
    overflow-wrap: anywhere;
}

.metric-card--primary .metric-card-value {
    color: var(--accent);
}

.metric-card-detail {
    margin-top: 0.35rem;
    color: var(--text-faint);
    font-size: 0.74rem;
    line-height: 1.4;
}

.readout {
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    background: var(--surface);
    overflow: hidden;
}

.readout-row {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 1rem;
    padding: 0.7rem 0.95rem;
    border-bottom: 1px solid var(--border);
}

.readout-row:last-child { border-bottom: none; }

.readout-label {
    color: var(--text-muted);
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.07em;
    text-transform: uppercase;
}

.readout-value {
    color: var(--text);
    font-size: 0.85rem;
    font-weight: 500;
    text-align: right;
    overflow-wrap: anywhere;
}

.st-key-response-panel {
    min-height: 150px;
    border: 1px solid var(--border) !important;
    border-left: 3px solid var(--accent) !important;
    border-radius: var(--radius-md) !important;
    background: var(--surface) !important;
}

.st-key-response-panel [data-testid="stMarkdownContainer"] {
    color: var(--text);
    line-height: 1.72;
    overflow-wrap: anywhere;
}

.st-key-response-panel code {
    background: rgba(148, 163, 184, 0.16);
    border-radius: 4px;
}

.st-key-routing-reasoning {
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    background: var(--surface) !important;
}

.st-key-routing-reasoning [data-testid="stMarkdownContainer"] {
    color: var(--text-muted);
    line-height: 1.6;
}

[data-baseweb="tab-list"] {
    gap: 0.35rem;
    border-bottom: 1px solid var(--border);
}

[data-baseweb="tab"] {
    height: auto;
    padding: 0.6rem 1rem;
    border-radius: var(--radius-sm) var(--radius-sm) 0 0;
    background: transparent;
    color: var(--text-muted);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.79rem;
}

[data-baseweb="tab"]:hover { color: var(--text); }

[data-baseweb="tab"][aria-selected="true"] {
    background: var(--surface);
    color: var(--text);
}

[data-baseweb="tab-highlight"] {
    background-color: var(--accent) !important;
}

button[kind="primary"] {
    min-height: 42px;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    background: var(--accent) !important;
    color: #1b1300 !important;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 650;
    letter-spacing: 0.02em;
    transition: all 0.18s ease;
}

button[kind="primary"]:hover {
    background: #ffc15c !important;
    transform: translateY(-1px);
    box-shadow: 0 8px 22px rgba(242, 169, 60, 0.20);
}

button[kind="primary"]:disabled {
    opacity: 0.55;
    transform: none;
    box-shadow: none;
}

.stButton > button {
    min-height: 40px;
    transition: all 0.18s ease;
}

textarea,
input {
    border-radius: var(--radius-sm) !important;
}

[data-baseweb="select"] > div {
    border-radius: var(--radius-sm);
}

[data-testid="stSlider"] [role="slider"] {
    background-color: var(--accent) !important;
    border-color: var(--accent) !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    background: var(--surface);
    overflow: hidden;
}

.js-plotly-plot {
    border: 1px solid var(--border);
    border-radius: var(--radius-md);
    background: var(--surface);
    padding: 0.85rem 0.6rem;
}

.empty-state {
    display: grid;
    place-items: center;
    min-height: 180px;
    padding: 1.5rem;
    border: 1px dashed var(--border-strong);
    border-radius: var(--radius-md);
    background: rgba(255, 255, 255, 0.015);
    text-align: center;
}

.empty-state-title {
    color: var(--text);
    font-size: 0.95rem;
    font-weight: 600;
}

.empty-state-detail {
    margin-top: 0.3rem;
    color: var(--text-muted);
    font-size: 0.8rem;
}

.app-footer {
    margin-top: 2.25rem;
    padding-top: 1.15rem;
    border-top: 1px solid var(--border);
    color: var(--text-faint);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.02em;
}

@media (max-width: 1000px) {
    .block-container {
        padding: 1.5rem 1.15rem 2.25rem;
    }

    .hero {
        padding: 1.35rem;
    }

    .hero-content {
        flex-direction: column;
        align-items: flex-start;
    }

    .hero h1 {
        font-size: 1.65rem;
    }

    .hero-meta {
        flex-wrap: wrap;
        gap: 1.1rem;
    }

    .metric-card {
        min-height: 105px;
    }
}
</style>
"""

st.markdown(THEME_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------


def escape_html(value: object) -> str:
    return html.escape(str(value))


def money(value: float) -> str:
    return f"${value:,.6f}"


def money_short(value: float) -> str:
    return f"${value:,.4f}"


def pct(value: float) -> str:
    return f"{value:.1%}"


def safe_rate(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def percentile(frame: pd.DataFrame, quantile: float) -> float:
    if frame.empty or "latency_s" not in frame.columns:
        return 0.0

    values = pd.to_numeric(frame["latency_s"], errors="coerce").dropna()
    return float(values.quantile(quantile)) if not values.empty else 0.0


def style_fig(fig: go.Figure, **layout_kwargs) -> go.Figure:
    fig.update_layout(
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(0, 0, 0, 0)",
        font={
            "family": "IBM Plex Sans, sans-serif",
            "color": "#c7ccd6",
            "size": 12,
        },
        colorway=CHART_COLORWAY,
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        hoverlabel={
            "bgcolor": "#10141d",
            "font_color": "#e9edf4",
            "bordercolor": "rgba(148, 163, 184, 0.30)",
        },
    )

    fig.update_xaxes(
        gridcolor="rgba(148, 163, 184, 0.12)",
        zerolinecolor="rgba(148, 163, 184, 0.18)",
        linecolor="rgba(148, 163, 184, 0.18)",
    )

    fig.update_yaxes(
        gridcolor="rgba(148, 163, 184, 0.12)",
        zerolinecolor="rgba(148, 163, 184, 0.18)",
        linecolor="rgba(148, 163, 184, 0.18)",
    )

    if layout_kwargs:
        fig.update_layout(**layout_kwargs)

    return fig


def render_section_label(label: str) -> None:
    st.markdown(
        f'<div class="section-label">{escape_html(label)}</div>',
        unsafe_allow_html=True,
    )


def render_metric_card(
    label: str,
    value: str,
    detail: str = "",
    accent: str = "neutral",
) -> None:
    markup = (
        f'<div class="metric-card metric-card--{escape_html(accent)}">'
        f'<div class="metric-card-label">{escape_html(label)}</div>'
        f'<div class="metric-card-value">{escape_html(value)}</div>'
        f'<div class="metric-card-detail">{escape_html(detail)}</div>'
        "</div>"
    )

    st.markdown(markup, unsafe_allow_html=True)


def render_status_card(
    title: str,
    value: str,
    detail: str = "",
    accent: str = "neutral",
) -> None:
    render_metric_card(title, value, detail, accent)


def render_routing_summary(rows: list[tuple[str, str]]) -> None:
    rows_markup = "".join(
        '<div class="readout-row">'
        f'<span class="readout-label">{escape_html(label)}</span>'
        f'<span class="readout-value">{escape_html(value)}</span>'
        "</div>"
        for label, value in rows
    )

    st.markdown(
        f'<div class="readout">{rows_markup}</div>',
        unsafe_allow_html=True,
    )


def render_empty_state(title: str, detail: str = "") -> None:
    markup = (
        '<div class="empty-state">'
        "<div>"
        f'<div class="empty-state-title">{escape_html(title)}</div>'
        f'<div class="empty-state-detail">{escape_html(detail)}</div>'
        "</div>"
        "</div>"
    )

    st.markdown(markup, unsafe_allow_html=True)


def render_quality_loop() -> None:
    markup = (
        '<div class="card">'
        "<p><b>1. Route</b><br>"
        "Predict complexity and select the configured tier.</p>"
        "<p><b>2. Verify</b><br>"
        "Check eligible lower-tier responses against the quality reference.</p>"
        "<p><b>3. Escalate</b><br>"
        "Record disagreement, return the stronger result when applicable, and "
        "feed the failure into learning data.</p>"
        "</div>"
    )

    st.markdown(markup, unsafe_allow_html=True)


def render_footer(text: str) -> None:
    st.markdown(
        f'<div class="app-footer">{escape_html(text)}</div>',
        unsafe_allow_html=True,
    )


def render_sidebar_brand() -> None:
    markup = (
        '<div class="sidebar-brand">'
        '<div class="sidebar-logo">⚡</div>'
        "<div>"
        '<div class="sidebar-title">AUTOPILOT</div>'
        '<div class="sidebar-subtitle">Operations Console</div>'
        "</div>"
        "</div>"
        '<div class="sidebar-description">'
        "Monitor routing decisions, inference spend, quality signals, and "
        "provider reliability from one workspace."
        "</div>"
    )

    st.markdown(markup, unsafe_allow_html=True)


def render_sidebar_section(label: str) -> None:
    st.markdown(
        f'<div class="sidebar-section">{escape_html(label)}</div>',
        unsafe_allow_html=True,
    )


def render_hero(
    total_requests: int,
    api_target: str,
    status: str,
    status_label: str,
) -> None:
    markup = (
        '<section class="hero">'
        '<div class="hero-topline">'
        '<div class="eyebrow">AUTOPILOT CONSOLE / OPERATIONS</div>'
        f'<div class="status-pill status-pill--{escape_html(status)}">'
        '<span class="status-dot"></span>'
        f"{escape_html(status_label)}"
        "</div>"
        "</div>"
        '<div class="hero-content">'
        '<div class="hero-brand">'
        '<div class="hero-icon">⚡</div>'
        "<div>"
        "<h1>LLM Cost Autopilot</h1>"
        "<p>Intelligent model routing, quality verification, fallback protection, and cost observability.</p>"
        "</div>"
        "</div>"
        '<div class="hero-meta">'
        "<div>"
        "<span>Environment</span>"
        f"<strong>{'CLOUD' if is_cloud_mode() else 'LOCAL'}</strong>"
        "</div>"
        "<div>"
        "<span>Audit records</span>"
        f"<strong>{total_requests:,}</strong>"
        "</div>"
        "<div>"
        "<span>Service</span>"
        "<strong>FastAPI</strong>"
        "</div>"
        "</div>"
        "</div>"
        "</section>"
    )

    st.markdown(markup, unsafe_allow_html=True)


# ---------------------------------------------------------------------
# Database access
# ---------------------------------------------------------------------


@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS)
def load_requests(db_path: str) -> pd.DataFrame:
    path = Path(db_path)

    if not path.exists():
        return pd.DataFrame()

    try:
        with sqlite3.connect(path) as connection:
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }

            if "request_log" not in tables:
                return pd.DataFrame()

            columns = {
                row[1]
                for row in connection.execute(
                    "PRAGMA table_info(request_log)"
                ).fetchall()
            }

            requested_columns = [
                "request_id",
                "timestamp",
                "prompt_hash",
                "tier",
                "primary_model",
                "routed_model",
                "used_fallback",
                "input_tokens",
                "output_tokens",
                "cost_usd",
                "latency_s",
                "quality_score",
                "escalated",
                "verified",
                "error_type",
                "primary_error_type",
                "circuit_state",
                "classifier_tier",
                "classification_confidence",
                "low_confidence",
            ]

            safe_columns = [column for column in requested_columns if column in columns]

            if not safe_columns:
                return pd.DataFrame()

            frame = pd.read_sql_query(
                f"""
                SELECT {", ".join(safe_columns)}
                FROM request_log
                ORDER BY timestamp DESC
                """,
                connection,
            )

    except sqlite3.Error as exc:
        raise RuntimeError(f"SQLite error: {exc}") from exc

    if frame.empty:
        return frame

    defaults = {
        "request_id": "",
        "timestamp": None,
        "prompt_hash": "",
        "tier": 0,
        "primary_model": "unknown",
        "routed_model": "unknown",
        "used_fallback": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "latency_s": 0.0,
        "quality_score": None,
        "escalated": 0,
        "verified": 0,
        "error_type": None,
        "primary_error_type": None,
        "circuit_state": "unknown",
        "classifier_tier": None,
        "classification_confidence": None,
        "low_confidence": 0,
    }

    for column, default in defaults.items():
        if column not in frame.columns:
            frame[column] = default

    frame["timestamp"] = pd.to_datetime(
        frame["timestamp"],
        errors="coerce",
        utc=True,
    )

    for column in [
        "tier",
        "classifier_tier",
        "input_tokens",
        "output_tokens",
        "cost_usd",
        "latency_s",
        "quality_score",
        "classification_confidence",
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["classifier_tier"] = frame["classifier_tier"].where(
        frame["classifier_tier"].isin([1, 2, 3])
    )

    frame["classification_confidence"] = frame["classification_confidence"].clip(
        lower=0.0, upper=1.0
    )

    for column in ["used_fallback", "escalated", "verified", "low_confidence"]:
        frame[column] = (
            pd.to_numeric(frame[column], errors="coerce")
            .fillna(0)
            .astype(int)
            .astype(bool)
        )

    frame["tier"] = frame["tier"].fillna(0).astype(int)
    frame["cost_usd"] = frame["cost_usd"].fillna(0.0)
    frame["latency_s"] = frame["latency_s"].fillna(0.0)
    frame["total_tokens"] = frame["input_tokens"].fillna(0) + frame[
        "output_tokens"
    ].fillna(0)
    frame["success"] = frame["error_type"].isna() | frame["error_type"].astype(
        str
    ).str.strip().eq("")
    frame["date"] = frame["timestamp"].dt.date

    return frame


@st.cache_data(ttl=REFRESH_INTERVAL_SECONDS)
def load_benchmark(path: str) -> dict:
    benchmark_path = Path(path)

    if not benchmark_path.exists():
        return {}

    try:
        with benchmark_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}


# ---------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------


def sync_cloud_provider_secrets() -> None:
    """Expose Streamlit Cloud secrets to the existing provider config layer."""
    if not is_cloud_mode():
        return

    for name in (
        "MISTRAL_API_KEY",
        "GROQ_API_KEY",
        "GROQ_MODEL",
        "PROVIDER_TIMEOUT_S",
        "CIRCUIT_FAILURE_THRESHOLD",
        "CIRCUIT_COOLDOWN_S",
    ):
        value = get_secret(name)
        if value:
            os.environ[name] = value


def get_secret(name: str) -> str:
    try:
        value = st.secrets.get(name, "")
    except (FileNotFoundError, KeyError):
        value = ""

    if value:
        return str(value).strip()

    return os.getenv(name, "").strip()


def get_api_base_url() -> str:
    return get_secret("AUTOPILOT_API_BASE_URL").rstrip("/") or API_BASE_URL


def is_cloud_mode() -> bool:
    try:
        value = st.secrets.get("AUTOPILOT_CLOUD_MODE", "")
    except (FileNotFoundError, KeyError):
        value = ""

    if value != "":
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    return CLOUD_MODE


def api_error_message(response: http_requests.Response) -> str:
    try:
        payload = response.json()
        detail = payload.get("detail")
        if isinstance(detail, str) and detail:
            return detail
    except ValueError:
        pass

    if response.status_code == 401:
        return "Authentication failed. Check your AUTOPILOT_API_KEY."

    if response.status_code == 503:
        return "The API or selected provider is temporarily unavailable."

    return f"API request failed with HTTP {response.status_code}."


def _direct_completion(prompt: str, wait_for_verification: bool) -> dict:
    """Run the same routing engine used by FastAPI without HTTP."""
    sync_cloud_provider_secrets()
    from src.routing import route_request_with_verification

    result, verification = route_request_with_verification(
        prompt,
        synchronous=wait_for_verification,
    )

    if result.response.error:
        error_type = result.response.error_type or "provider_error"
        raise RuntimeError(
            f"Completion failed after routing and fallback protection: {error_type}."
        )

    if verification is None:
        verification_status = "skipped (tier 3)"
        final_text = result.response.output_text
    elif wait_for_verification:
        verification_status = "escalated" if verification.escalated else "passed"
        final_text = verification.final_response.output_text
    else:
        verification_status = "queued"
        final_text = result.response.output_text

    reasoning = {
        1: "classified as simple - routed to the cheapest model",
        2: "classified as moderate complexity - routed to a mid-tier model",
        3: "classified as complex - routed straight to the highest-quality model",
    }[result.tier]

    if result.classifier_tier != result.tier:
        reasoning += (
            " (low-confidence classifier prediction was promoted "
            "to a safer routing tier)"
        )

    if result.used_fallback:
        reasoning += (
            " (primary model for this tier was unavailable - used the fallback)"
        )

    return {
        "id": result.request_id,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": final_text,
                }
            }
        ],
        "routing": {
            "request_id": result.request_id,
            "tier": result.tier,
            "classifier_tier": result.classifier_tier,
            "classification_confidence": result.classification_confidence,
            "low_confidence": result.low_confidence,
            "selected_model": result.routed_model,
            "reasoning": reasoning,
            "used_fallback": result.used_fallback,
            "cost_usd": result.response.cost_usd,
            "latency_s": result.response.latency_s,
            "verification": verification_status,
        },
    }


def submit_completion(prompt: str, wait_for_verification: bool) -> dict:
    if is_cloud_mode():
        return _direct_completion(
            prompt=prompt,
            wait_for_verification=wait_for_verification,
        )

    api_key = get_secret("AUTOPILOT_API_KEY")

    if not api_key:
        raise RuntimeError(
            "AUTOPILOT_API_KEY is missing. Configure it in "
            ".streamlit/secrets.toml before sending a request."
        )

    response = http_requests.post(
        f"{get_api_base_url()}/v1/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "wait_for_verification": wait_for_verification,
        },
        timeout=API_TIMEOUT_SECONDS,
    )

    if not response.ok:
        raise RuntimeError(api_error_message(response))

    return response.json()


def reset_playground() -> None:
    st.session_state.playground_result = None
    st.session_state.playground_nonce = st.session_state.get("playground_nonce", 0) + 1


# ---------------------------------------------------------------------
# Playground
# ---------------------------------------------------------------------


def render_playground() -> None:
    st.markdown("### Request Playground")
    st.caption(
        "Send a real request through the routing layer. The Autopilot chooses "
        "the lowest-cost configured model that matches predicted complexity."
    )

    cloud_mode = is_cloud_mode()
    api_key_configured = bool(get_secret("AUTOPILOT_API_KEY"))
    playground_ready = cloud_mode or api_key_configured

    status_col, action_col = st.columns([4, 1])

    with status_col:
        if cloud_mode:
            st.success(
                "Connected: `Streamlit Cloud direct mode` — routing runs in-process."
            )
        elif api_key_configured:
            st.success(f"Connected target: `{get_api_base_url()}`")
        else:
            st.warning(
                "API key is not configured. Add `AUTOPILOT_API_KEY` to "
                "`.streamlit/secrets.toml`."
            )

    with action_col:
        st.button(
            "New request",
            use_container_width=True,
            on_click=reset_playground,
        )

    examples = {
        "Choose an example": "",
        "Simple explanation": (
            "Explain the difference between Python lists and tuples in plain language."
        ),
        "Summarization": (
            "Summarize the benefits and risks of using a multi-provider LLM "
            "routing layer in five concise bullet points."
        ),
        "Extraction": (
            "Extract company, product, savings percentage, and use case from: "
            "'Acme reduced inference costs by 38% after routing customer "
            "support questions to a smaller model.'"
        ),
        "Complex reasoning": (
            "Design an evaluation plan for an LLM cost router. Include quality, "
            "cost, latency, fallback, verification, and failure metrics."
        ),
    }

    option_col, verify_col = st.columns([2.2, 1])

    with option_col:
        selected_example = st.selectbox(
            "Try a sample request",
            options=list(examples),
        )

    with verify_col:
        wait_for_verification = st.toggle(
            "Wait for verification",
            value=False,
            help=(
                "Off: return quickly and verify eligible requests in the "
                "background. On: wait for the verification result."
            ),
        )

    st.caption(
        f"DEBUG: cloud_mode={is_cloud_mode()} | "
        f"cloud_secret_present={'AUTOPILOT_CLOUD_MODE' in st.secrets} | "
        f"api_base={get_api_base_url()}"
    )

    prompt_key = f"playground_prompt_{st.session_state.get('playground_nonce', 0)}"

    prompt = st.text_area(
        "Prompt",
        value=examples[selected_example],
        key=prompt_key,
        height=190,
        max_chars=16_000,
        placeholder="Enter a prompt for the LLM Cost Autopilot...",
    )

    sent = st.button(
        "⚡ Send through Autopilot",
        type="primary",
        use_container_width=True,
        disabled=not playground_ready,
    )

    if sent:
        cleaned_prompt = prompt.strip()

        if not cleaned_prompt:
            st.warning("Enter a prompt before sending the request.")
        else:
            with st.spinner(
                "Classifying complexity, selecting a route, and generating "
                "a response..."
            ):
                try:
                    st.session_state.playground_result = submit_completion(
                        prompt=cleaned_prompt,
                        wait_for_verification=wait_for_verification,
                    )
                    load_requests.clear()
                except http_requests.Timeout:
                    st.error(
                        "The request timed out. Try disabling "
                        "wait-for-verification or check provider availability."
                    )
                except http_requests.ConnectionError:
                    if is_cloud_mode():
                        st.error(
                            "The cloud request could not reach the provider. "
                            "Check provider secrets and availability."
                        )
                    else:
                        st.error(
                            "Could not reach FastAPI. Start it with: "
                            "`uvicorn src.api.main:app --reload`"
                        )
                except RuntimeError as exc:
                    st.error(str(exc))
                except http_requests.RequestException as exc:
                    st.error(f"Network request failed: {exc}")

    result = st.session_state.get("playground_result")

    if not result:
        render_empty_state(
            "No response yet",
            "Your answer, route selection, cost, latency, fallback behavior, "
            "and verification state will appear here after you send a request.",
        )
        return

    choices = result.get("choices", [])
    routing = result.get("routing", {})

    if not choices:
        st.error("The API returned no completion choices.")
        return

    answer = str(choices[0].get("message", {}).get("content", "")).strip()

    st.divider()

    answer_col, route_col = st.columns([2.15, 1])

    with answer_col:
        st.markdown("### Generated response")

        if answer:
            with st.container(key="response-panel", border=True):
                st.markdown(answer)
        else:
            st.warning("The API returned an empty response.")

    with route_col:
        st.markdown("### Routing decision")

        tier = routing.get("tier", "—")
        classifier_tier = routing.get("classifier_tier", "—")
        classification_confidence = routing.get(
            "classification_confidence",
            None,
        )
        low_confidence = bool(routing.get("low_confidence", False))
        selected_model = routing.get("selected_model", "Unknown")
        cost_usd = float(routing.get("cost_usd", 0) or 0)
        latency_s = float(routing.get("latency_s", 0) or 0)
        verification = str(routing.get("verification", "unknown"))
        fallback_used = bool(routing.get("used_fallback", False))

        confidence_display = (
            f"{float(classification_confidence):.1%}"
            if classification_confidence is not None
            else "N/A"
        )

        promotion_display = (
            "Promoted by safety policy"
            if str(classifier_tier) != str(tier) and classifier_tier != "—"
            else "No promotion"
        )

        render_routing_summary(
            [
                ("Selected model", str(selected_model)),
                ("Raw ML tier", f"Tier {classifier_tier}"),
                ("Classifier confidence", confidence_display),
                (
                    "Safety policy",
                    "Low confidence" if low_confidence else "Normal confidence",
                ),
                ("Final routing tier", f"Tier {tier}"),
                ("Routing transition", promotion_display),
                ("Request cost", money(cost_usd)),
                ("Latency", f"{latency_s:.3f}s"),
                ("Verification", verification.replace("_", " ").title()),
                (
                    "Fallback protection",
                    "Fallback used" if fallback_used else "Primary route served",
                ),
            ]
        )

    st.markdown("### Why this route?")

    reasoning = str(routing.get("reasoning", "No routing explanation was returned."))
    request_id = routing.get("request_id", result.get("id", "unknown"))

    with st.container(key="routing-reasoning", border=True):
        st.markdown(reasoning)
        st.caption(f"Request ID: `{request_id}`")

    if verification == "queued":
        st.info(
            "Background quality verification is queued. Refresh shortly to see "
            "the resulting record in the Quality and Audit tabs."
        )


# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------

try:
    request_log = load_requests(str(DB_PATH))
except RuntimeError as exc:
    st.error(f"Unable to load the request audit database: {exc}")
    request_log = pd.DataFrame()

benchmark = load_benchmark(str(BENCHMARK_PATH))
database_is_empty = request_log.empty


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------

with st.sidebar:
    render_sidebar_brand()
    st.divider()

    if database_is_empty:
        render_empty_state(
            "No audit records yet",
            "Send your first request from the Playground tab.",
        )
        filtered = request_log.copy()
    else:
        min_timestamp = request_log["timestamp"].min()
        max_timestamp = request_log["timestamp"].max()

        selected_dates = st.date_input(
            "Reporting period",
            value=(min_timestamp.date(), max_timestamp.date()),
            min_value=min_timestamp.date(),
            max_value=max_timestamp.date(),
        )

        filtered = request_log.copy()

        if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
            start_date, end_date = selected_dates
            filtered = filtered[
                (filtered["date"] >= start_date) & (filtered["date"] <= end_date)
            ].copy()

        render_sidebar_section("Filters")

        tier_options = sorted(filtered["tier"].dropna().unique().tolist())
        selected_tiers = st.multiselect(
            "Complexity tier",
            options=tier_options,
            default=tier_options,
        )

        model_options = sorted(
            filtered["routed_model"].dropna().astype(str).unique().tolist()
        )
        selected_models = st.multiselect(
            "Routed model",
            options=model_options,
            default=model_options,
        )

        if selected_tiers:
            filtered = filtered[filtered["tier"].isin(selected_tiers)]
        else:
            filtered = filtered.iloc[0:0]

        if selected_models:
            filtered = filtered[filtered["routed_model"].isin(selected_models)]
        else:
            filtered = filtered.iloc[0:0]

    render_sidebar_section("Quality controls")

    quality_threshold = st.slider(
        "Quality pass threshold",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_QUALITY_THRESHOLD,
        step=0.05,
    )

    render_sidebar_section("Data sources")

    st.caption("FastAPI completion service")
    st.caption("SQLite audit trail")
    st.caption("Provider resilience metadata")
    st.caption("Quality verification records")

    if st.button("Refresh dashboard", use_container_width=True):
        st.cache_data.clear()
        st.rerun()


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

view_requests = len(filtered)
view_cost = float(filtered["cost_usd"].sum()) if not filtered.empty else 0.0
view_success = int(filtered["success"].sum()) if not filtered.empty else 0
view_failures = view_requests - view_success
view_fallbacks = int(filtered["used_fallback"].sum()) if not filtered.empty else 0

view_avg_cost = safe_rate(view_cost, view_requests)
view_fallback_rate = safe_rate(view_fallbacks, view_requests)
view_success_rate = safe_rate(view_success, view_requests)

verified = (
    filtered[filtered["verified"]].copy() if not filtered.empty else pd.DataFrame()
)

quality_scores = (
    pd.to_numeric(verified["quality_score"], errors="coerce").dropna()
    if not verified.empty
    else pd.Series(dtype=float)
)

avg_quality = float(quality_scores.mean()) if not quality_scores.empty else None
quality_pass_rate = (
    float((quality_scores >= quality_threshold).mean())
    if not quality_scores.empty
    else None
)

escalation_count = int(filtered["escalated"].sum()) if not filtered.empty else 0
verified_count = len(quality_scores)
escalation_rate = safe_rate(escalation_count, verified_count)

total_requests = len(request_log)
total_cost = float(request_log["cost_usd"].sum()) if not request_log.empty else 0.0
total_tokens = int(request_log["total_tokens"].sum()) if not request_log.empty else 0
total_success = int(request_log["success"].sum()) if not request_log.empty else 0
total_fallbacks = (
    int(request_log["used_fallback"].sum()) if not request_log.empty else 0
)

# Phase 3D: confidence-aware routing observability.
# Older records may not have classifier metadata, so analytics only
# use rows where the raw classifier tier and confidence are available.
classification_view = (
    filtered[
        filtered["classifier_tier"].notna()
        & filtered["classification_confidence"].notna()
    ].copy()
    if not filtered.empty
    else pd.DataFrame()
)

classification_total = len(classification_view)

avg_classification_confidence = (
    float(classification_view["classification_confidence"].mean())
    if classification_total
    else None
)

low_confidence_count = (
    int(classification_view["low_confidence"].sum()) if classification_total else 0
)

low_confidence_rate = safe_rate(
    low_confidence_count,
    classification_total,
)

promotion_mask = (
    classification_view["classifier_tier"].astype(int)
    != classification_view["tier"].astype(int)
    if classification_total
    else pd.Series(dtype=bool)
)

promotion_count = int(promotion_mask.sum()) if classification_total else 0

promotion_rate = safe_rate(
    promotion_count,
    classification_total,
)

raw_tier_distribution = (
    classification_view["classifier_tier"]
    .astype(int)
    .value_counts()
    .sort_index()
    .to_dict()
    if classification_total
    else {}
)

routing_transitions = (
    (
        classification_view["classifier_tier"].astype(int).astype(str)
        + "_to_"
        + classification_view["tier"].astype(int).astype(str)
    )
    .value_counts()
    .sort_index()
    .to_dict()
    if classification_total
    else {}
)


# ---------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------

if total_requests == 0:
    hero_status, hero_label = "info", "STANDBY"
elif total_success == total_requests:
    hero_status, hero_label = "positive", "NOMINAL"
else:
    hero_status, hero_label = "success", "OPERATIONAL"

render_hero(
    total_requests=total_requests,
    api_target=get_api_base_url(),
    status=hero_status,
    status_label=hero_label,
)


# ---------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------

(
    playground_tab,
    overview_tab,
    routing_tab,
    reliability_tab,
    quality_tab,
    benchmark_tab,
    audit_tab,
) = st.tabs(
    [
        "Playground",
        "Overview",
        "Routing",
        "Reliability",
        "Quality",
        "Benchmark",
        "Audit trail",
    ]
)


# ---------------------------------------------------------------------
# Playground tab
# ---------------------------------------------------------------------

with playground_tab:
    render_playground()


# ---------------------------------------------------------------------
# Overview tab
# ---------------------------------------------------------------------

with overview_tab:
    render_section_label("Selected-period performance")

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    with kpi1:
        render_metric_card(
            "Requests",
            f"{view_requests:,}",
            "Selected period",
        )

    with kpi2:
        render_metric_card(
            "Inference cost",
            money_short(view_cost),
            "Total spend",
            "primary",
        )

    with kpi3:
        render_metric_card(
            "Average cost",
            money(view_avg_cost),
            "Per request",
        )

    with kpi4:
        render_metric_card(
            "Success rate",
            pct(view_success_rate),
            f"{view_failures:,} failed requests",
            "success" if view_success_rate >= 0.99 else "warning",
        )

    with kpi5:
        render_metric_card(
            "Fallback rate",
            pct(view_fallback_rate),
            f"{view_fallbacks:,} fallback requests",
            "warning" if view_fallback_rate else "success",
        )

    render_section_label("Classifier safety observability")

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        render_metric_card(
            "Classifier confidence",
            (
                f"{avg_classification_confidence:.1%}"
                if avg_classification_confidence is not None
                else "N/A"
            ),
            f"{classification_total:,} requests with classifier metadata",
            "info",
        )

    with c2:
        render_metric_card(
            "Low-confidence rate",
            pct(low_confidence_rate),
            f"{low_confidence_count:,} requests below 85% confidence",
            "warning" if low_confidence_count else "success",
        )

    with c3:
        render_metric_card(
            "Safety promotions",
            f"{promotion_count:,}",
            f"{promotion_rate:.1%} of classified requests",
            "warning" if promotion_count else "success",
        )

    with c4:
        render_metric_card(
            "Raw classifier tiers",
            (
                f"{raw_tier_distribution.get(1, 0):,} / "
                f"{raw_tier_distribution.get(2, 0):,} / "
                f"{raw_tier_distribution.get(3, 0):,}"
            ),
            "T1 / T2 / T3 before safety policy",
            "primary",
        )

    render_section_label("Cost and performance")

    chart_left, chart_right = st.columns([1.35, 1])

    with chart_left:
        st.markdown("#### Daily inference cost")

        if not filtered.empty:
            daily = (
                filtered.groupby("date", dropna=True)
                .agg(
                    requests=("request_id", "count"),
                    cost_usd=("cost_usd", "sum"),
                    avg_latency_s=("latency_s", "mean"),
                )
                .reset_index()
                .sort_values("date")
            )

            fig = px.area(
                daily,
                x="date",
                y="cost_usd",
                markers=True,
                labels={"date": "Date", "cost_usd": "Cost (USD)"},
            )

            fig.update_traces(fillcolor="rgba(242, 169, 60, 0.18)")
            style_fig(fig, height=350, hovermode="x unified")

            st.plotly_chart(fig, width="stretch")
        else:
            render_empty_state(
                "No cost data yet",
                "Send a request from the Playground to generate audit data.",
            )

    with chart_right:
        st.markdown("#### Cost by routed model")

        if not filtered.empty:
            by_model = (
                filtered.groupby("routed_model", dropna=False)
                .agg(
                    requests=("request_id", "count"),
                    cost_usd=("cost_usd", "sum"),
                )
                .reset_index()
                .rename(
                    columns={
                        "routed_model": "Model",
                        "cost_usd": "Cost",
                    }
                )
            )

            fig = px.bar(
                by_model,
                x="Model",
                y="Cost",
                text="Cost",
            )

            fig.update_traces(
                texttemplate="$%{text:.6f}",
                textposition="outside",
                textfont_color="#e9edf4",
            )

            style_fig(fig, height=350)

            st.plotly_chart(fig, width="stretch")
        else:
            render_empty_state(
                "No model-cost data available",
                "Model-level cost appears after requests are logged.",
            )

    render_section_label("Operational status")

    status1, status2, status3, status4 = st.columns(4)

    with status1:
        render_status_card(
            "Audit logging",
            "Enabled",
            f"{total_requests:,} requests persisted",
            "info",
        )

    with status2:
        render_status_card(
            "Provider success",
            pct(safe_rate(total_success, total_requests)),
            f"{total_requests - total_success:,} failed requests",
            "success" if total_success == total_requests else "warning",
        )

    with status3:
        render_status_card(
            "Fallback protection",
            pct(safe_rate(total_fallbacks, total_requests)),
            f"{total_fallbacks:,} fallback requests",
            "warning" if total_fallbacks else "success",
        )

    with status4:
        render_status_card(
            "Quality verification",
            f"{avg_quality:.3f}" if avg_quality is not None else "N/A",
            f"{verified_count:,} verified requests in selected view",
            "primary",
        )


# ---------------------------------------------------------------------
# Routing tab
# ---------------------------------------------------------------------

with routing_tab:
    st.markdown("### Routing distribution")

    if filtered.empty:
        render_empty_state(
            "No routing data matches the selected filters",
            "Adjust the sidebar filters or send a request from the Playground.",
        )
    else:
        route_left, route_right = st.columns([1.3, 1])

        distribution = (
            filtered["routed_model"]
            .fillna("unknown")
            .value_counts()
            .rename_axis("Model")
            .reset_index(name="Requests")
        )

        distribution["Share"] = (
            distribution["Requests"] / distribution["Requests"].sum()
        )

        with route_left:
            fig = px.bar(
                distribution,
                x="Requests",
                y="Model",
                orientation="h",
                text="Requests",
            )

            fig.update_traces(
                textposition="outside",
                textfont_color="#e9edf4",
            )

            style_fig(
                fig,
                height=350,
                margin={"l": 10, "r": 35, "t": 10, "b": 10},
            )

            st.plotly_chart(fig, width="stretch")

        with route_right:
            display = distribution.copy()
            display["Share"] = display["Share"].map(lambda value: f"{value:.1%}")

            st.dataframe(
                display,
                width="stretch",
                hide_index=True,
            )

        st.markdown("### Classifier decision observability")

        obs_left, obs_right = st.columns(2)

        with obs_left:
            st.markdown("#### Raw classifier tier distribution")

            if classification_total:
                raw_distribution = pd.DataFrame(
                    [
                        {
                            "Raw tier": f"Tier {tier}",
                            "Requests": raw_tier_distribution.get(tier, 0),
                        }
                        for tier in [1, 2, 3]
                    ]
                )

                fig = px.bar(
                    raw_distribution,
                    x="Raw tier",
                    y="Requests",
                    text="Requests",
                )

                fig.update_traces(
                    textposition="outside",
                    textfont_color="#e9edf4",
                )

                style_fig(fig, height=300)

                st.plotly_chart(fig, width="stretch")
            else:
                render_empty_state(
                    "No classifier metadata",
                    "Classifier observability appears for Phase 3C+ requests.",
                )

        with obs_right:
            st.markdown("#### Raw → final routing transitions")

            if routing_transitions:
                transition_data = pd.DataFrame(
                    [
                        {
                            "Transition": key.replace("_to_", " → "),
                            "Requests": value,
                        }
                        for key, value in routing_transitions.items()
                    ]
                )

                fig = px.bar(
                    transition_data,
                    x="Requests",
                    y="Transition",
                    orientation="h",
                    text="Requests",
                )

                fig.update_traces(
                    textposition="outside",
                    textfont_color="#e9edf4",
                )

                style_fig(
                    fig,
                    height=300,
                    margin={"l": 10, "r": 35, "t": 10, "b": 10},
                )

                st.plotly_chart(fig, width="stretch")
            else:
                render_empty_state(
                    "No routing transitions",
                    "Transitions appear when classifier metadata is available.",
                )

        st.markdown("#### Routing decision chain")

        with st.container(border=True):
            st.markdown(
                """
                **Raw ML prediction**
                → **classifier confidence**
                → **confidence safety policy**
                → **final routing tier**
                → **provider**
                → **fallback protection**
                → **quality verification**
                """
            )

            st.caption(
                "**Confidence safety policy:** "
                "≥85% confidence → retain the classifier tier · "
                "<85% confidence → promote T1/T2 by one tier · "
                "T3 remains T3 when confidence is low."
            )

            if promotion_count:
                st.info(
                    f"{promotion_count:,} classified requests were promoted "
                    "to a safer routing tier because the classifier confidence "
                    "was below the 85% safety threshold."
                )
            else:
                st.caption(
                    "No confidence-driven tier promotions are present in the "
                    "selected view."
                )

        st.markdown("### Complexity-tier traffic")

        tier_data = (
            filtered.groupby("tier")
            .agg(
                requests=("request_id", "count"),
                cost_usd=("cost_usd", "sum"),
                avg_latency_s=("latency_s", "mean"),
                fallback_rate=("used_fallback", "mean"),
            )
            .reset_index()
        )

        tier_data["Tier"] = "Tier " + tier_data["tier"].astype(str)

        st.dataframe(
            tier_data.rename(
                columns={
                    "requests": "Requests",
                    "cost_usd": "Cost",
                    "avg_latency_s": "Avg latency (s)",
                    "fallback_rate": "Fallback rate",
                }
            ),
            width="stretch",
            hide_index=True,
            column_config={
                "Cost": st.column_config.NumberColumn(format="$%.6f"),
                "Avg latency (s)": st.column_config.NumberColumn(format="%.3f"),
                "Fallback rate": st.column_config.ProgressColumn(
                    format="%.1%",
                    min_value=0,
                    max_value=1,
                ),
            },
        )


# ---------------------------------------------------------------------
# Reliability tab
# ---------------------------------------------------------------------

with reliability_tab:
    st.markdown("### Provider reliability")

    if filtered.empty:
        render_empty_state(
            "No reliability data matches the selected filters",
            "Provider health metrics appear once requests are available.",
        )
    else:
        working = filtered.copy()

        def provider_name(model: object) -> str:
            value = str(model or "unknown").strip().lower()

            if "/" in value:
                return value.split("/", 1)[0]

            if "-" in value:
                return value.split("-", 1)[0]

            return value

        working["Provider"] = working["routed_model"].map(provider_name)

        provider_health = (
            working.groupby("Provider")
            .agg(
                Requests=("request_id", "count"),
                Failed=("success", lambda values: int((~values).sum())),
                **{
                    "Avg latency (s)": ("latency_s", "mean"),
                    "Total cost": ("cost_usd", "sum"),
                    "Fallback requests": ("used_fallback", "sum"),
                },
            )
            .reset_index()
        )

        provider_health["Failure rate"] = (
            provider_health["Failed"] / provider_health["Requests"]
        )

        st.dataframe(
            provider_health,
            width="stretch",
            hide_index=True,
            column_config={
                "Failure rate": st.column_config.ProgressColumn(
                    format="%.1%",
                    min_value=0,
                    max_value=1,
                ),
                "Avg latency (s)": st.column_config.NumberColumn(format="%.3f"),
                "Total cost": st.column_config.NumberColumn(format="$%.6f"),
            },
        )

        st.markdown("### Error taxonomy")

        errors = filtered.loc[
            ~filtered["success"],
            "error_type",
        ].fillna("unknown")

        if errors.empty:
            st.success("No provider failures were recorded in this view.")
        else:
            error_data = (
                errors.value_counts()
                .rename_axis("Error type")
                .reset_index(name="Occurrences")
            )

            fig = px.bar(
                error_data,
                x="Occurrences",
                y="Error type",
                orientation="h",
                text="Occurrences",
            )

            fig.update_traces(
                textposition="outside",
                textfont_color="#e9edf4",
                marker_color="#f2545b",
            )

            style_fig(
                fig,
                height=320,
                margin={"l": 10, "r": 35, "t": 10, "b": 10},
            )

            st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------
# Quality tab
# ---------------------------------------------------------------------

with quality_tab:
    st.markdown("### Quality verification")

    q1, q2, q3, q4 = st.columns(4)

    with q1:
        render_metric_card(
            "Verified requests",
            f"{verified_count:,}",
            "Quality checks completed",
        )

    with q2:
        render_metric_card(
            "Average quality",
            f"{avg_quality:.3f}" if avg_quality is not None else "N/A",
            "Across verified requests",
            "primary",
        )

    with q3:
        render_metric_card(
            "Threshold pass rate",
            pct(quality_pass_rate) if quality_pass_rate is not None else "N/A",
            f"Threshold: {quality_threshold:.2f}",
            "success"
            if quality_pass_rate is not None and quality_pass_rate >= 0.8
            else "warning",
        )

    with q4:
        render_metric_card(
            "Escalation rate",
            pct(escalation_rate),
            "Verified requests escalated",
            "warning" if escalation_rate else "success",
        )

    quality_left, quality_right = st.columns(2)

    with quality_left:
        st.markdown("#### Quality-score distribution")

        if quality_scores.empty:
            render_empty_state(
                "No verified quality scores",
                "Quality scores appear after verification completes.",
            )
        else:
            fig = px.histogram(
                x=quality_scores,
                nbins=20,
                labels={"x": "Quality score"},
                color_discrete_sequence=["#4ea8de"],
            )

            fig.add_vline(
                x=quality_threshold,
                line_dash="dash",
                line_color="#f2a93c",
                annotation_text=f"Threshold: {quality_threshold:.2f}",
                annotation_font_color="#e9edf4",
            )

            style_fig(fig, height=350, showlegend=False)

            st.plotly_chart(fig, width="stretch")

    with quality_right:
        st.markdown("#### Quality-control loop")
        render_quality_loop()


# ---------------------------------------------------------------------
# Benchmark tab
# ---------------------------------------------------------------------

with benchmark_tab:
    st.markdown("### Deterministic benchmark")

    if not benchmark:
        render_empty_state(
            "No benchmark result found",
            f"Expected benchmark file: `{BENCHMARK_PATH}`",
        )
    else:
        metrics = benchmark.get("metrics", {})
        benchmark_meta = benchmark.get("benchmark", {})
        cost_metrics = metrics.get("cost_usd", {})
        latency_metrics = metrics.get("latency_s", {})

        benchmark_actual = float(cost_metrics.get("actual", 0))
        benchmark_baseline = float(cost_metrics.get("gpt4o_baseline", 0))
        savings_rate = float(cost_metrics.get("savings_rate", 0))

        b1, b2, b3, b4 = st.columns(4)

        with b1:
            render_metric_card(
                "Benchmark requests",
                f"{metrics.get('requests', 0):,}",
                "Deterministic suite",
            )

        with b2:
            render_metric_card(
                "Routing accuracy",
                pct(float(metrics.get("routing_accuracy", 0))),
                "Correct route selections",
            )

        with b3:
            render_metric_card(
                "Estimated cost reduction",
                pct(savings_rate),
                "Vs configured all-GPT-4o price baseline",
                "primary",
            )

        with b4:
            render_metric_card(
                "Provider calls",
                str(benchmark_meta.get("provider_calls", "unknown")),
                "During benchmark execution",
            )

        comparison = pd.DataFrame(
            {
                "Scenario": ["Autopilot routing", "GPT-4o baseline"],
                "Cost": [benchmark_actual, benchmark_baseline],
            }
        )

        fig = px.bar(
            comparison,
            x="Cost",
            y="Scenario",
            orientation="h",
            text="Cost",
            color="Scenario",
            color_discrete_map={
                "Autopilot routing": "#f2a93c",
                "GPT-4o baseline": "#4a5568",
            },
        )

        fig.update_traces(
            texttemplate="$%{text:.6f}",
            textposition="outside",
            textfont_color="#e9edf4",
        )

        style_fig(
            fig,
            height=310,
            margin={"l": 10, "r": 65, "t": 10, "b": 10},
            showlegend=False,
        )

        st.plotly_chart(fig, width="stretch")

        latency_data = pd.DataFrame(
            [
                {
                    "Metric": "Average",
                    "Seconds": latency_metrics.get("average", 0),
                },
                {"Metric": "P50", "Seconds": latency_metrics.get("p50", 0)},
                {"Metric": "P95", "Seconds": latency_metrics.get("p95", 0)},
                {"Metric": "P99", "Seconds": latency_metrics.get("p99", 0)},
            ]
        )

        st.markdown("#### Benchmark latency")
        st.dataframe(latency_data, width="stretch", hide_index=True)

        st.caption(
            "The GPT-4o figure is a price baseline. The deterministic benchmark "
            "uses mocked provider calls and does not make live provider requests."
        )


# ---------------------------------------------------------------------
# Audit tab
# ---------------------------------------------------------------------

with audit_tab:
    st.markdown("### Request audit trail")
    st.caption(
        "Raw prompts are intentionally never displayed. The database stores "
        "only a privacy-preserving prompt hash."
    )

    if filtered.empty:
        render_empty_state(
            "No requests match the selected filters",
            "Use the Playground to generate new audit records.",
        )
    else:
        audit_columns = [
            "timestamp",
            "request_id",
            "prompt_hash",
            "tier",
            "primary_model",
            "routed_model",
            "used_fallback",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cost_usd",
            "latency_s",
            "quality_score",
            "escalated",
            "verified",
            "error_type",
            "primary_error_type",
            "circuit_state",
            "classifier_tier",
            "classification_confidence",
            "low_confidence",
        ]

        audit_view = filtered[audit_columns].head(MAX_AUDIT_ROWS).copy()

        audit_view["timestamp"] = audit_view["timestamp"].dt.strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )

        audit_view["cost_usd"] = audit_view["cost_usd"].round(8)
        audit_view["latency_s"] = audit_view["latency_s"].round(4)
        audit_view["quality_score"] = audit_view["quality_score"].round(4)
        audit_view["classification_confidence"] = audit_view[
            "classification_confidence"
        ].round(4)

        audit_view = audit_view.rename(
            columns={
                "classifier_tier": "raw_classifier_tier",
                "classification_confidence": "classifier_confidence",
                "low_confidence": "low_confidence_policy",
            }
        )

        st.download_button(
            "Download audit CSV",
            data=audit_view.to_csv(index=False).encode("utf-8"),
            file_name="llm_cost_autopilot_audit.csv",
            mime="text/csv",
        )

        st.dataframe(
            audit_view,
            width="stretch",
            hide_index=True,
            height=560,
        )

    st.markdown("### System architecture")

    with st.container(border=True):
        st.graphviz_chart(
            """
            digraph {
                rankdir=LR;
                bgcolor="#10141d";

                node [
                    shape=box,
                    style="rounded,filled",
                    fillcolor="#151a25",
                    color="#f2a93c",
                    fontcolor="#e9edf4",
                    fontname="Helvetica",
                    penwidth=1.2
                ];

                edge [
                    color="#4a5568",
                    fontcolor="#8b95a7",
                    fontname="Helvetica"
                ];

                ui [label="Streamlit Playground"];
                api [label="FastAPI Gateway"];
                classifier [label="Complexity Classifier"];
                confidence [label="Confidence Check"];
                router [label="Routing Policy"];
                provider [label="Mistral / Groq"];
                fallback [label="Fallback Protection"];
                verifier [label="Quality Verifier"];
                audit [label="SQLite Audit Log"];
                dashboard [label="Analytics Dashboard"];

                ui -> api;
                api -> classifier;
                classifier -> confidence;
                confidence -> router [label="safe tier"];
                confidence -> router [label="promote if low confidence"];
                router -> provider;
                provider -> fallback [label="failure"];
                provider -> verifier;
                fallback -> verifier;
                classifier -> audit;
                provider -> audit;
                fallback -> audit;
                verifier -> audit;
                audit -> dashboard;
            }
            """
        )


# ---------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------

render_footer(
    f"{total_requests:,} audited requests · "
    f"{total_tokens:,} tokens · "
    f"{money_short(total_cost)} inference cost · "
    f"P95 {percentile(request_log, 0.95):.3f}s · "
    f"{total_fallbacks:,} fallback requests · "
    f"LLM Cost Autopilot"
)
