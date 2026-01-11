import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List

# -------------------- CONFIG --------------------
st.set_page_config(
    page_title="Hallucination & ROUGE Analysis",
    layout="wide"
)

st.title("📊 Visualisation for Fine-tuning LLMs using Entity Hallucination Index")
st.caption("Research Demo | Before vs After Fine-Tuning Analysis")

FIGSIZE = (4.8, 2.2)
DPI = 120

# -------------------- HELPERS --------------------
def extract_model_name(filename: str) -> str:
    name = filename.lower()
    if "mistral" in name:
        return "Mistral"
    if "distil" in name:
        return "DistilBART"
    if "flan" in name or "t5" in name:
        return "Flan-T5"
    return filename.split(".")[0]


def load_json(uploaded_file):
    return json.load(uploaded_file)


def normalize_records(data: List[Dict], limit: int) -> pd.DataFrame:
    rows = []
    for r in data[:limit]:
        row = {}
        for m in ["EHI", "EF1", "PH", "OF", "NH", "LF", "EF"]:
            row[f"{m}_before"] = r.get(f"{m}_before")
            row[f"{m}_after"] = r.get(f"{m}_after")
        rows.append(row)
    return pd.DataFrame(rows)


# -------------------- AXIS CONTROL --------------------
def get_plot_indices(n: int):
    if n <= 20:
        step = 3
    elif n <= 50:
        step = 5
    else:
        step = 10
    return list(range(0, n, step)), step


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=8)


# -------------------- PLOTTING --------------------
def line_plot(df, metric, model):
    n = len(df)
    idx, _ = get_plot_indices(n)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    ax.plot(idx, df.loc[idx, f"{metric}_before"], label="Before FT", linewidth=1.2)
    ax.plot(idx, df.loc[idx, f"{metric}_after"], label="After FT", linewidth=1.2)

    ax.set_title(f"{model} — {metric}", fontsize=10)
    ax.set_xlabel("Record Number", fontsize=8)
    ax.set_ylabel(metric, fontsize=8)
    ax.set_ylim(0, 1.2)

    ax.set_xticks(idx)
    ax.set_xticklabels([str(i + 1) for i in idx])

    style_axes(ax)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        ncol=2,
        fontsize=7,
        frameon=False
    )

    st.pyplot(fig, clear_figure=True)


def multi_metric_plot(df, metrics, model, mode="after"):
    """
    All Hallucination Metrics plot
    Values > 1 are clipped to 1 (visualization only)
    """
    n = len(df)
    idx, _ = get_plot_indices(n)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    suffix = "_before" if mode == "before" else "_after"
    title_mode = "Before FT" if mode == "before" else "After FT"

    for m in metrics:
        values = df.loc[idx, f"{m}{suffix}"].clip(upper=1.0)
        ax.plot(
            idx,
            values,
            linewidth=1.2,
            label=m
        )

    ax.set_title(
        f"{model} — All Hallucination Metrics ({title_mode})",
        fontsize=10
    )
    ax.set_xlabel("Record Number", fontsize=8)
    ax.set_ylabel("Metric Value", fontsize=8)
    ax.set_ylim(0, 1.0)

    ax.set_xticks(idx)
    ax.set_xticklabels([str(i + 1) for i in idx])

    style_axes(ax)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.38),
        ncol=4,
        fontsize=7,
        frameon=False
    )

    st.pyplot(fig, clear_figure=True)


def compare_models_plot(model_dfs, metric):
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)

    for model, df in model_dfs.items():
        n = len(df)
        idx, _ = get_plot_indices(n)

        ax.plot(
            idx,
            df.loc[idx, f"{metric}_before"],
            linestyle="--",
            linewidth=1.2,
            label=f"{model} (Before)"
        )

        ax.plot(
            idx,
            df.loc[idx, f"{metric}_after"],
            linestyle="-",
            linewidth=1.4,
            label=f"{model} (After)"
        )

    ax.set_title(f"Model Comparison — {metric} (Before vs After FT)", fontsize=10)
    ax.set_xlabel("Record Number", fontsize=8)
    ax.set_ylabel(metric, fontsize=8)
    ax.set_ylim(0, 1.2)

    ax.set_xticks(idx)
    ax.set_xticklabels([str(i + 1) for i in idx])

    style_axes(ax)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.45),
        ncol=3,
        fontsize=7,
        frameon=False
    )

    st.pyplot(fig, clear_figure=True)


def avg_table(df, metrics, title):
    data = []
    for m in metrics:
        data.append({
            "Metric": m,
            "Before FT": round(df[f"{m}_before"].mean(), 4),
            "After FT": round(df[f"{m}_after"].mean(), 4)
        })
    st.subheader(title)
    st.dataframe(pd.DataFrame(data), use_container_width=True)


# -------------------- UI --------------------
uploaded_files = st.file_uploader(
    "Upload 3 JSON metric files",
    type=["json"],
    accept_multiple_files=True
)

num_records = st.slider("Number of records to visualize", 10, 100, 50, 10)

# -------------------- MAIN --------------------
if uploaded_files:
    model_data = {}

    model_files = {}
    for f in uploaded_files:
        model = extract_model_name(f.name)
        if model not in model_files:
            model_files[model] = f

    tabs = st.tabs(list(model_files.keys()))

    for tab, (model, file) in zip(tabs, model_files.items()):
        with tab:
            df = normalize_records(load_json(file), num_records)
            model_data[model] = df

            st.header(f"🔍 {model} Analysis")

            c1, c2 = st.columns(2)
            with c1:
                line_plot(df, "EHI", model)
            with c2:
                line_plot(df, "EF1", model)

            c1, c2 = st.columns(2)
            with c1:
                multi_metric_plot(
                    df,
                    ["EHI", "EF1", "OF", "NH", "LF", "EF", "PH"],
                    model,
                    mode="before"
                )
            with c2:
                multi_metric_plot(
                    df,
                    ["EHI", "EF1", "OF", "NH", "LF", "EF", "PH"],
                    model,
                    mode="after"
                )

            avg_table(
                df,
                ["EHI", "EF1", "OF", "NH", "LF", "EF", "PH"],
                "Average Hallucination Metrics"
            )

    st.markdown("---")
    st.header("📈 Cross-Model Comparison")

    c1, c2 = st.columns(2)
    with c1:
        compare_models_plot(model_data, "EHI")
    with c2:
        compare_models_plot(model_data, "EF1")

else:
    st.info("⬆️ Upload the three model metric files to begin.")
