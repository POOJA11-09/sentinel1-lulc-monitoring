"""
Streamlit web app for the Sentinel-1 LULC prototype.

Start (from the prototype folder):
    streamlit run app.py

Pages: Overview · Run prototype · Model comparison · Experiments & findings
All processing is done by lulc_prototype.py (the same code as the notebook).
"""
import contextlib, datetime as dt, io, os, site, sys

_pylibs = "/home/jovyan/vault/heidelberg_lulc_2025/pylibs/lib/python3.11/site-packages"
if os.path.isdir(_pylibs):                      # SRH Data Lab: packages live in the persistent vault
    site.addsitedir(_pylibs)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(HERE)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
import pandas as pd
import streamlit as st

import lulc_prototype as P
import temporal_smoothing as TS

# ------------------------------------------------------------------ project information (edit if needed)
PROJECT_TITLE = "Monthly Land-Use and Land-Cover Monitoring from Sentinel-1 SAR"
SUBTITLE = "Deep-learning semantic segmentation with radar as the only input"
AUTHOR = "Pooja Chandrappa"
COURSE = "Deep Learning"
UNIVERSITY = "SRH Hochschule Heidelberg"
YEAR = "2026"

NAVY, STEEL, GREY = "#003B6F", "#5B7FA6", "#A7B4C2"
CMAP, NORM = ListedColormap(P.PALETTE), BoundaryNorm(np.arange(-0.5, P.NC + 0.5), P.NC)
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titleweight": "bold", "axes.edgecolor": "#555555"})

st.set_page_config(page_title="Sentinel-1 LULC Monitoring", layout="wide", initial_sidebar_state="expanded")
st.markdown(f"""
<style>
  #MainMenu, footer {{visibility: hidden;}}
  .block-container {{padding-top: 1.2rem; max-width: 1350px;}}
  .app-header {{background: {NAVY}; color: #FFFFFF; padding: 1.1rem 1.6rem; border-radius: 4px; margin-bottom: 1.2rem;}}
  .app-header h1 {{color: #FFFFFF; font-size: 1.65rem; margin: 0 0 0.25rem 0; font-weight: 600;}}
  .app-header p {{color: #DCE4EE; margin: 0; font-size: 0.95rem;}}
  h2, h3 {{color: {NAVY};}}
  div[data-testid="stMetricValue"] {{color: {NAVY};}}
  .app-footer {{color: #6B7280; font-size: 0.8rem; border-top: 1px solid #E5E7EB; margin-top: 2.5rem; padding-top: 0.6rem;}}
</style>
<div class="app-header">
  <h1>{PROJECT_TITLE}</h1>
  <p>{SUBTITLE} &nbsp;·&nbsp; {AUTHOR} &nbsp;·&nbsp; {COURSE}, {UNIVERSITY} &nbsp;·&nbsp; {YEAR}</p>
</div>
""", unsafe_allow_html=True)


def footer():
    st.markdown(f'<div class="app-footer">{AUTHOR} · {COURSE} · {UNIVERSITY} · {YEAR} — '
                'Data: Copernicus Sentinel-1 (ESA), Google Dynamic World, ESA WorldCover; '
                'accessed via Google Earth Engine.</div>', unsafe_allow_html=True)


CITIES = {  # name: (lat, lon)
    "Frankfurt am Main (case study)": (50.1109, 8.6821),
    "Mannheim (inside training area)": (49.4875, 8.4660),
    "Heidelberg (training area centre)": (49.3980, 8.6750),
    "Stuttgart": (48.7758, 9.1829),
    "Munich": (48.1374, 11.5755),
    "Berlin": (52.5200, 13.4050),
    "Paris": (48.8566, 2.3522),
    "Madrid": (40.4168, -3.7038),
    "Cairo (very different region)": (30.0444, 31.2357),
    "Custom coordinates": None,
}
ARCH_NAMES = {"segformer": "SegFormer", "unet": "U-Net", "deeplabv3plus": "DeepLabV3+", "fpn": "FPN"}
MODE_NAMES = {"test10": "10 m data",
              "test40": "40 m data",
              "test40_from10": "40 m map from 10 m data (predict at 10 m, aggregate 4×4)"}
TRAIN_NAMES = {"10": "trained 10 m", "40": "trained 40 m", "mix": "trained 10 m + 40 m (mixed)"}


# ------------------------------------------------------------------ helpers
def available_models(input_res=10):
    """Models offered for the chosen input resolution (best match first)."""
    found, other = {}, {}
    has_best = input_res == 10 and os.path.exists("model/best_model.pt")
    if has_best:
        found["SegFormer, trained at 10 m (best model)"] = "model/best_model.pt"
    proj_models = os.path.join(HERE, "..", "models")
    if os.path.isdir(proj_models):
        for run in sorted(os.listdir(proj_models)):
            path = os.path.join(proj_models, run, "best.pt")
            if run.startswith("smoke_") or "_abl-" in run or "_ft-" in run or not os.path.exists(path):
                continue
            arch, res_tag = run.split("_train")[0], run.split("_train")[1].split("_")[0]
            name = ARCH_NAMES.get(arch, arch)
            if res_tag == "mix":
                found[f"{name}, trained on 10 m + 40 m (mixed)"] = path
            elif res_tag == str(input_res):
                if not (has_best and arch == "segformer"):
                    found[f"{name}, trained at {res_tag} m"] = path
            else:
                other[f"{name}, trained at {res_tag} m (resolution mismatch)"] = path
    if input_res == 40:      # best 40 m models first: mixed SegFormer, then the 40 m models
        found = dict(sorted(found.items(), key=lambda kv: ("mixed" not in kv[0], "SegFormer" not in kv[0])))
    return {**found, **other}


def find_file(name):
    for d in (os.path.join(HERE, "..", "outputs", "evaluation"), os.path.join(HERE, "evaluation")):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def find_figure(name):
    for d in (os.path.join(HERE, "..", "outputs", "figures"), os.path.join(HERE, "evaluation", "figures")):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def model_label(arch, train_res):
    return f"{ARCH_NAMES.get(arch, arch)} ({TRAIN_NAMES.get(str(train_res), str(train_res))})"


def load_results():
    p_over, p_cls = find_file("summary_overall.csv"), find_file("summary_per_class.csv")
    ov = pc = None
    if p_over:
        ov = pd.read_csv(p_over)
        ov = ov[~ov.run.str.startswith("smoke_")].copy()
        ov["model"] = [model_label(a, r) for a, r in zip(ov.arch, ov.train_res.astype(str))]
        ov["data"] = ov["mode"].map(MODE_NAMES)
    if p_cls:
        pc = pd.read_csv(p_cls)
        pc = pc[~pc.run.str.startswith("smoke_")].copy()
        pc["model"] = [model_label(a, r) for a, r in zip(pc.arch, pc.train_res.astype(str))]
    return ov, pc


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("### Navigation")
    page = st.radio("Page", ["Overview", "Run prototype", "Model comparison", "Experiments & findings"], label_visibility="collapsed")
    st.divider()
    if page == "Run prototype":
        st.markdown("### Input")
        city = st.selectbox("Location", list(CITIES))
        default = CITIES[city] or (50.1109, 8.6821)
        lat = st.number_input("Latitude (°)", -60.0, 75.0, default[0], step=0.01, format="%.4f",
                              key=f"lat_{city}", disabled=CITIES[city] is not None)
        lon = st.number_input("Longitude (°)", -180.0, 180.0, default[1], step=0.01, format="%.4f",
                              key=f"lon_{city}", disabled=CITIES[city] is not None)
        years = list(range(2017, dt.date.today().year + 1))
        year = st.selectbox("Start year", years, index=years.index(2025) if 2025 in years else len(years) - 1)
        month = st.selectbox("Start month", list(range(1, 13)),
                             format_func=lambda i: dt.date(2000, i, 1).strftime("%B"))
        size_km = st.slider("Area size (km × km)", 5, 20, 10)
        res_label = st.radio("Input resolution", ["10 m", "40 m"], horizontal=True,
                             help="40 m: the 10 m Sentinel-1 composites are averaged over 4 × 4 pixels "
                                  "(linear power), as for the 40 m training data, and a 40 m or mixed model is used.")
        input_res = 40 if res_label == "40 m" else 10
        models = available_models(input_res)
        if not models:
            st.error("No model weights found (expected model/best_model.pt).")
            st.stop()
        model_name = st.selectbox("Model", list(models))
        if "mismatch" in model_name:
            st.warning(f"This model was trained at a different resolution than the {input_res} m input — expect "
                       "poor maps (demonstrates the resolution mismatch).")
        with st.expander("Advanced settings"):
            project = st.text_input("Earth Engine / Google Cloud project ID",
                                    os.environ.get("EE_PROJECT", "heidelberg-lulc-research"),
                                    help="Your own Cloud project registered for Earth Engine. "
                                         "Can also be set with the environment variable EE_PROJECT.")
            do_wc = st.checkbox("Sanity check against ESA WorldCover 2021", True)
        run_clicked = st.button("Run analysis", type="primary", use_container_width=True)
        st.map(pd.DataFrame({"lat": [lat], "lon": [lon]}), zoom=8, size=400, color=NAVY)
    else:
        st.caption("Select **Run prototype** to analyse a location.")


# ------------------------------------------------------------------ page: overview
def overview_page():
    ov, pc = load_results()
    st.markdown("## Welcome")
    st.markdown(
        "This application maps **land use and land cover (LULC)** for any location and any 12-month period "
        "using **Sentinel-1 synthetic aperture radar (SAR) as the only input**. Because radar is independent of "
        "clouds and daylight, a land-cover map can be produced for every month of the year. For a chosen location "
        "and start month the application produces **12 monthly LULC maps**, the **share of each land-cover class**, "
        "the **month-to-month change** and the **total change** in percentage points, and checks the result against "
        "an independent land-cover product.")

    if ov is not None:
        b = ov[ov["mode"] == "test10"].sort_values("mIoU", ascending=False).iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best model", ARCH_NAMES.get(b.arch, b.arch) + " (10 m)")
        c2.metric("Test overall accuracy", f"{100 * b.OA:.1f} %")
        c3.metric("Test mean IoU (9 classes)", f"{b.mIoU:.3f}")
        if pc is not None:
            f1 = pc[(pc.run == b.run) & (pc["mode"] == "test10") & (pc["class"] == "Built-up")]
            if len(f1):
                c4.metric("Built-up F1", f"{f1.F1.iloc[0]:.3f}")
        st.caption("Test results on spatially separate areas; reference: Google Dynamic World.")

    left, right = st.columns(2)
    with left:
        st.markdown("### How to use")
        st.markdown(
            "1. Open **Run prototype** in the sidebar\n"
            "2. Select a location or enter coordinates, a start year and a start month\n"
            "3. Press **Run analysis** (about 1–2 minutes with a GPU, a few minutes on a CPU)\n"
            "4. Review the maps, class percentages, changes and the WorldCover check\n"
            "5. **Model comparison** shows the test results of all trained models\n"
            "6. **Experiments & findings** summarises all experiments and diagnostics")
    with right:
        st.markdown("### Land-cover classes")
        legend = "".join(f'<span style="display:inline-block;width:12px;height:12px;background:{c};'
                         f'margin:0 6px 0 14px;vertical-align:middle;border:1px solid #888"></span>{n}'
                         for n, c in zip(P.CLASS_NAMES, P.PALETTE))
        st.markdown(f'<div style="line-height:2">{legend}</div>', unsafe_allow_html=True)
        st.caption("Model trained on Sentinel-1 monthly composites around Heidelberg (2025) with Google Dynamic "
                   "World labels. Methods, limitations and references are documented in the project report.")

    arch = find_figure("system_architecture.png")
    if arch:
        st.markdown("### System architecture")
        st.image(arch, use_container_width=True,
                 caption="Data flow of the prototype: Google Earth Engine retrieval → monthly compositing → "
                         "segmentation → statistics and optional temporal smoothing → web app.")


# ------------------------------------------------------------------ page: model comparison
def comparison_page():
    ov, pc = load_results()
    st.markdown("## Model comparison")
    st.markdown("All models were evaluated once on the same **spatially separate test areas** (4 blocks of "
                "7.68 km × 7.68 km, all 12 months). Scores measure agreement with Dynamic World.")
    if ov is None:
        st.warning("Evaluation results not found (expected evaluation/summary_overall.csv).")
        return

    marc = find_figure("model_architectures.png")
    if marc:
        st.markdown("### Model architectures")
        st.image(marc, use_container_width=True,
                 caption="The four segmentation architectures compared (segmentation-models-pytorch 0.6.0, "
                         "ImageNet-pre-trained encoders, 22–25 M parameters each).")

    st.markdown("### Recommended model by type of data")
    rec = []
    for mode in ["test10", "test40", "test40_from10"]:
        sub = ov[ov["mode"] == mode]
        if len(sub):
            b = sub.loc[sub.mIoU.idxmax()]
            rec.append({"Data / goal": MODE_NAMES[mode], "Recommended model": b.model,
                        "OA": f"{b.OA:.3f}", "mIoU": f"{b.mIoU:.3f}", "Mean F1": f"{b.mF1:.3f}"})
    both = ov[ov["mode"].isin(["test10", "test40"])].pivot_table(index="model", columns="mode", values="mIoU")
    if {"test10", "test40"} <= set(both.columns):
        m = both.min(axis=1).idxmax()
        rec.append({"Data / goal": "Unknown or mixed resolution (one model for both)", "Recommended model": m,
                    "OA": "—", "mIoU": f"{both.loc[m, 'test10']:.3f} (10 m) / {both.loc[m, 'test40']:.3f} (40 m)",
                    "Mean F1": "—"})
    st.dataframe(pd.DataFrame(rec), hide_index=True, use_container_width=True)
    st.caption("The prototype uses 10 m Sentinel-1 data; its default model is therefore the best 10 m model. "
               "Models applied to a resolution they were not trained on perform poorly.")

    st.markdown("### All models")
    metric = st.radio("Metric", ["mIoU", "OA", "mF1"], horizontal=True,
                      format_func={"mIoU": "Mean IoU", "OA": "Overall accuracy", "mF1": "Mean F1"}.get)
    table = ov.pivot_table(index="model", columns="data", values=metric).round(3)
    table = table[[c for c in MODE_NAMES.values() if c in table.columns]]
    st.dataframe(table.style.format("{:.3f}").background_gradient(cmap="Blues", axis=None),
                 use_container_width=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    table.plot.bar(ax=ax, color=[NAVY, STEEL, GREY][:table.shape[1]], width=0.8, edgecolor="white")
    ax.set_ylabel(metric); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=0.3); ax.set_xlabel("")
    ax.set_title(f"Test {metric} by model and data resolution")
    ax.set_xticklabels([t.get_text().replace(" (", "\n(") for t in ax.get_xticklabels()], rotation=0, fontsize=7)
    ax.legend(fontsize=8, frameon=False)
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

    if pc is not None:
        st.markdown("### Per-class results")
        c1, c2 = st.columns(2)
        mode_lab = c1.selectbox("Data", list(MODE_NAMES.values()), index=0)
        score = c2.selectbox("Score", ["IoU", "F1"])
        mode = {v: k for k, v in MODE_NAMES.items()}[mode_lab]
        t = pc[pc["mode"] == mode].pivot_table(index="class", columns="model", values=score)
        t = t.reindex([c for c in P.CLASS_NAMES if c in t.index]).round(3)
        st.dataframe(t.style.format("{:.3f}").background_gradient(cmap="Blues", vmin=0, vmax=1, axis=None),
                     use_container_width=True)
        st.caption("Large classes (built-up, trees, crops, water) are mapped well; rare or radar-ambiguous classes "
                   "(grass, shrub, bare ground, flooded vegetation, snow) are not.")

    p69 = find_file("comparison_9_vs_6_classes_test10.csv")
    if p69:
        st.markdown("### Nine classes vs six merged classes (post-hoc analysis, 10 m models)")
        st.dataframe(pd.read_csv(p69), hide_index=True, use_container_width=True)
    figs = [("Training curves — models trained at 10 m", "training_curves_train10.png"),
            ("Training curves — models trained at 40 m", "training_curves_train40.png"),
            ("Training curves — mixed-resolution training", "training_curves_trainmix.png"),
            ("Confusion matrices, SegFormer 10 m (9 vs 6 classes)", "confusion_9_vs_6_segformer.png")]
    shown = [(t, find_figure(f)) for t, f in figs if find_figure(f)]
    if shown:
        st.markdown("### Training curves and confusion matrices")
        for title, path in shown:
            with st.expander(title):
                st.image(path, use_container_width=True)


# ------------------------------------------------------------------ page: experiments & findings
BASE_RUN = "segformer_train10_VV-VH-RATIO_s42"
ABL_RUNS = {BASE_RUN: "Baseline (seed 42)",
            "segformer_train10_abl-E1-seed1": "E1 · seed 1 (noise check)",
            "segformer_train10_abl-X3-month": "X3 · + month encoding",
            "segformer_train10_abl-X2b-confweight": "X2b · confidence-weighted loss",
            "segformer_train10_abl-E3-lee": "E3 · Lee speckle filter 5×5",
            "segformer_train10_ft-F1-scratch": "F1 · no ImageNet pre-training",
            "segformer_train10_ft-F2-mitb0": "F2 · smaller encoder (MiT-B0)",
            "segformer_train10_ft-F3-mitb4": "F3 · larger encoder (MiT-B4)",
            "segformer_train10_ft-F4-cosine": "F4 · cosine learning-rate schedule",
            "segformer_train10_ft-F5-lr1e-4": "F5 · learning rate 1e-4",
            "segformer_train10_ft-F6-noclassweight": "F6 · no class weights in the loss"}


def find_diag(name):
    for d in (os.path.join(HERE, "..", "outputs", "diagnostics", "weak_classes", BASE_RUN),
              os.path.join(HERE, "evaluation", "diagnostics", "weak_classes", BASE_RUN)):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def finding(text):
    st.info(f"**Key finding.** {text}")


def heat(ax, data, rows, cols, fmt="{:.3f}", vmin=0, vmax=1, cmap="Blues"):
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=8)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8,
                        color="white" if (v - vmin) / (vmax - vmin + 1e-9) > 0.6 else "#1F2937")
    return im


def tab_resolution(ov):
    st.markdown("#### Does a model work on a resolution it was not trained on?")
    st.markdown("Every architecture was trained three times (10 m data, 40 m data, both mixed) and each model was "
                "tested on 10 m data, on 40 m data, and by predicting at 10 m and aggregating to 40 m (4×4 majority).")
    if ov is None:
        st.warning("Evaluation results not found."); return
    archs = [a for a in ["segformer", "unet", "deeplabv3plus", "fpn"] if a in set(ov.arch)]
    trains, modes = ["10", "40", "mix"], ["test10", "test40", "test40_from10"]
    fig, axs = plt.subplots(1, len(archs), figsize=(4.6 * len(archs), 3.9))
    axs = np.atleast_1d(axs)
    for ax, a in zip(axs, archs):
        sub = ov[ov.arch == a]
        m = np.full((3, 3), np.nan)
        for i, t in enumerate(trains):
            for j, md in enumerate(modes):
                v = sub[(sub.train_res.astype(str) == t) & (sub["mode"] == md)].mIoU
                if len(v): m[i, j] = v.iloc[0]
        heat(ax, m, ["trained 10 m", "trained 40 m", "trained mixed"], ["test 10 m", "test 40 m", "10 m→40 m"],
             vmin=0, vmax=0.5)
        ax.set_title(ARCH_NAMES.get(a, a), fontsize=10)
    fig.suptitle("Test mIoU: training resolution (rows) × test data (columns)", fontsize=11, color=NAVY)
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    finding("Models only work at the resolution they were trained on: 40 m models applied to 10 m data collapse "
            "(OA 0.34–0.52) and 10 m models applied to 40 m data drop to OA 0.56–0.66. Predicting at 10 m and "
            "aggregating to 40 m is as good as the 10 m result, so a separate 40 m model is not needed. Mixed "
            "training gives one model for both resolutions at a small cost (SegFormer: 0.422 vs 0.432 mIoU at 10 m).")


def tab_fitting(ov):
    st.markdown("#### Over- or underfitting? Same model, three data splits")
    st.markdown("Training, validation and test areas are spatially separate blocks (7.68 km); the same split is "
                "used for every month and both resolutions, so no area is seen in training and testing.")
    rows = []
    for split, rel in [("Train", "train/summary_overall.csv"), ("Validation", "val/summary_overall.csv"),
                       ("Test", "summary_overall.csv")]:
        p = find_file(rel)
        if not p:
            continue
        d = pd.read_csv(p)
        d = d[~d.run.str.startswith("smoke_") & ~d.run.str.contains("abl-")]
        d = d[((d.train_res.astype(str) == "40") & (d["mode"] == "test40")) |
              ((d.train_res.astype(str) != "40") & (d["mode"] == "test10"))]
        for _, r in d.iterrows():
            rows.append({"model": model_label(r.arch, str(r.train_res)), "split": split, "mIoU": r.mIoU, "OA": r.OA})
    if not rows:
        st.warning("Train/validation results not found (evaluation/train, evaluation/val)."); return
    df = pd.DataFrame(rows)
    t = df.pivot_table(index="model", columns="split", values="mIoU")
    t = t[[c for c in ["Train", "Validation", "Test"] if c in t.columns]]
    if {"Train", "Test"} <= set(t.columns):
        t["Gap train − test"] = t["Train"] - t["Test"]
    t = t.sort_values("Test", ascending=False) if "Test" in t.columns else t
    fig, ax = plt.subplots(figsize=(12, 4.2))
    t[[c for c in ["Train", "Validation", "Test"] if c in t.columns]].plot.bar(
        ax=ax, color=[GREY, STEEL, NAVY], width=0.8, edgecolor="white")
    ax.set_ylabel("mIoU (matched resolution)"); ax.set_xlabel(""); ax.grid(axis="y", alpha=0.3)
    ax.set_xticklabels([x.get_text().replace(" (", "\n(") for x in ax.get_xticklabels()], rotation=0, fontsize=7)
    ax.legend(frameon=False); ax.set_title("Train / validation / test mIoU per model")
    plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    st.dataframe(t.round(3).style.format("{:.3f}"), use_container_width=True)
    finding("The gap between training and test mIoU is moderate (about 0.05–0.11) and validation scores did not "
            "decline over time, so there is no strong overfitting; early stopping selected the best epoch. FPN shows "
            "the largest gap (slight overfitting), the mixed U-Net the lowest training score (underfitting). For "
            "SegFormer most of the gap comes from two rare classes (flooded vegetation, bare ground).")


def tab_errors(pc):
    st.markdown("#### Where does the best model make mistakes, and why?")
    p = find_file(f"{BASE_RUN}/confusion_test10.csv")
    if p:
        cm = pd.read_csv(p, index_col=0).values.astype(float)
        names = list(P.CLASS_NAMES)
        rown = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        c1, c2 = st.columns([3, 2])
        with c1:
            fig, ax = plt.subplots(figsize=(7, 5.6))
            heat(ax, 100 * rown, names, names, fmt="{:.0f}", vmin=0, vmax=100)
            ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
            ax.set_xlabel("predicted class"); ax.set_ylabel("reference class (Dynamic World)")
            ax.set_title("Confusion matrix, SegFormer 10 m, test (% of each reference class)", fontsize=10)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
        with c2:
            acc = pd.DataFrame({"Recall": np.diag(cm) / np.maximum(cm.sum(1), 1),
                                "Precision": np.diag(cm) / np.maximum(cm.sum(0), 1),
                                "Ref. area %": 100 * cm.sum(1) / cm.sum(),
                                "Pred. area %": 100 * cm.sum(0) / cm.sum()}, index=names)
            st.dataframe(acc.style.format("{:.2f}"), use_container_width=True)
            po, pe = np.trace(cm) / cm.sum(), (cm.sum(0) * cm.sum(1)).sum() / cm.sum() ** 2
            st.caption("Recall = producer's accuracy, precision = user's accuracy.")
            st.metric("Cohen's kappa", f"{(po - pe) / (1 - pe):.3f}")
    else:
        st.warning("Confusion matrix not found.")

    st.markdown("#### Why are some classes weak? (diagnostics on the test set)")
    c1, c2 = st.columns(2)
    q = find_diag("3_label_quality_per_class.csv")
    if q:
        with c1:
            d = pd.read_csv(q, index_col=0)
            fig, ax = plt.subplots(figsize=(6, 3.6))
            d["conf<0.5_%"].sort_values().plot.barh(ax=ax, color=NAVY)
            ax.set_xlabel("% of reference pixels with label confidence < 0.5"); ax.grid(axis="x", alpha=0.3)
            ax.set_title("Label quality of Dynamic World per class", fontsize=10)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    s = find_diag("5_sar_separability.csv")
    if s:
        with c2:
            d = pd.read_csv(s)
            d = d[(d.season == "all") & d.JM_distance.notna()].sort_values("JM_distance")
            fig, ax = plt.subplots(figsize=(6, 3.6))
            ax.barh(d.pair, d.JM_distance, color=[("#C0392B" if v < 0.3 else STEEL) for v in d.JM_distance])
            ax.set_xlabel("Jeffries–Matusita distance in VV/VH (0 = identical, 2 = separable)")
            ax.grid(axis="x", alpha=0.3); ax.set_title("Can radar separate the confused classes?", fontsize=10)
            ax.tick_params(axis="y", labelsize=8)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    f = find_diag("fig_recall_stratified.png")
    if f:
        with st.expander("Recall stratified by label confidence, 40 m purity and month"):
            st.image(f, use_container_width=True)
    finding("The main confusions are physically plausible: grass→crops, shrub→trees, bare ground→built-up and "
            "flooded vegetation→water have almost identical VV/VH backscatter (JM distance 0.02–0.16). In addition, "
            "88–99.9 % of the weak-class labels have Dynamic World confidence below 0.5 and are mostly winter scenes, "
            "whereas on clean, confident pixels the strong classes reach IoU 0.93–0.97. Class weighting is not the cause.")


EXP_RUNS = [BASE_RUN, "segformer_train10_abl-E1-seed1", "segformer_train10_abl-X3-month",
            "segformer_train10_abl-X2b-confweight", "segformer_train10_abl-E3-lee"]
FT_RUNS = [BASE_RUN, "segformer_train10_abl-E1-seed1", "segformer_train10_ft-F1-scratch",
           "segformer_train10_ft-F2-mitb0", "segformer_train10_ft-F3-mitb4", "segformer_train10_ft-F4-cosine",
           "segformer_train10_ft-F5-lr1e-4", "segformer_train10_ft-F6-noclassweight"]


def load_ablation():
    pt, pv, pcls = (find_file("ablation/summary_overall.csv"), find_file("ablation/val/summary_overall.csv"),
                    find_file("ablation/summary_per_class.csv"))
    te = pd.read_csv(pt) if pt else None
    va = pd.read_csv(pv) if pv else None
    pc = pd.read_csv(pcls) if pcls else None
    return te, va, pc


def run_table(runs, te, va):
    t = te[(te["mode"] == "test10") & te.run.isin(runs)].set_index("run")
    tab = t[["best_epoch", "OA", "mIoU", "mF1"]].rename(columns={"mIoU": "Test mIoU", "OA": "Test OA",
                                                                   "mF1": "Test mean F1", "best_epoch": "Best epoch"})
    if va is not None:
        v = va[(va["mode"] == "test10") & va.run.isin(runs)].set_index("run")
        tab.insert(1, "Val mIoU", v.mIoU)
    tab = tab.loc[[r for r in runs if r in tab.index]]
    tab["Δ test mIoU"] = tab["Test mIoU"] - tab.loc[BASE_RUN, "Test mIoU"]
    return tab


def show_runs(runs, te, va, pc):
    """Table, bar chart with seed-noise band and per-class heatmap for a group of runs."""
    tab = run_table(runs, te, va)
    if len(tab) == 0:
        return tab
    shown = tab.copy(); shown.index = shown.index.map(ABL_RUNS)
    st.dataframe(shown.style.format({c: "{:.4f}" for c in shown.columns if c != "Best epoch"}),
                 use_container_width=True)
    noise = abs(shown["Δ test mIoU"].iloc[1]) if len(shown) > 1 else 0
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6.2, 3.9))
        base = shown["Test mIoU"].iloc[0]
        ax.axhspan(base - noise, base + noise, color=GREY, alpha=0.35, label="baseline ± seed variation")
        cols = [NAVY, GREY] + [("#C0392B" if d < -noise else STEEL) for d in shown["Δ test mIoU"].iloc[2:]]
        ax.bar(range(len(shown)), shown["Test mIoU"], color=cols[:len(shown)])
        for i, v in enumerate(shown["Test mIoU"]):
            ax.text(i, v + 0.001, f"{v:.3f}", ha="center", fontsize=8)
        ax.set_xticks(range(len(shown)))
        ax.set_xticklabels([n.split(" · ")[0].replace("Baseline (seed 42)", "Base") for n in shown.index], fontsize=8)
        ax.set_ylim(min(0.39, shown["Test mIoU"].min() - 0.01), max(0.46, shown["Test mIoU"].max() + 0.01))
        ax.set_ylabel("test mIoU"); ax.legend(fontsize=8, frameon=False, loc="lower right")
        ax.set_title("Test mIoU vs. random variation (red = clearly worse)", fontsize=10)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    with c2:
        if pc is not None:
            p2 = pc[(pc["mode"] == "test10") & pc.run.isin(runs)]
            t = p2.pivot_table(index="class", columns="run", values="IoU")
            t = t[[r for r in runs if r in t.columns]]
            t = t.reindex([c for c in P.CLASS_NAMES if c in t.index])
            fig, ax = plt.subplots(figsize=(6.2, 3.9))
            heat(ax, t.values, list(t.index), [ABL_RUNS[r].split(" · ")[0].replace("Baseline (seed 42)", "Base") for r in t.columns], fmt="{:.2f}")
            ax.set_title("Per-class test IoU", fontsize=10)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    return tab


def tab_ablation():
    st.markdown("#### Can the best model be improved? Input and data experiments")
    st.markdown("Each run changes **one** thing compared with the SegFormer 10 m baseline; data, split, nine "
                "classes and all other settings are identical. Run **E1** only changes the random seed and "
                "therefore measures how much results vary by chance (grey band).")
    te, va, pc = load_ablation()
    if te is None:
        st.warning("Experiment results not found (evaluation/ablation)."); return
    show_runs(EXP_RUNS, te, va, pc)
    st.caption("Speckle filter (E3): the Lee filter lowered the local coefficient of variation of VV from 0.369 to "
               "0.242 (−34 %) and changed pixel values by 0.53 dB on average.")
    finding("No change improved the model by more than the seed-to-seed variation (±0.011 mIoU): month encoding "
            "+0.006, confidence weighting +0.001. The Lee speckle filter made it worse (−0.008), mainly for shrub "
            "(−42 %, texture removed) and water (blurred edges), which supports using temporal averaging only. "
            "The baseline is robust; the remaining errors come from the radar signal and the reference labels.")


def tab_finetuning():
    st.markdown("#### Fine-tuning the architecture and training")
    st.markdown("Six further SegFormer 10 m runs, again changing **one** setting each: transfer learning "
                "(F1), encoder size (F2, F3), learning-rate schedule (F4), learning rate (F5) and class weighting "
                "in the loss (F6). Baseline: MiT-B2 encoder (24.7 M parameters), ImageNet pre-training, AdamW "
                "lr 3e-4 with OneCycle, weighted cross-entropy + Dice.")
    te, va, pc = load_ablation()
    if te is None:
        st.warning("Experiment results not found (evaluation/ablation)."); return
    show_runs(FT_RUNS, te, va, pc)

    t = te[(te["mode"] == "test10") & te.run.isin(ABL_RUNS)].set_index("run")
    c1, c2 = st.columns(2)
    with c1:
        size = [(r, n) for r, n in [("segformer_train10_ft-F2-mitb0", 3.7), (BASE_RUN, 24.7),
                                    ("segformer_train10_ft-F3-mitb4", 62.0)] if r in t.index]
        if len(size) > 1:
            fig, ax = plt.subplots(figsize=(6.2, 3.6))
            xs, ys = [n for _, n in size], [t.loc[r, "mIoU"] for r, _ in size]
            ax.plot(xs, ys, "-o", color=NAVY)
            for (r, n), y in zip(size, ys):
                ax.annotate({3.7: "MiT-B0", 24.7: "MiT-B2", 62.0: "MiT-B4"}[n], (n, y),
                            textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
            ax.set_xscale("log"); ax.set_xticks([3.7, 24.7, 62.0]); ax.set_xticklabels(["3.7 M", "24.7 M", "62 M"])
            ax.minorticks_off(); ax.set_xlabel("encoder parameters (log scale)")
            ax.set_ylabel("test mIoU"); ax.set_ylim(0.40, 0.46); ax.grid(alpha=0.3)
            ax.set_title("Model size vs. accuracy", fontsize=10)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    with c2:
        pre = t[t.index != "segformer_train10_ft-F1-scratch"]["mIoU"]
        if len(pre) > 2 and "segformer_train10_ft-F1-scratch" in t.index:
            fig, ax = plt.subplots(figsize=(6.2, 3.6))
            m, sd = pre.mean(), pre.std()
            ax.axvspan(m - 2 * sd, m + 2 * sd, color=GREY, alpha=0.35, label=f"pre-trained runs: mean ± 2σ")
            ax.scatter(pre, np.zeros(len(pre)), color=STEEL, s=60, label=f"{len(pre)} pre-trained runs")
            ax.scatter([t.loc["segformer_train10_ft-F1-scratch", "mIoU"]], [0], color="#C0392B", s=80,
                       label="F1 without pre-training")
            ax.set_yticks([]); ax.set_xlabel("test mIoU"); ax.legend(fontsize=8, frameon=False, loc="upper left")
            ax.set_ylim(-1, 1.6); ax.grid(axis="x", alpha=0.3)
            ax.set_title(f"All runs: mean {m:.3f}, σ {sd:.3f}", fontsize=10)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    finding("Only ImageNet pre-training matters: without it (F1) test mIoU drops by 0.032 and every class gets "
            "worse, far outside the spread of all pre-trained runs (σ ≈ 0.005). Low-level features learned on "
            "photographs (edges, textures) transfer to SAR. Encoder size from 3.7 M to 62 M parameters, learning "
            "rate and schedule change nothing (MiT-B4 overfits earliest). Removing class weights raises overall "
            "accuracy (0.840, best of all runs) but lowers the rare classes, so mIoU is unchanged. Accuracy is "
            "limited by the information in the radar signal and the labels, not by the model design.")


def tab_smoothing():
    st.markdown("#### How much of the calculated change is real? Temporal smoothing of the prototype output")
    st.markdown("The 12 monthly maps of each city were post-processed in two ways: **majority3** (each month takes "
                "the majority of the previous, current and next month) and **persistence** (a pixel only changes "
                "class if the new class lasts at least two months). *Gross change* = share of pixels whose class "
                "differs from the previous month.")
    p = os.path.join(HERE, "results", "temporal_smoothing", "summary.csv")
    if not os.path.exists(p):
        st.warning("No smoothing results yet (run temporal_smoothing.py after some prototype runs)."); return
    sm = pd.read_csv(p)
    order = ["raw", "majority3", "persistence"]
    a = sm.pivot(index="site", columns="method", values="mean_gross_change_%")[order]
    b = sm.pivot(index="site", columns="method", values="total_built_up_change_pp")[order]
    c1, c2 = st.columns(2)
    for col, data, ylab, title in [(c1, a, "mean monthly gross change (%)", "Apparent monthly change"),
                                   (c2, b, "built-up change first → last month (pp)", "Total built-up change")]:
        with col:
            fig, ax = plt.subplots(figsize=(6, 3.8))
            data.plot.bar(ax=ax, color=["#C0392B", "#8FB9DD", NAVY], width=0.8)
            ax.axhline(0, color="black", lw=0.7); ax.set_ylabel(ylab); ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=0, labelsize=8); ax.grid(axis="y", alpha=0.3)
            ax.legend(fontsize=8, frameon=False); ax.set_title(title, fontsize=10)
            plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    show = sm.rename(columns={"site": "City", "method": "Method", "mean_gross_change_%": "Gross change %/month",
                              "mean_gross_change_summer_%": "Summer %/month",
                              "max_abs_monthly_built_up_pp": "Max monthly built-up Δ (pp)",
                              "total_built_up_change_pp": "Built-up Δ total (pp)",
                              "first_last_gross_change_%": "First→last gross %",
                              "pixels_modified_vs_raw_%": "Pixels modified %"})
    st.dataframe(show.style.format({c: "{:.2f}" for c in show.columns[2:]}), hide_index=True,
                 use_container_width=True)
    finding("Smoothing changes only 0.7–2.7 % of the pixels but removes 53–73 % of the apparent monthly change, so "
            "most raw change is classification noise. Annual built-up changes of −5 to −7 pp remain in German cities "
            "(real change is far below 1 pp per year) and Mannheim even changes sign, whereas Cairo, without winter "
            "effects, is stable. Seasonal backscatter changes, not land-cover change, inflate the change statistics.")


EXPERIMENT_LOG = [
    # (group, experiment, question, result, conclusion)
    ("Data", "Spatial block split + leakage check", "Are training and test areas really separate?",
     "17/4/4 super-blocks (7.68 km), same split for all months and resolutions; 0 overlapping blocks",
     "Test scores measure generalisation to unseen areas"),
    ("Data", "Class distribution per split", "Are the splits comparable?",
     "Trees/built-up/crops = 87 % of pixels; major classes differ ≤ 2 pp between splits",
     "Fair split; strong imbalance → class weights, Dice loss, mIoU as main metric"),
    ("Architecture", "4 architectures at 10 m", "Which architecture fits SAR segmentation best?",
     "SegFormer 0.432, U-Net 0.423, DeepLabV3+ 0.407, FPN 0.400 (test mIoU)",
     "SegFormer ≈ U-Net (within seed noise); both better than DeepLabV3+ and FPN"),
    ("Resolution", "Train 10 m / 40 m / mixed × test 10 m / 40 m", "Does a model transfer between resolutions?",
     "40 m models on 10 m data: OA 0.34–0.52; 10 m models on 40 m data: OA 0.56–0.66",
     "Models only work at their training resolution"),
    ("Resolution", "Predict at 10 m, aggregate to 40 m", "Best way to produce 40 m maps?",
     "SegFormer 0.433 mIoU vs 0.348 for the 40 m model", "Predict at 10 m and aggregate"),
    ("Resolution", "Mixed-resolution training", "Can one model serve both resolutions?",
     "SegFormer mixed: 0.422 (10 m) / 0.357 (40 m)", "Yes, at a small cost at 10 m"),
    ("Classes", "9 vs 6 merged classes (post-hoc)", "Would fewer classes help?",
     "Merging weak classes raises mIoU but removes information (Model comparison page)",
     "Nine classes kept as required"),
    ("Diagnosis", "Train / validation / test gap", "Over- or underfitting?",
     "Gap 0.05–0.11 mIoU, no validation decline; FPN largest gap, mixed U-Net underfits",
     "No strong overfitting; early stopping works"),
    ("Diagnosis", "Weak-class analysis", "Why are some classes weak?",
     "JM separability 0.02–0.16; 88–99.9 % of weak labels with confidence < 0.5; mostly winter",
     "Limited by radar physics and label quality, not by class weights"),
    ("Input", "E1 · random seed 1", "How large is random variation?", "+0.011 mIoU", "Noise level for all comparisons"),
    ("Input", "X3 · month encoding", "Does knowing the month help?", "+0.006 mIoU", "No effect beyond noise"),
    ("Input", "X2b · confidence-weighted loss", "Does down-weighting uncertain labels help?", "+0.001 mIoU",
     "No effect"),
    ("Input", "E3 · Lee speckle filter 5×5", "Is spatial speckle filtering needed?",
     "Speckle −34 %, but −0.008 mIoU (shrub −42 %, water −0.038)", "Worse; temporal averaging is enough"),
    ("Fine-tuning", "F1 · no ImageNet pre-training", "Does transfer learning help for radar?",
     "−0.032 mIoU, every class worse", "Yes: the only clear effect"),
    ("Fine-tuning", "F2 · MiT-B0 (3.7 M parameters)", "Is a smaller model enough?", "−0.000 mIoU",
     "Same accuracy with 7× fewer parameters"),
    ("Fine-tuning", "F3 · MiT-B4 (62 M parameters)", "Does a larger model help?", "−0.001 mIoU, best epoch 12",
     "No; overfits earlier"),
    ("Fine-tuning", "F4 · cosine schedule", "Was OneCycle cut short by early stopping?", "+0.002 mIoU",
     "No effect"),
    ("Fine-tuning", "F5 · learning rate 1e-4", "Was 3e-4 too high?", "+0.003 mIoU", "No effect"),
    ("Fine-tuning", "F6 · no class weights", "Do class weights help rare classes?",
     "OA 0.840 (best), mIoU +0.002, rare classes lower", "Weights shift accuracy to rare classes"),
    ("Prototype", "WorldCover case study", "Do the maps agree with an independent product?",
     "Frankfurt 65.5 %, Mannheim 69.2 %, Cairo 89.1 % agreement", "Plausible maps; Cairo inflated by built-up"),
    ("Prototype", "Transfer to Cairo", "Does the model work in a very different region?",
     "Desert classified as built-up; very stable months", "Not valid outside temperate regions"),
    ("Prototype", "Temporal smoothing (6 cities)", "How much calculated change is real?",
     "−53–73 % apparent change, ≤ 2.7 % pixels modified", "Most raw change is noise / seasonal"),
]

CLASS_DIST = pd.DataFrame(
    {"Train": [3.23, 42.26, 6.20, 0.03, 21.96, 1.75, 23.68, 0.33, 0.57],
     "Validation": [4.59, 41.22, 4.67, 0.01, 23.81, 1.84, 22.56, 0.76, 0.53],
     "Test": [3.24, 41.34, 6.25, 0.01, 22.54, 1.79, 23.34, 0.41, 1.08]},
    index=["Water", "Trees", "Grass", "Flooded vegetation", "Crops", "Shrub and scrub", "Built-up",
           "Bare ground", "Snow and ice"])


def tab_summary():
    st.markdown("#### All experiments at a glance")
    st.markdown("Every experiment was designed to answer one question. Scores are test mIoU (SegFormer 10 m "
                "unless stated); differences are compared with the run-to-run variation caused by the random "
                "seed (±0.011 mIoU). Details for each group are in the following tabs.")
    df = pd.DataFrame(EXPERIMENT_LOG, columns=["Group", "Experiment", "Question", "Result", "Conclusion"])
    df.insert(0, "#", range(1, len(df) + 1))
    groups = ["All"] + list(dict.fromkeys(df.Group))
    g = st.radio("Show", groups, horizontal=True, key="exp_group")
    if g != "All":
        df = df[df.Group == g]
    st.table(df.set_index("#"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Trained models", "20")
    c2.metric("Experiments", f"{len(EXPERIMENT_LOG)}")
    c3.metric("Best test OA / mIoU", "0.832 / 0.432")
    c4.metric("Run-to-run noise", "±0.011 mIoU")
    finding("Out of all design, input and training changes, only ImageNet pre-training and matching the training "
            "resolution to the data clearly affect accuracy. All other changes stay within random variation. "
            "The remaining errors are explained by low radar separability of some classes, uncertain winter labels "
            "and very rare classes; in the prototype, these errors and seasonal effects inflate the change statistics.")


def tab_dataset():
    st.markdown("#### Dataset, class distribution and spatial split")
    c1, c2 = st.columns([3, 2])
    with c1:
        d = CLASS_DIST.sort_values("Train")
        fig, ax = plt.subplots(figsize=(7, 4.2))
        y = np.arange(len(d)); h = 0.27
        for k, (col, colr) in enumerate(zip(d.columns, [NAVY, STEEL, GREY])):
            ax.barh(y + (k - 1) * h, d[col], height=h, color=colr, label=col)
        ax.set_yticks(y); ax.set_yticklabels(d.index, fontsize=8); ax.set_xscale("log")
        ax.set_xlabel("% of labelled pixels (log scale)"); ax.grid(axis="x", alpha=0.3)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
        ax.set_title("Class distribution per split (10 m, all 12 months)", fontsize=10)
        plt.tight_layout(); st.pyplot(fig); plt.close(fig)
    with c2:
        st.dataframe(CLASS_DIST.style.format("{:.2f}"), use_container_width=True)
    st.markdown(
        "| Item | Setting |\n|---|---|\n"
        "| Input | Sentinel-1 GRD, IW, VV + VH, ascending, relative orbit 15 (Google Earth Engine) |\n"
        "| Labels | Google Dynamic World V1, 9 classes, monthly mean probability → argmax |\n"
        "| Area / period | Heidelberg – Rhine-Neckar, ≈ 40 km × 40 km, UTM 32N; January–December 2025 |\n"
        "| Compositing | common VV/VH mask, mean in linear power → dB, 2–8 scenes per month (temporal multi-looking) |\n"
        "| Inputs to the model | VV, VH, VV−VH (dB), clipped and standardised with training-split statistics |\n"
        "| Patches | 256 × 256 px at 10 m (2.56 km); 64 × 64 px at 40 m (majority label + purity) |\n"
        "| Split | 25 super-blocks of 7.68 km: 17 train / 4 validation / 4 test, same for all months |\n"
        "| Patches (month × block) | 1,670 train · 399 validation · 404 test |\n"
        "| Leakage check | 0 blocks shared between splits |\n"
        "| Augmentation | random 90° rotations + horizontal flip (training only) |")
    finding("The data are strongly imbalanced: trees, built-up and crops make up 87 % of the pixels, flooded "
            "vegetation only 0.01–0.03 %. The spatial split keeps the major classes within 2 pp between splits, "
            "so the test areas are representative, while the rare classes vary by up to a factor of two.")


def findings_page():
    ov, pc = load_results()
    st.markdown("## Experiments & findings")
    st.markdown("Summary of all experiments behind the prototype: resolution transfer, fitting behaviour, error "
                "analysis, input experiments, fine-tuning and the effect of temporal smoothing on change "
                "estimates. All scores are agreement with Google Dynamic World on spatially separate test areas.")
    tabs = st.tabs(["0 · Summary", "1 · Data", "2 · Resolution", "3 · Fitting", "4 · Errors",
                    "5 · Inputs", "6 · Fine-tuning", "7 · Smoothing"])
    with tabs[0]: tab_summary()
    with tabs[1]: tab_dataset()
    with tabs[2]: tab_resolution(ov)
    with tabs[3]: tab_fitting(ov)
    with tabs[4]: tab_errors(pc)
    with tabs[5]: tab_ablation()
    with tabs[6]: tab_finetuning()
    with tabs[7]: tab_smoothing()


if page == "Overview":
    overview_page(); footer(); st.stop()
if page == "Model comparison":
    comparison_page(); footer(); st.stop()
if page == "Experiments & findings":
    findings_page(); footer(); st.stop()


# ------------------------------------------------------------------ page: run prototype
class LiveLog(io.StringIO):
    """Shows the prototype's progress messages live in the page."""
    def __init__(self, box):
        super().__init__()
        self.box, self.lines = box, []

    def write(self, s):
        if s.strip():
            self.lines.append(s.rstrip())
            self.box.code("\n".join(self.lines[-20:]), language=None)
        return len(s)


st.markdown("## Run prototype")
if run_clicked:
    start = f"{year}-{month:02d}"
    try:
        import ee
        ee.Initialize(project=project)
    except Exception as e:
        st.error("Could not connect to Google Earth Engine. Check two things:\n\n"
                 "1. You are signed in: run `earthengine authenticate` in a terminal "
                 "(on a remote server: `earthengine authenticate --auth_mode=notebook`).\n"
                 "2. The project ID under **Advanced settings** is a Google Cloud project registered for "
                 "Earth Engine that your account can use.\n\n"
                 f"Details: {e}")
        footer(); st.stop()
    with st.status(f"Processing ({lat:.4f}, {lon:.4f}), 12 months from {start} …", expanded=True) as status:
        box = st.empty()
        try:
            with contextlib.redirect_stdout(LiveLog(box)):
                res = P.run(lat, lon, start, models[model_name], project, size_km=size_km,
                            out_dir="results", worldcover=do_wc, input_res=input_res)
            plt.close("all")
            st.session_state["res"] = res
            st.session_state["smooth_cache"] = {}
            st.session_state["meta"] = {"lat": lat, "lon": lon, "start": start, "size_km": size_km,
                                        "city": city, "model": model_name, "input_res": input_res}
            status.update(label=f"Completed — 12 monthly maps for ({lat:.4f}, {lon:.4f})", state="complete",
                          expanded=False)
        except Exception as e:
            status.update(label="Run failed", state="error")
            st.error(str(e))
            footer(); st.stop()

res = st.session_state.get("res")
if res is None:
    st.info("Select a location, start year and start month in the sidebar, then press **Run analysis**.")
    footer(); st.stop()

meta = st.session_state["meta"]
current = (round(lat, 4), round(lon, 4), f"{year}-{month:02d}", size_km, model_name, input_res)
shown = (round(meta["lat"], 4), round(meta["lon"], 4), meta["start"], meta["size_km"], meta["model"],
         meta.get("input_res", 10))
if current != shown:
    st.warning("The results below belong to the **previous run** "
               f"({meta['city']}, {meta['start']}). The inputs in the sidebar have changed — "
               "press **Run analysis** to compute results for the new selection.")
place = meta["city"] if not meta["city"].startswith("Custom") else "Custom location"
st.markdown(f"**{place}** — {meta['lat']:.4f}° N, {meta['lon']:.4f}° E · {res['labels'][0]} to "
            f"{res['labels'][-1]} · {meta['size_km']} km × {meta['size_km']} km · "
            f"**{meta.get('input_res', 10)} m input** · Model: {meta['model']}")

SMOOTH = {"None (raw model output)": "raw",
          "3-month majority filter": "majority3",
          "Persistence rule (a change must last ≥ 2 months)": "persistence"}
choice = st.radio("Temporal post-processing of the 12 monthly maps", list(SMOOTH), horizontal=True,
                  help="Monthly maps flicker because of classification errors. Smoothing over time removes "
                       "short-lived class changes; the raw output is what the model produced.")
mode = SMOOTH[choice]
cache = st.session_state.setdefault("smooth_cache", {})


def gross_change(maps):
    """Mean share (%) of pixels that change class between consecutive months."""
    g = []
    for k in range(1, len(maps)):
        v = (maps[k - 1] != P.NODATA) & (maps[k] != P.NODATA)
        if v.any():
            g.append(100 * (maps[k - 1][v] != maps[k][v]).mean())
    return float(np.mean(g)) if g else float("nan")


if "raw_gross" not in cache:
    cache["raw_gross"] = gross_change(res["maps"])
if mode not in cache:
    if mode == "raw":
        cache[mode] = {"maps": res["maps"], "pct": res["pct"], "mom": res["mom"], "total": res["total"],
                       "fig_maps": res["figures"]["maps"], "fig_shares": res["figures"]["shares"]}
    else:
        stack = np.stack(res["maps"])
        maps_v = list(TS.majority3(stack) if mode == "majority3" else TS.persistence(stack))
        p_, m_, t_ = P.class_statistics(maps_v, res["labels"])
        cache[mode] = {"maps": maps_v, "pct": p_, "mom": m_, "total": t_,
                       "fig_maps": P.plot_maps(maps_v, res["labels"], f"Sentinel-1 LULC — {choice}"),
                       "fig_shares": P.plot_shares(p_, t_)}
        plt.close("all")
    cache[mode]["gross"] = gross_change(cache[mode]["maps"])
cur = cache[mode]
pct, mom, total = cur["pct"], cur["mom"], cur["total"]
if mode != "raw":
    st.caption(f"Pixels changing class between consecutive months: {cur['gross']:.1f} % on average "
               f"(raw output: {cache['raw_gross']:.1f} %). Smoothing is a post-processing step; "
               "the reported accuracy figures refer to the raw model output.")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Built-up, first month", f"{pct['Built-up'].iloc[0]:.1f} %")
c2.metric("Built-up, last month", f"{pct['Built-up'].iloc[-1]:.1f} %", f"{total['Built-up']:+.2f} pp",
          delta_color="off")
c3.metric("Sentinel-1 orbit", f"{res['orbit']} (ascending)")
if "worldcover_cmp" in res:
    c4.metric("Agreement with WorldCover", f"{100 * res['worldcover_cmp']['agreement']:.1f} %")

tabs = st.tabs(["Maps — 12 months", "Single month", "Class percentages", "Change (pp)",
                "WorldCover check", "Details & downloads"])

with tabs[0]:
    st.pyplot(cur["fig_maps"])

with tabs[1]:
    lab = st.select_slider("Month", options=res["labels"], value=res["labels"][0])
    k = res["labels"].index(lab)
    a = cur["maps"][k].astype(float)
    a[cur["maps"][k] == P.NODATA] = np.nan
    left, right = st.columns([3, 2])
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(a, cmap=CMAP, norm=NORM, interpolation="nearest")
    ax.set_title(f"Land cover {lab}"); ax.axis("off")
    ax.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in P.PALETTE], P.CLASS_NAMES,
              loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=8, frameon=False)
    left.pyplot(fig); plt.close(fig)
    right.markdown(f"**Class share in {lab} (% of area)**")
    right.dataframe(pct.loc[lab].round(2).rename("% of area").to_frame(), use_container_width=True)

with tabs[2]:
    st.markdown("Percentage of the valid area covered by each class, per month.")
    st.dataframe(pct.round(2), use_container_width=True)
    st.pyplot(cur["fig_shares"])

with tabs[3]:
    v = float(np.nanmax(np.abs(mom.values))) or 1.0
    st.markdown("**Month-to-month change** = share(month) − share(previous month), in percentage points (pp).")
    st.dataframe(mom.round(2).style.format("{:+.2f}", na_rep="—")
                 .background_gradient(cmap="RdBu", vmin=-v, vmax=v), use_container_width=True)
    st.markdown(f"**Total change** = share({res['labels'][-1]}) − share({res['labels'][0]}), in percentage points.")
    st.dataframe(total.round(2).to_frame().style.format("{:+.2f}")
                 .background_gradient(cmap="RdBu", vmin=-v, vmax=v), use_container_width=True)
    st.caption("Note: changes involving winter months, and changes of a few percentage points, are frequently "
               "caused by seasonal radar effects or classification error rather than real land-use change.")

with tabs[4]:
    if "worldcover_cmp" in res:
        cmp = res["worldcover_cmp"]
        st.markdown("Most frequent predicted class over the 12 months compared with **ESA WorldCover 2021**, "
                    "mapped to the 9 classes. WorldCover was **not** used for training. Differences also arise "
                    "from the different year and class definitions (e.g. street trees and gardens are 'Trees' in "
                    "WorldCover but part of 'Built-up' in Dynamic World).")
        st.dataframe(cmp["table"].round(2), use_container_width=True)
        st.pyplot(res["figures"]["worldcover"])
    else:
        st.info("The WorldCover check was switched off for this run.")

with tabs[5]:
    st.markdown(f"**Sentinel-1:** relative orbit {res['orbit']}, ascending; scenes per orbit: "
                f"{res['orbit_histogram']}")
    st.markdown("**Scenes per month:** " + ", ".join(f"{k}: {n}" for k, n in zip(res["labels"], res["n_images"])))
    st.markdown(f"**Outputs saved in:** `{res['out_dir']}` (12 GeoTIFFs, figures, CSV tables)")
    d1, d2, d3 = st.columns(3)
    d1.download_button("Class percentages (CSV)", pct.round(4).to_csv(), "class_percent_per_month.csv")
    d2.download_button("Month-to-month change (CSV)", mom.round(4).to_csv(), "month_to_month_change_pp.csv")
    d3.download_button("Total change (CSV)", total.round(4).to_csv(), "total_change_pp.csv")

footer()
