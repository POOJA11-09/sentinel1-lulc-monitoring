# Sentinel-1 LULC Prototype — monthly land-cover maps from radar only

Given a **latitude/longitude** and a **start month**, the prototype downloads Sentinel-1 radar data from
Google Earth Engine, builds **12 monthly composites**, runs a trained deep-learning model (Sentinel-1 is the
**only** input) and returns:

- 12 monthly land-use / land-cover (LULC) maps (9 Dynamic World classes)
- the **percentage of the area** covered by each class in each month
- the **month-to-month change** and the **total change** (first → last month), in **percentage points**
- an **independent sanity check** against ESA WorldCover 2021 (not used for training)
- optional **temporal smoothing** of the 12 maps (3-month majority or persistence rule)
- **10 m or 40 m input** (40 m = 4 × 4 average of the 10 m composites in linear power, as for the 40 m training data)
- a **model comparison** page and an **experiments & findings** page with all test results

Classes: 0 Water · 1 Trees · 2 Grass · 3 Flooded vegetation · 4 Crops · 5 Shrub and scrub · 6 Built-up ·
7 Bare ground · 8 Snow and ice

---

## 1. What you need

| What | Notes |
|---|---|
| **Python 3.10–3.12** (tested: 3.11) | Windows, macOS or Linux. Download: <https://www.python.org/downloads/> |
| **A Google account registered for Google Earth Engine** | free for noncommercial / research use: <https://code.earthengine.google.com/register> |
| **A Google Cloud project enabled for Earth Engine** | created during registration; you need its **project ID** (e.g. `my-ee-project`). No billing or payment card is needed for noncommercial use. |
| about 3 GB free disk space | for Python packages (mainly PyTorch) |
| internet connection | data are streamed from Earth Engine |

No API key file is needed. You sign in once with your own Google account (step 3). A GPU is **not** required.

**Model weights:** download `best_model_segformer_10m.pt` from the model-weights link in the main README and save it as
`prototype/model/best_model.pt`. For the 40 m mode or other architectures, unzip the other weight files into a
folder `models/` next to `prototype/` (one sub-folder per model, as in the zip files).

## 2. Install (once)

Open a terminal (Windows: *Command Prompt* or *PowerShell*) **in this folder** and run:

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

(On macOS/Linux use `python3` if `python` is not found.)

## 3. Sign in to Earth Engine (once per computer)

```bash
earthengine authenticate
```

A browser window opens: choose your Google account and allow access. When it says *"Successfully saved
authorization token"*, you are done. On a remote server without a browser use
`earthengine authenticate --auth_mode=notebook` and paste the code shown in the browser.

## 4. Run

### Option A — web app (recommended)

```bash
streamlit run app.py
```

The app opens in your browser at <http://localhost:8501>.

1. In the sidebar choose a **location** (or *Custom coordinates*), the **start year and month**, the area size,
   the **input resolution** (10 m or 40 m) and the **model**.
2. Under **Advanced settings**, enter **your own Earth Engine project ID**.
3. Press **Run analysis** (≈1–2 min with a GPU, ≈3–6 min on a laptop CPU for a 10 km × 10 km area).
4. Explore the tabs: 12 maps, single month, class percentages, changes (pp), WorldCover check, downloads.
   The radio buttons above the tabs switch between raw maps and temporally smoothed maps.
5. **Model comparison** shows the test results of all models and which model fits which data;
   **Experiments & findings** summarises all experiments (resolution, fitting, errors, inputs, fine-tuning, smoothing).

Tip: set the project once with an environment variable instead of typing it each time —
Windows `set EE_PROJECT=my-ee-project`, macOS/Linux `export EE_PROJECT=my-ee-project` — then `streamlit run app.py`.

### Option B — notebook

Open `07_prototype.ipynb` (Frankfurt case study) in JupyterLab (`jupyter lab`), set `LAT`, `LON`, `START_MONTH`,
`GEE_PROJECT` and `INPUT_RES` in the parameter cell, then *Run → Run All Cells*. `07_prototype_mannheim.ipynb` and
`07_prototype_cairo.ipynb` are the same notebook run for Mannheim and Cairo. All tables and figures in a notebook are
computed from that notebook's own run.

### Option C — command line

```bash
python lulc_prototype.py --lat 50.1109 --lon 8.6821 --start 2025-01 --project my-ee-project
# 40 m input with a 40 m or mixed model:
python lulc_prototype.py --lat 50.1109 --lon 8.6821 --start 2025-01 --project my-ee-project --input-res 40 --weights <path to best.pt>
```

## 5. Folder contents

```
prototype/
├── app.py                        # Streamlit web app
├── lulc_prototype.py             # all processing code (also usable from the command line)
├── temporal_smoothing.py         # temporal post-processing + smoothing analysis over saved runs
├── 07_prototype.ipynb            # notebook interface — Frankfurt case study
├── 07_prototype_mannheim.ipynb   # same notebook, Mannheim
├── 07_prototype_cairo.ipynb      # same notebook, Cairo
├── collect_results.sh            # copies evaluation tables into evaluation/ (Data Lab helper)
├── requirements.txt
├── README.md
├── .streamlit/config.toml        # app colours
├── model/                        # put best_model.pt here (download link in the main README)
├── evaluation/                   # test results and figures shown on the comparison / experiments pages
└── example_results/              # tables and figures of the example runs (Frankfurt, Mannheim, Cairo, …)
```

Outputs of each run are written to `results/lat…_lon…_YYYY-MM/`: 12 GeoTIFF maps (open in QGIS), figures and
CSV tables (`class_percent_per_month.csv`, `month_to_month_change_pp.csv`, `total_change_pp.csv`,
`worldcover_comparison.csv`, `run_info.json`).

## 6. How it works

1. **Area and grid.** A square around the point in the local UTM zone, 10 m pixels, corners on a 40 m lattice.
2. **Data retrieval.** `COPERNICUS/S1_GRD` from Earth Engine — already calibrated to σ⁰, thermal-noise
   corrected, terrain corrected and in dB. Filters: IW mode, VV + VH, **ascending passes only**. Of the ascending
   relative orbits covering the area, the one with the most scenes in the 12 months is used, so all months share
   one viewing geometry (training used relative orbit 15).
3. **Monthly compositing.** Per scene, only pixels where both VV and VH are valid are kept; scenes are converted
   from dB to linear power, averaged over the month and converted back to dB. Averaging the 2–8 scenes of a month
   reduces speckle (temporal multi-looking); no additional spatial speckle filter is applied — identical to the
   training data (verified: 0.000 dB difference for Heidelberg, June 2025).
4. **Model.** Inputs VV, VH and VV−VH (dB), clipped and standardised with statistics from the training split.
   Prediction in overlapping 512 × 512 px tiles with blended probabilities (no tile seams).
5. **Statistics.** Share of a class = class pixels ÷ valid pixels × 100. Month-to-month change =
   share(month) − share(previous month); total change = share(last) − share(first); both in percentage points.
6. **Independent check.** ESA WorldCover 2021 mapped to the 9 classes, compared with the most frequent predicted
   class per pixel over the 12 months.
7. **40 m mode.** The 10 m composites are averaged over 4 × 4 pixels in linear power (as for the 40 m training
   data); WorldCover is aggregated by 4 × 4 majority.
8. **Temporal smoothing (optional).** *3-month majority*: each month takes the majority class of the previous,
   current and next month. *Persistence*: a pixel only changes class if the new class lasts at least two months.

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `Could not connect to Google Earth Engine` / `Please authorize access` | run `earthengine authenticate` again; check the project ID under *Advanced* |
| `Caller does not have required permission` / project not registered | use a Cloud project that is registered for Earth Engine under your account |
| `No ascending Sentinel-1 … scenes found` | location outside Sentinel-1 ascending coverage for that period — try another start month or location |
| `The 12 months … are not complete yet` | choose an earlier start month (all 12 months must already be over) |
| installing `rasterio` fails | upgrade pip (`pip install --upgrade pip`) and retry; use Python 3.10–3.12 |
| slow on a laptop | normal on CPU; use a smaller area (sidebar slider) |

## 8. Limitations

- Trained on **one region** (Heidelberg, temperate Central Europe, 2025) and **one orbit** (15, ascending);
  accuracy is lower for different landscapes, climates or viewing geometries.
- Reference labels are **Dynamic World** (Sentinel-2-based), not ground truth.
- **Seasonal effects** (leaf-off forest, wet/frozen soil, snow, harvested fields) cause large winter shifts that
  are mostly **not real land-use change**.
- With ≈17 % of pixels misclassified, month-to-month changes of a few percentage points are within the error.
- A model must be applied at the resolution it was trained on (10 m models → 10 m input, 40 m models → 40 m
  input; mixed models work for both).
- Output folders are named by location and resolution only; a new run of the same place overwrites the previous one.
