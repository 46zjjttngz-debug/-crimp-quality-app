"""
crimp_app.py — Interactive dashboard for the TAC 1 crimp quality project.

Sections:
  1. Process overview (raw data + SPC control chart)
  2. Detection model (same-cycle pass/fail classification)
  3. Forecasting model (will this line fail within the next N cycles?) with
     a live alert stream and lead-time metric
  4. Try it yourself — enter readings for a new cycle and get both a
     detection verdict and a forecast alert. Includes a "sustained cycles"
     control so a single hypothetical reading can be simulated as a run of
     consecutive cycles.

NOTE on the detection model: it uses a StandardScaler + LogisticRegression
pipeline. Without scaling, pull_force (raw scale ~0-100) completely dominated
crimp_height (raw scale ~0.5) in the fitted coefficients -- the model would
essentially ignore crimp_height/height_deviation unless pull_force was also
moved, which made the "Try it yourself" tab look broken when only the height
slider was changed. Scaling fixes that.

Run with:  streamlit run crimp_app.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

st.set_page_config(page_title="TAC 1 — Crimp Quality Monitoring", layout="wide")

CSV_PATH = "crimp_data_v2.csv"
GOLDEN_HEIGHT = 0.500
HORIZON = 20
ROLL_WINDOW = 20
HEALTHY_REF_CYCLES = 300
TEST_FRACTION = 0.20


@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH)
    df["pass_fail_bin"] = (
        df["pass_fail"].map({"pass": 1, "fail": 0})
        if df["pass_fail"].dtype == object else df["pass_fail"]
    ).astype(int)
    df["height_deviation"] = (df["crimp_height"] - GOLDEN_HEIGHT).abs()
    df["height_rolling_mean"] = df["crimp_height"].rolling(ROLL_WINDOW).mean()
    df["height_rolling_std"] = df["crimp_height"].rolling(ROLL_WINDOW).std()
    df["peak_force_rolling_mean"] = df["peak_force"].rolling(ROLL_WINDOW).mean()
    df["pull_force_rolling_mean"] = df["pull_force"].rolling(ROLL_WINDOW).mean()
    return df


@st.cache_data
def compute_spc(df):
    healthy = df.iloc[:HEALTHY_REF_CYCLES]
    sigma = healthy["crimp_height"].std()
    ucl, lcl = GOLDEN_HEIGHT + 3 * sigma, GOLDEN_HEIGHT - 3 * sigma
    warn_up, warn_lo = GOLDEN_HEIGHT + 2 * sigma, GOLDEN_HEIGHT - 2 * sigma
    drift_alert = (df["height_rolling_mean"] > ucl) | (df["height_rolling_mean"] < lcl)
    drift_warning = (df["height_rolling_mean"] > warn_up) | (df["height_rolling_mean"] < warn_lo)
    first_warning = df.loc[drift_warning, "cycle_id"].min()
    first_alert = df.loc[drift_alert, "cycle_id"].min()
    return dict(sigma=sigma, ucl=ucl, lcl=lcl, warn_up=warn_up, warn_lo=warn_lo,
                drift_alert=drift_alert, drift_warning=drift_warning,
                first_warning=first_warning, first_alert=first_alert)


@st.cache_resource
def train_detection_model(df):
    features = ["peak_force", "crimp_height", "pull_force",
                "height_deviation", "height_rolling_mean", "height_rolling_std"]
    d = df.dropna(subset=features).reset_index(drop=True)
    X, y = d[features], d["pass_fail_bin"].map({1: 0, 0: 1})  # 1 = fail, for consistency with report
    split = int(len(d) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # StandardScaler is essential here: pull_force (~0-100) and crimp_height
    # (~0.5) are on wildly different raw scales, and unscaled LogisticRegression
    # let pull_force's coefficient swamp everything else.
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    report = classification_report(y_test, pred, target_names=["pass", "fail"],
                                    output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, pred)
    return model, features, report, cm


@st.cache_resource
def train_forecasting_model(df):
    fail_flag = (df["pass_fail_bin"] == 0).astype(int).to_numpy()
    n = len(df)
    will_fail_soon = np.full(n, np.nan)
    for t in range(n - HORIZON):
        will_fail_soon[t] = 1 if fail_flag[t + 1: t + 1 + HORIZON].sum() > 0 else 0
    df = df.copy()
    df["will_fail_soon"] = will_fail_soon

    features = ["peak_force", "crimp_height", "pull_force", "cycle_time",
                "height_deviation", "height_rolling_mean", "height_rolling_std",
                "peak_force_rolling_mean", "pull_force_rolling_mean"]
    d = df.dropna(subset=features + ["will_fail_soon"]).reset_index(drop=True)

    split = int(len(d) * (1 - TEST_FRACTION))
    train_df, test_df = d.iloc[:split], d.iloc[split:].reset_index(drop=True)
    X_train, y_train = train_df[features], train_df["will_fail_soon"]
    X_test, y_test = test_df[features], test_df["will_fail_soon"]

    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42)
    rf.fit(X_train, y_train)
    proba_rf = rf.predict_proba(X_test)[:, 1]

    precision_rf, recall_rf, thresholds = precision_recall_curve(y_test, proba_rf)
    target_recall = 0.70
    candidate = [(t, p, r) for t, p, r in zip(thresholds, precision_rf[:-1], recall_rf[:-1]) if r >= target_recall]
    threshold = max(candidate, key=lambda x: x[0])[0] if candidate else 0.5

    alerts = test_df.loc[proba_rf >= threshold, ["cycle_id"]].copy()
    alerts["probability"] = proba_rf[proba_rf >= threshold]
    alerts["actual_will_fail_soon"] = y_test[proba_rf >= threshold].values

    pred_tuned = (proba_rf >= threshold).astype(int)
    report = classification_report(y_test, pred_tuned, target_names=["no_fail_soon", "fail_soon"],
                                    output_dict=True, zero_division=0)

    first_true_alert = alerts.loc[alerts["actual_will_fail_soon"] == 1, "cycle_id"].min()
    first_actual_fail = test_df.loc[test_df["pass_fail_bin"] == 0, "cycle_id"].min()

    return dict(model=rf, features=features, test_df=test_df, proba=proba_rf,
                threshold=threshold, alerts=alerts, report=report,
                first_true_alert=first_true_alert, first_actual_fail=first_actual_fail)


# ---------------------------------------------------------------------------
df = load_data()
spc = compute_spc(df)
det_model, det_features, det_report, det_cm = train_detection_model(df)
fc = train_forecasting_model(df)

st.title("TAC 1 — Crimp Quality Monitoring")
st.caption("Proactive defect prevention: SPC drift detection + detection model + forecasting model")

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Process & SPC", "🔍 Detection model", "🔮 Forecasting model", "🧪 Try it yourself"
])

# --- Tab 1: SPC ---------------------------------------------------------
with tab1:
    st.subheader("Crimp height — SPC control chart")
    c1, c2, c3 = st.columns(3)
    c1.metric("Process sigma (healthy period)", f"{spc['sigma']:.4f} mm")
    c2.metric("First drift WARNING", f"cycle {int(spc['first_warning'])}" if pd.notna(spc['first_warning']) else "none")
    c3.metric("First drift ALERT", f"cycle {int(spc['first_alert'])}" if pd.notna(spc['first_alert']) else "none")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["cycle_id"], df["crimp_height"], lw=0.4, alpha=0.35, color="gray", label="crimp_height (raw)")
    ax.plot(df["cycle_id"], df["height_rolling_mean"], lw=1.8, color="tab:blue", label="rolling mean (20)")
    ax.axhline(GOLDEN_HEIGHT, color="green", label=f"golden height {GOLDEN_HEIGHT}")
    ax.axhline(spc["ucl"], color="red", ls="--", label="control limits (±3σ)")
    ax.axhline(spc["lcl"], color="red", ls="--")
    ax.axhline(spc["warn_up"], color="orange", ls=":", label="warning limits (±2σ)")
    ax.axhline(spc["warn_lo"], color="orange", ls=":")
    fails = df[df["pass_fail_bin"] == 0]
    ax.scatter(fails["cycle_id"], fails["crimp_height"], s=10, color="red", zorder=3, label="fail")
    ax.set_xlabel("cycle_id"); ax.set_ylabel("crimp height (mm)")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.subheader("Raw data sample")
    st.dataframe(df[["cycle_id", "peak_force", "crimp_height", "pull_force", "cycle_time", "pass_fail"]].tail(10),
                 use_container_width=True)

# --- Tab 2: Detection model ---------------------------------------------
with tab2:
    st.subheader("Detection model — is THIS completed cycle defective?")
    st.caption("Logistic Regression (scaled features), same-cycle features, 80/20 split")
    c1, c2, c3 = st.columns(3)
    c1.metric("Fail recall", f"{det_report['fail']['recall']:.2f}")
    c2.metric("Fail precision", f"{det_report['fail']['precision']:.2f}")
    c3.metric("Overall accuracy", f"{det_report['accuracy']:.2f}")

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(det_cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(det_cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["pred_pass", "pred_fail"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["actual_pass", "actual_fail"])
    st.pyplot(fig)

# --- Tab 3: Forecasting model -------------------------------------------
with tab3:
    st.subheader(f"Forecasting model — will it fail within the next {HORIZON} cycles?")
    st.caption("Random Forest, trailing-only features, chronological 80/20 split — trained on the past, tested only on the future")

    c1, c2, c3 = st.columns(3)
    c1.metric("Fail-soon precision", f"{fc['report']['fail_soon']['precision']:.2f}")
    c2.metric("Fail-soon recall", f"{fc['report']['fail_soon']['recall']:.2f}")
    lead = None
    if pd.notna(fc["first_true_alert"]) and pd.notna(fc["first_actual_fail"]):
        lead = int(fc["first_actual_fail"] - fc["first_true_alert"])
    c3.metric("Lead time (first correct alert vs. first failure)", f"{lead} cycles" if lead is not None else "n/a")

    fig, ax = plt.subplots(figsize=(12, 5))
    test_df = fc["test_df"]
    ax.plot(test_df["cycle_id"], fc["proba"], lw=1.2, color="tab:blue", label="P(fail within next 20 cycles)")
    ax.axhline(fc["threshold"], color="black", ls="--", lw=1, label=f"alert threshold ({fc['threshold']:.2f})")
    true_fails = test_df[test_df["pass_fail_bin"] == 0]
    ax.scatter(true_fails["cycle_id"], [0.02] * len(true_fails), marker="x", color="red", s=30, label="actual fail")
    alert_rows = test_df.loc[fc["proba"] >= fc["threshold"]]
    ax.scatter(alert_rows["cycle_id"], fc["proba"][fc["proba"] >= fc["threshold"]], color="orange", s=15, label="alert raised")
    ax.set_xlabel("cycle_id"); ax.set_ylabel("P(fail_soon)")
    ax.legend(loc="upper left", fontsize=8); ax.grid(alpha=0.3)
    st.pyplot(fig)

    st.subheader("Alert log (test period)")
    display_alerts = fc["alerts"].copy()
    display_alerts["status"] = np.where(display_alerts["actual_will_fail_soon"] == 1, "✅ correct", "⚠️ false alarm")
    st.dataframe(display_alerts[["cycle_id", "probability", "status"]].reset_index(drop=True),
                 use_container_width=True, height=300)

# --- Tab 4: Try it yourself ----------------------------------------------
with tab4:
    st.subheader("Try it yourself — score a new cycle")
    st.caption("Enter readings for a new crimp cycle. The app runs both models: "
               "detection (is this part OK?) and forecasting (is the process trending toward failure?).")

    recent = df.tail(ROLL_WINDOW).reset_index(drop=True)
    col1, col2 = st.columns(2)
    with col1:
        peak_force = st.slider("peak_force (N)", 300.0, 1400.0, float(df["peak_force"].tail(50).mean()))
        crimp_height = st.slider("crimp_height (mm)", 0.40, 0.60, float(GOLDEN_HEIGHT), step=0.001, format="%.3f")
        pull_force = st.slider("pull_force (N)", 0.0, 100.0, float(df["pull_force"].tail(50).mean()))
        cycle_time = st.slider("cycle_time (s)", 2.0, 2.7, float(df["cycle_time"].tail(50).mean()))
        persist_cycles = st.slider(
            "How many CONSECUTIVE cycles at this reading? (simulate sustained drift)",
            1, ROLL_WINDOW, 1,
            help="A single one-off reading barely moves a 20-cycle rolling average -- that's correct "
                 "behavior for the FORECAST side. Raise this to simulate the tool holding this reading "
                 "for several cycles in a row, the way real tool wear behaves."
        )
        st.caption("Tip: crimp_height, peak_force and pull_force are physically linked in the real process "
                   "(a bad crimp height also produces a bad pull force). Move pull_force down together with "
                   "crimp_height for a realistic 'bad cycle' — dragging height alone while leaving pull_force "
                   "at a healthy value describes a combination the model rarely saw in training.")

    # Build a simulated 20-cycle window: keep the oldest (ROLL_WINDOW - persist_cycles)
    # real cycles, then replace the most recent `persist_cycles` with the new reading.
    keep = max(ROLL_WINDOW - persist_cycles, 0)
    sim_height = pd.concat([recent["crimp_height"].iloc[:keep], pd.Series([crimp_height] * persist_cycles)])
    sim_peak = pd.concat([recent["peak_force"].iloc[:keep], pd.Series([peak_force] * persist_cycles)])
    sim_pull = pd.concat([recent["pull_force"].iloc[:keep], pd.Series([pull_force] * persist_cycles)])

    height_deviation = abs(crimp_height - GOLDEN_HEIGHT)
    height_rolling_mean = sim_height.mean()
    height_rolling_std = sim_height.std() if len(sim_height) > 1 else 0.0
    peak_force_rolling_mean = sim_peak.mean()
    pull_force_rolling_mean = sim_pull.mean()

    with col2:
        st.write("**Derived trend features** (rolling window with the last "
                 f"{persist_cycles} cycle(s) replaced by your input):")
        st.write(f"- height_deviation: `{height_deviation:.4f}`")
        st.write(f"- height_rolling_mean: `{height_rolling_mean:.4f}`")
        st.write(f"- height_rolling_std: `{height_rolling_std:.4f}`")
        st.write(f"- peak_force_rolling_mean: `{peak_force_rolling_mean:.1f}`")
        st.write(f"- pull_force_rolling_mean: `{pull_force_rolling_mean:.1f}`")

    det_X = pd.DataFrame([[peak_force, crimp_height, pull_force, height_deviation,
                            height_rolling_mean, height_rolling_std]], columns=det_features)
    det_pred = det_model.predict(det_X)[0]
    det_proba = det_model.predict_proba(det_X)[0][1]

    fc_X = pd.DataFrame([[peak_force, crimp_height, pull_force, cycle_time, height_deviation,
                           height_rolling_mean, height_rolling_std,
                           peak_force_rolling_mean, pull_force_rolling_mean]], columns=fc["features"])
    fc_proba = fc["model"].predict_proba(fc_X)[0][1]

    st.divider()
    r1, r2 = st.columns(2)
    with r1:
        if det_pred == 1:
            st.error(f"🔴 DETECTION: this cycle is predicted DEFECTIVE (P(fail)={det_proba:.2f})")
        else:
            st.success(f"🟢 DETECTION: this cycle is predicted OK (P(fail)={det_proba:.2f})")
    with r2:
        if fc_proba >= fc["threshold"]:
            st.warning(f"⚠️ FORECAST ALERT: process trending toward failure within the next {HORIZON} cycles "
                       f"(P={fc_proba:.2f}, threshold={fc['threshold']:.2f})")
        else:
            st.info(f"✅ FORECAST: no early-warning signal (P={fc_proba:.2f}, threshold={fc['threshold']:.2f})")
