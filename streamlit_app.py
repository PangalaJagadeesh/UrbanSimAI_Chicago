
import json
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import joblib
import holidays
import streamlit as st
import pydeck as pdk

st.set_page_config(page_title="UrbanSimAI Chicago", layout="wide")

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
MODELS_DIR = BASE / "models"

MODEL_PATH = MODELS_DIR / "best_speed_model.pkl"
META_PATH  = MODELS_DIR / "best_speed_model_meta.json"

TRAFFIC_ZIP_HOURLY = DATA_DIR / "traffic_zip_hourly.parquet"
CTA_ZIP = DATA_DIR / "cta_stops_by_zip.parquet"
ZILLOW_ZIP = DATA_DIR / "zillow_zhvi_chicago_zip_long.parquet"
ZIP_GEOJSON = DATA_DIR / "zip_boundaries.geojson"

US_HOL = holidays.US(state="IL")

@st.cache_resource
def load_model():
    model = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text())
    return model, meta

@st.cache_data
def load_data():
    traffic = pd.read_parquet(TRAFFIC_ZIP_HOURLY)
    cta = pd.read_parquet(CTA_ZIP)
    zillow = pd.read_parquet(ZILLOW_ZIP)

    traffic["zip"] = traffic["zip"].astype(str).str.zfill(5)
    cta["zip"] = cta["zip"].astype(str).str.zfill(5)
    zillow["zip"] = zillow["zip"].astype(str).str.zfill(5)

    traffic["hour_ts"] = pd.to_datetime(traffic["hour_ts"])
    zillow["date"] = pd.to_datetime(zillow["date"])
    return traffic, cta, zillow

def time_features(ts: pd.Timestamp):
    ts = pd.to_datetime(ts)
    return {
        "hour": int(ts.hour),
        "dow": int(ts.dayofweek),
        "month": int(ts.month),
        "weekofyear": int(ts.isocalendar().week),
        "is_weekend": int(ts.dayofweek >= 5),
        "is_holiday": int(ts.date() in US_HOL),
    }

def get_static_features(zip_code, cta_df, zillow_df, ts):
    row = cta_df[cta_df["zip"] == zip_code]
    cta_cnt = float(row["cta_stop_count"].iloc[0]) if len(row) else 0.0

    z = zillow_df[zillow_df["zip"] == zip_code].sort_values("date")
    if len(z) == 0:
        zhvi = float(zillow_df["zhvi"].median())
    else:
        z2 = z[z["date"] <= ts]
        zhvi = float(z2["zhvi"].iloc[-1]) if len(z2) else float(z["zhvi"].iloc[0])
    return cta_cnt, zhvi

def build_feature_row(zip_code, ts, hist_df, cta_df, zillow_df):
    h = hist_df.sort_values("hour_ts").copy()

    def lag(hours_back):
        target = ts - timedelta(hours=hours_back)
        m = h[h["hour_ts"] == target]
        return float(m["avg_speed"].iloc[0]) if len(m) else np.nan

    prev24 = h[h["hour_ts"] < ts].tail(24)["avg_speed"].astype(float).to_numpy()

    cta_cnt, zhvi = get_static_features(zip_code, cta_df, zillow_df, ts)
    tf = time_features(ts)

    row = {
        "zip": zip_code,
        "cta_stop_count": cta_cnt,
        "zhvi": zhvi,
        **tf,
        "lag_1": lag(1),
        "lag_2": lag(2),
        "lag_3": lag(3),
        "lag_24": lag(24),
        "lag_168": lag(168),
        "roll_mean_3": float(np.mean(prev24[-3:])) if len(prev24) >= 3 else np.nan,
        "roll_mean_6": float(np.mean(prev24[-6:])) if len(prev24) >= 6 else np.nan,
        "roll_mean_24": float(np.mean(prev24)) if len(prev24) >= 2 else np.nan,
        "roll_std_24": float(np.std(prev24)) if len(prev24) >= 2 else np.nan,
    }
    return pd.DataFrame([row])

def align_onehot(X, model):
    X = pd.get_dummies(X, columns=["zip"], drop_first=False)
    if hasattr(model, "feature_names_in_"):
        return X.reindex(columns=list(model.feature_names_in_), fill_value=0)
    return X

def forecast_next_24(zip_code, traffic_df, cta_df, zillow_df, model):
    h = traffic_df[traffic_df["zip"] == zip_code][["hour_ts", "avg_speed"]].copy()
    h = h.sort_values("hour_ts").tail(2000).copy()

    last_ts = h["hour_ts"].max()
    preds = []
    cur = h.copy()

    for step in range(1, 25):
        ts = last_ts + timedelta(hours=step)
        X = build_feature_row(zip_code, ts, cur, cta_df, zillow_df)
        if X.isna().any().any():
            break
        Xa = align_onehot(X, model)
        yhat = float(model.predict(Xa)[0])
        preds.append({"hour_ts": ts, "pred_speed": yhat})
        cur = pd.concat([cur, pd.DataFrame([{"hour_ts": ts, "avg_speed": yhat}])], ignore_index=True)

    return pd.DataFrame(preds)

def load_geojson():
    import json as _json
    return _json.loads(ZIP_GEOJSON.read_text())

def build_map(geojson, value_by_zip):
    feats = geojson["features"]
    vals = []
    for f in feats:
        z = str(f["properties"].get("zip", "")).zfill(5)
        v = value_by_zip.get(z, None)
        f["properties"]["speed"] = None if v is None else float(v)
        if v is not None:
            vals.append(float(v))

    vmin = float(np.min(vals)) if vals else 0.0
    vmax = float(np.max(vals)) if vals else 1.0
    if vmax == vmin:
        vmax = vmin + 1.0

    layer = pdk.Layer(
        "GeoJsonLayer",
        geojson,
        stroked=True,
        filled=True,
        opacity=0.65,
        get_line_color=[60, 60, 60],
        get_fill_color=f"[255*(speed-{vmin})/({vmax}-{vmin}), 120, 255*(1-(speed-{vmin})/({vmax}-{vmin}))]",
        pickable=True,
    )
    view = pdk.ViewState(latitude=41.8781, longitude=-87.6298, zoom=9)
    return pdk.Deck(layers=[layer], initial_view_state=view, tooltip={"text": "ZIP: {zip}\nSpeed: {speed} mph"})

# ---------------- UI ----------------
st.title("UrbanSimAI Chicago — Traffic Speed Forecasting")
st.caption("XGBoost model + open data. Forecast next 24 hours by ZIP + map view.")

model, meta = load_model()
traffic_df, cta_df, zillow_df = load_data()

zip_list = sorted(traffic_df["zip"].unique().tolist())

c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    zip_code = st.selectbox("Select ZIP", zip_list, index=0)
with c2:
    mode = st.radio("Mode", ["Forecast next 24 hours", "Backtest (pick an hour)"])
with c3:
    st.write("**Best model:**", meta.get("best_model", "XGBoost"))
    st.write("**Test MAE:**", round(meta.get("final_test_mae", 0.0), 3),
             "| **RMSE:**", round(meta.get("final_test_rmse", 0.0), 3))

if mode == "Backtest (pick an hour)":
    sub = traffic_df[traffic_df["zip"] == zip_code].sort_values("hour_ts")
    ts = st.selectbox("Pick an hour", sub["hour_ts"].tail(500).tolist())
    X = build_feature_row(zip_code, pd.to_datetime(ts), sub[["hour_ts","avg_speed"]], cta_df, zillow_df)

    if X.isna().any().any():
        st.warning("Not enough history for lag features here. Pick a later hour.")
    else:
        pred = float(model.predict(align_onehot(X, model))[0])
        actual = float(sub[sub["hour_ts"] == pd.to_datetime(ts)]["avg_speed"].iloc[0])
        st.metric("Predicted speed (mph)", round(pred, 2))
        st.metric("Actual speed (mph)", round(actual, 2))
        st.dataframe(X)
else:
    pred_df = forecast_next_24(zip_code, traffic_df, cta_df, zillow_df, model)
    if len(pred_df) == 0:
        st.warning("Not enough history to forecast. Try another ZIP.")
    else:
        st.subheader(f"Next 24 hours forecast — ZIP {zip_code}")
        st.dataframe(pred_df)
        st.line_chart(pred_df.set_index("hour_ts")["pred_speed"])

st.divider()
st.subheader("Map: Latest observed hourly avg speed by ZIP")
last_hour = traffic_df["hour_ts"].max()
snap = traffic_df[traffic_df["hour_ts"] == last_hour][["zip", "avg_speed"]].copy()
value_by_zip = dict(zip(snap["zip"], snap["avg_speed"]))

geojson = load_geojson()
deck = build_map(geojson, value_by_zip)
st.pydeck_chart(deck)
st.caption(f"Latest hour in dataset: {last_hour}")

st.divider()
st.caption("Disclaimer: research/demo only. Do not use for safety-critical decisions.")
