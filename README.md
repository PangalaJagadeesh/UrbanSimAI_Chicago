# UrbanSimAI Chicago — Traffic Speed Forecasting

[![Streamlit App](https://img.shields.io/badge/Live%20Demo-Streamlit-brightgreen)](https://urbansimaichicago-xvd9jjog6vmlzu9u4f2fjl.streamlit.app)
[![Map Section](https://img.shields.io/badge/View-Map-blue)](https://urbansimaichicago-xvd9jjog6vmlzu9u4f2fjl.streamlit.app/#map-latest-observed-hourly-avg-speed-by-zip)

Live demo:
- App: https://urbansimaichicago-xvd9jjog6vmlzu9u4f2fjl.streamlit.app
- Map section: https://urbansimaichicago-xvd9jjog6vmlzu9u4f2fjl.streamlit.app/#map-latest-observed-hourly-avg-speed-by-zip

## What this project does
UrbanSimAI Chicago predicts **hourly average traffic speed by ZIP code** for the **next 24 hours** using an **XGBoost** model trained on open Chicago data.  
The web app supports:
- ZIP selection
- 24-hour forecast chart + table
- Backtest mode
- Map visualization of the latest observed speeds by ZIP

## Model
- Best model: XGBoost
- Metrics shown inside the app (Test MAE / RMSE)
- Saved model: `models/best_speed_model.pkl`

## Data sources (open)
- City of Chicago Traffic Tracker (historical + latest snapshot)
- CTA GTFS (stops)
- Zillow ZHVI (ZIP home values)
- OpenStreetMap roads (OSMnx)

## Repo structure
- streamlit_app.py
- requirements.txt
- models/
- data/

## Run locally
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```
## Disclaimer
This is a research / portfolio project for educational purposes. Predictions may be inaccurate and should not be used for safety-critical decisions.
