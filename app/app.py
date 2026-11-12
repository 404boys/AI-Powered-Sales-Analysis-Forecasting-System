"""Streamlit interface for Rossmann sales forecasting and scenario analysis."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from features.build_features import build_features
from modeling.forecast import forecast_store
from modeling.predict import predict_sales

DATA_PATH = ROOT / "Data" / "Cleaned" / "store_sales_cleaned.csv"
MODEL_PATH = ROOT / "artifacts" / "model" / "lightgbm_sales_model.joblib"
METRICS_PATH = ROOT / "artifacts" / "model" / "metrics.json"


st.set_page_config(
    page_title="Rossmann Sales Intelligence",
    page_icon="📈",
    layout="wide",
)


@st.cache_data(show_spinner="Loading cleaned sales data...")
def load_raw_data(path_string: str) -> pd.DataFrame:
    """Load the cleaned source file once."""
    return pd.read_csv(path_string, low_memory=False)


@st.cache_data(show_spinner="Building historical forecasting features...")
def load_featured_data(path_string: str) -> pd.DataFrame:
    """Construct model-ready features from the cached raw dataset."""
    return build_features(load_raw_data(path_string))


@st.cache_resource(show_spinner="Loading trained forecasting model...")
def load_model(path_string: str):
    """Load the locally trained LightGBM model."""
    return joblib.load(path_string)


st.title("Rossmann Sales Intelligence")
st.caption("Store-level sales forecasting, model diagnostics, and scenario exploration")

if not DATA_PATH.exists():
    st.error(f"Cleaned dataset not found: {DATA_PATH}")
    st.info("Place store_sales_cleaned.csv inside Data/Cleaned/ and restart the app.")
    st.stop()

if not MODEL_PATH.exists():
    st.warning("The forecasting model has not been trained on this machine yet.")
    st.code(
        "pip install -r requirements.txt\n"
        "python src/modeling/train_lightgbm.py\n"
        "streamlit run app/app.py",
        language="powershell",
    )
    st.stop()

raw_data = load_raw_data(str(DATA_PATH))
featured = load_featured_data(str(DATA_PATH))
model = load_model(str(MODEL_PATH))

# Fail clearly if a stale or incompatible model artifact is supplied.
saved_feature_names = list(model.booster_.feature_name())
if saved_feature_names != FEATURE_COLUMNS:
    st.error(
        "The saved model does not match the current feature pipeline. "
        "Retrain it with: python src/modeling/train_lightgbm.py"
    )
    st.stop()

if featured.empty:
    st.error("The cleaned dataset produced no usable store-day rows.")
    st.stop()

# Sidebar navigation and shared selectors.
st.sidebar.header("Explore")
page = st.sidebar.radio(
    "Section",
    ["Overview", "Sales Explorer", "Future Forecast", "What-If Simulator", "Model Performance"],
)

minimum_date = featured["Date"].min().date()
maximum_date = featured["Date"].max().date()
stores = sorted(featured["Store"].dropna().unique().tolist())

st.sidebar.caption(f"Data coverage: {minimum_date:%d %b %Y} – {maximum_date:%d %b %Y}")
st.sidebar.caption(f"Stores in cleaned data: {len(stores):,}")

if page == "Overview":
    total_sales = featured["Sales"].sum()
    average_daily_sales = featured["Sales"].mean()
    average_store_day = featured.groupby(["Store", "Date"])["Sales"].sum().mean()
    store_count = featured["Store"].nunique()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Recorded sales", f"{total_sales:,.0f}")
    c2.metric("Average recorded row", f"{average_daily_sales:,.0f}")
    c3.metric("Stores", f"{store_count:,}")
    c4.metric("Open store-days", f"{len(featured):,}")

    st.subheader("Daily sales trend")
    daily = featured.groupby("Date", as_index=False)["Sales"].sum()
    fig = px.line(daily, x="Date", y="Sales", title="Total sales across all recorded open stores")
    fig.update_layout(xaxis_title=None, yaxis_title="Sales")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Sales by store type")
    store_type = featured.groupby("StoreType", as_index=False)["Sales"].mean()
    fig = px.bar(store_type, x="StoreType", y="Sales", title="Average sales per recorded store-day")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "The source file contains open, positive-sales store-days only. "
        "These totals are not a complete calendar-day total for closed stores."
    )

elif page == "Sales Explorer":
    st.subheader("Explore one store")
    selected_store = st.selectbox("Store", stores)
    store_data = featured.loc[featured["Store"] == selected_store].sort_values("Date")

    fig = px.line(
        store_data,
        x="Date",
        y="Sales",
        title=f"Recorded daily sales — Store {selected_store}",
    )
    st.plotly_chart(fig, use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.metric("Average recorded sales", f"{store_data['Sales'].mean():,.0f}")
    with right:
        st.metric("Total recorded sales", f"{store_data['Sales'].sum():,.0f}")

    st.dataframe(
        store_data[["Date", "Sales", "Promo", "Customers", "SchoolHoliday"]].tail(30),
        use_container_width=True,
        hide_index=True,
    )

elif page == "Future Forecast":
    st.subheader("Future Store Sales Forecast")
    st.write(
        "Generate a recursive daily forecast after the selected store's last recorded date. "
        "Each predicted day is fed into the next day's lag features."
    )

    selected_store = st.selectbox("Choose a store", stores, key="future_store")
    horizon = st.selectbox("Forecast horizon", [7, 14, 28], index=0)
    st.markdown("**Assumptions for every forecast day**")
    x, y, z = st.columns(3)
    with x:
        forecast_promo = st.selectbox(
            "Promotion",
            [0, 1],
            format_func=lambda value: "On" if value else "Off",
            key="future_promo",
        )
    with y:
        forecast_school_holiday = st.selectbox(
            "School holiday",
            [0, 1],
            format_func=lambda value: "Yes" if value else "No",
            key="future_school_holiday",
        )
    with z:
        forecast_state_holiday = st.selectbox(
            "State holiday",
            ["0", "a", "b", "c"],
            format_func=lambda value: "None" if value == "0" else value.upper(),
            key="future_state_holiday",
        )

    store_history = raw_data.loc[raw_data["Store"] == selected_store]
    last_store_date = pd.to_datetime(store_history["Date"]).max()
    st.caption(
        f"Forecast starts after {last_store_date:%d %b %Y}. "
        "The store is assumed to remain open, and the selected assumptions are held constant."
    )

    forecast = forecast_store(
        raw_data,
        model,
        store_id=selected_store,
        horizon=int(horizon),
        promo=int(forecast_promo),
        school_holiday=int(forecast_school_holiday),
        state_holiday=forecast_state_holiday,
    )
    total_forecast = forecast["Predicted_Sales"].sum()
    average_forecast = forecast["Predicted_Sales"].mean()
    k1, k2, k3 = st.columns(3)
    k1.metric("Forecast total", f"{total_forecast:,.0f}")
    k2.metric("Average daily forecast", f"{average_forecast:,.0f}")
    k3.metric("Forecast days", f"{len(forecast):,}")

    fig = px.line(
        forecast,
        x="Date",
        y="Predicted_Sales",
        markers=True,
        title=f"Recursive {horizon}-day sales forecast — Store {selected_store}",
    )
    fig.update_layout(xaxis_title=None, yaxis_title="Predicted sales")
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(forecast, use_container_width=True, hide_index=True)
    st.download_button(
        "Download forecast CSV",
        data=forecast.to_csv(index=False).encode("utf-8"),
        file_name=f"store_{selected_store}_{horizon}_day_forecast.csv",
        mime="text/csv",
    )
    st.warning(
        "This is a historical-data demonstration, not a live 2026 forecast. "
        "It assumes the store is open every forecast day and holds promotion/holiday "
        "assumptions constant. Forecast uncertainty is not currently quantified."
    )

elif page == "What-If Simulator":
    st.subheader("What-If Analysis")
    st.write(
        "Choose a historical store-day, then change its assumptions. "
        "The model recalculates the prediction while keeping its historical lag features fixed."
    )

    selected_store = st.selectbox("Choose a store", stores, key="scenario_store")
    store_data = featured.loc[featured["Store"] == selected_store].sort_values("Date")
    eligible_dates = store_data["Date"].dt.date.tolist()
    selected_date = st.selectbox(
        "Choose a date with recorded sales",
        eligible_dates,
        index=len(eligible_dates) - 1,
        key="scenario_date",
    )
    base_row = store_data.loc[store_data["Date"].dt.date == selected_date].iloc[0].copy()

    st.markdown("**Original assumptions**")
    a, b, c = st.columns(3)
    a.metric("Promotion", "On" if int(base_row["Promo"]) else "Off")
    b.metric("School holiday", "Yes" if int(base_row["SchoolHoliday"]) else "No")
    c.metric("State holiday", str(base_row["StateHoliday"]))

    st.markdown("**Change the scenario**")
    x, y, z = st.columns(3)
    with x:
        scenario_promo = st.selectbox(
            "Promotion",
            [0, 1],
            index=int(base_row["Promo"]),
            format_func=lambda value: "On" if value else "Off",
        )
    with y:
        scenario_school_holiday = st.selectbox(
            "School holiday",
            [0, 1],
            index=int(base_row["SchoolHoliday"]),
            format_func=lambda value: "Yes" if value else "No",
        )
    with z:
        state_options = ["0", "a", "b", "c"]
        original_state = str(base_row["StateHoliday"])
        if original_state not in state_options:
            state_options.append(original_state)
        scenario_state_holiday = st.selectbox(
            "State holiday",
            state_options,
            index=state_options.index(original_state),
            format_func=lambda value: "None" if value == "0" else value.upper(),
        )

    scenario_row = base_row.copy()
    scenario_row["Promo"] = int(scenario_promo)
    scenario_row["SchoolHoliday"] = int(scenario_school_holiday)
    scenario_row["StateHoliday"] = str(scenario_state_holiday)

    original_prediction = predict_sales(model, base_row)
    scenario_prediction = predict_sales(model, scenario_row)
    difference = scenario_prediction - original_prediction
    percentage = (difference / original_prediction * 100) if original_prediction else 0

    p1, p2, p3 = st.columns(3)
    p1.metric("Original model prediction", f"{original_prediction:,.0f}")
    p2.metric("Scenario prediction", f"{scenario_prediction:,.0f}", f"{difference:+,.0f}")
    p3.metric("Scenario change", f"{percentage:+.2f}%")

    comparison = pd.DataFrame(
        {
            "Scenario": ["Original assumptions", "Changed assumptions"],
            "Predicted sales": [original_prediction, scenario_prediction],
        }
    )
    fig = px.bar(
        comparison,
        x="Scenario",
        y="Predicted sales",
        text_auto=",.0f",
        title="Predicted sales comparison",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.metric("Actual recorded sales on selected date", f"{base_row['Sales']:,.0f}")
    st.warning(
        "This is a model-based scenario comparison, not proof that changing a promotion "
        "or holiday will cause this exact sales change. Historical lag features are held fixed. "
        "Use the result for exploration, not as a guaranteed business outcome."
    )

elif page == "Model Performance":
    st.subheader("Validation performance")
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        m1, m2, m3 = st.columns(3)
        m1.metric("RMSPE", f"{metrics.get('RMSPE', float('nan')):.2%}")
        m2.metric("MAE", f"{metrics.get('MAE', float('nan')):,.1f}")
        m3.metric("RMSE", f"{metrics.get('RMSE', float('nan')):,.1f}")
        st.caption(
            f"Validation: {metrics.get('validation_start', 'N/A')} to "
            f"{metrics.get('validation_end', 'N/A')} · "
            f"{metrics.get('forecast_type', 'Forecast protocol not recorded')}"
        )

        baseline_rmspe = metrics.get("Baseline_7DayLag_RMSPE")
        matched_model_rmspe = metrics.get("Model_On_BaselineRows_RMSPE")
        if baseline_rmspe is not None and matched_model_rmspe is not None:
            st.markdown("**Fair baseline comparison (same validation rows)**")
            baseline_cols = st.columns(3)
            baseline_cols[0].metric("Model RMSPE", f"{matched_model_rmspe:.2%}")
            baseline_cols[1].metric("7-day lag baseline RMSPE", f"{baseline_rmspe:.2%}")
            baseline_cols[2].metric(
                "Rows compared",
                f"{metrics.get('baseline_rows', 0):,}",
            )
            st.caption(
                "The matched comparison excludes validation rows without a seven-day "
                "history value. Overall model metrics above still use the full validation set."
            )
        st.json(metrics)
    else:
        st.info("Run the training script to generate validation metrics.")

    importance_path = MODEL_PATH.parent / "feature_importance.csv"
    if importance_path.exists():
        importance = pd.read_csv(importance_path).head(20).sort_values("importance")
        fig = px.bar(
            importance,
            x="importance",
            y="feature",
            orientation="h",
            title="Top 20 model feature importances",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Feature importance will appear after model training.")
