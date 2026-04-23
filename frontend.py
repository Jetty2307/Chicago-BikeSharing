import streamlit as st
import requests
import pandas as pd
import altair as alt

st.set_page_config(layout="wide")

# Streamlit UI Layout
st.title("Chicago Bike Sharing Forecast Application")

col1, col2 = st.columns(2, gap="large")

# Initialize session state
if "selected_interval" not in st.session_state:
    st.session_state.selected_interval = "month"  # Default interval set to monthly

if "forecast_weeks" not in st.session_state:
    st.session_state.forecast = 12  # Default forecast duration (months)

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

if "selected_forecast_date" not in st.session_state:
    st.session_state.selected_forecast_date = None

if "slider_moved" not in st.session_state:
    st.session_state.slider_moved = False  # Tracks if the slider was moved

if "show_retro" not in st.session_state:
    st.session_state.show_retro = False  # Tracks if retro forecast is run

if "lookback" not in st.session_state:
    st.session_state.lookback = 3 # default lookback


# **Step 1: Select Interval (Daily / Weekly / Monthly)**
day_col, week_col, month_col = st.columns(3, gap="large")

with day_col:
    if st.button("Daily"):
        st.session_state.selected_interval = "day"
        st.session_state.slider_moved = False

with week_col:
    if st.button("Weekly"):
        st.session_state.selected_interval = "week"
        st.session_state.slider_moved = False  # Reset slider tracking

with month_col:
    if st.button("Monthly"):
        st.session_state.selected_interval = "month"
        st.session_state.slider_moved = False  # Reset slider tracking

# **Step 2: Adjust the Forecast Slider (Enforced)**
if st.session_state.selected_interval == "day":
    dates_response = requests.get("http://fastapi:8003/daily_forecast_dates")
    if dates_response.status_code != 200:
        st.error("Could not load available daily pseudo-forecast dates.")
    else:
        selectable_forecast_days = dates_response.json().get("dates", [])
        if not selectable_forecast_days:
            st.error("Not enough daily data to build a pseudo-forecast window.")
        else:
            default_forecast_date = selectable_forecast_days[-1]
            if st.session_state.selected_forecast_date not in selectable_forecast_days:
                st.session_state.selected_forecast_date = default_forecast_date
            st.session_state.selected_forecast_date = st.select_slider(
                "Select Forecast Day for Daily Pseudo-Forecast",
                options=selectable_forecast_days,
                value=st.session_state.selected_forecast_date,
            )
            st.session_state.forecast = 1

elif st.session_state.selected_interval == "week":
    new_value = st.slider(
        "Select Number of Weeks for Forecast",
        min_value=1, max_value=104, value=st.session_state.forecast
    )
    st.session_state.forecast = new_value

elif st.session_state.selected_interval == "month":
    new_value = st.slider(
        "Select Number of Months for Forecast",
        min_value=1, max_value=60, value=st.session_state.forecast
    )
    st.session_state.forecast = new_value

st.write(f"Selected interval: {st.session_state.selected_interval}")
if st.session_state.selected_interval == "day" and st.session_state.selected_forecast_date:
    st.write(f"Selected forecast day: {st.session_state.selected_forecast_date}")

# **Step 3: Select Model & Calculate (Only When Model Button is Pressed)**
def fetch_and_display_forecast(endpoint):
    """Fetch forecast data from FastAPI backend and display Altair chart."""
    payload = {
        "steps": st.session_state.forecast,
        "interval": st.session_state.selected_interval,
        "include_retro": st.session_state.show_retro,
        "lookback": int(st.session_state.lookback)
    }
    if st.session_state.selected_interval == "day":
        payload["forecast_date"] = st.session_state.selected_forecast_date

    response = requests.post(f"http://fastapi:8003/{endpoint}", json=payload)

    if response.status_code == 200:
        data = response.json()
        description = str(data["description"])

        if st.session_state.selected_interval == "day":
            forecast = pd.DataFrame(data["forecast"])
            actual = pd.DataFrame(data["actual"])
            metrics_col, description_col = st.columns([3, 2], gap="large")

            with metrics_col:
                st.write(f'Daily pseudo-forecast using {endpoint.removeprefix("forecast_bikes_").upper()}')
                forecast_value = int(forecast["rides"].iloc[0])
                actual_value = int(actual["rides"].iloc[0])
                target_day = forecast["year_day"].iloc[0]
                metric1, metric2 = st.columns(2)
                metric1.metric("Forecasted rides", f"{forecast_value:,}")
                metric2.metric("Actual rides", f"{actual_value:,}")
                st.caption(f"Target day: {target_day}")

            with description_col:
                if description:
                    st.markdown("Forecast analysis")
                    st.info(description)

            return

        # Convert historical and forecast data
        historical = pd.DataFrame(data["historical"])
        forecast = pd.DataFrame(data["forecast"])

        # Combine and mark data types
        historical["type"] = "Historical"
        forecast["type"] = "Forecast"
        combined_data = pd.concat([
            historical[[f"year_{st.session_state.selected_interval}", "rides", "type"]],
            forecast[[f"year_{st.session_state.selected_interval}", "rides", "type"]]
        ])

        # Create Altair chart
        st.write(f'Forecast of rides using {endpoint.removeprefix("forecast_bikes_").upper()}')
        line_chart = alt.Chart(combined_data).mark_line().encode(
            x=alt.X(f"year_{st.session_state.selected_interval}:T", title="Month", axis=alt.Axis(format="%Y-%m", labelFontSize=18, titleFontSize=20)),
            y=alt.Y("rides:Q", title="Number of rides", axis=alt.Axis(labelFontSize=18, titleFontSize=20)),
            color=alt.Color("type:N", legend=alt.Legend(title="Data Type", titleFontSize=20, labelFontSize=18),
                            scale=alt.Scale(domain=["Historical", "Forecast"], range=["blue", "orange"])),
            tooltip=[f"year_{st.session_state.selected_interval}:T", "rides:Q", "type:N"]
        ).properties(
            width=1000,
            height=500,
            title=alt.TitleParams("Bike Sharing in Chicago, Forecasts", fontSize=20)
        ).configure_legend(titleFontSize=16, labelFontSize=14).interactive()

        chart_col, description_col = st.columns([4, 2], gap="large")

        with chart_col:
            st.altair_chart(line_chart, use_container_width=True)

        with description_col:
            if description:
                st.markdown("Forecast analysis")
                st.info(description)

    else:
        st.error("Error fetching data. Please check the data.")

# **Step 4: Model Selection (Triggers Calculation Only When Button is Pressed)**
if st.session_state.selected_interval == "day":
    model_col1, model_col2 = st.columns(2, gap="small")

    with model_col1:
        if st.button("GAM"):
            st.session_state.selected_model = "forecast_bikes_gam"

    with model_col2:
        if st.button("XGBoost"):
            st.session_state.selected_model = "forecast_bikes_xgboost"
else:
    col1, col2, col3 = st.columns(3, gap="small")

    with col1:
        if st.button("SARIMA"):
            st.session_state.selected_model = "forecast_bikes_sarima"

    with col2:
        if st.button("GAM"):
            st.session_state.selected_model = "forecast_bikes_gam"

    with col3:
        if st.button("XGBoost"):
            st.session_state.selected_model = "forecast_bikes_xgboost"


# **Trigger Forecast Calculation**
if st.session_state.selected_model:
    fetch_and_display_forecast(st.session_state.selected_model)

# st.subheader("Retro forecast")
#
# if st.session_state.selected_interval == "week":
#     st.session_state.lookback = st.number_input(
#         "Number of weeks to look back for retro forecast",
#         min_value=1, max_value=52, value=int(st.session_state.lookback), step=1
#     )
# else:
#     st.session_state.lookback = st.number_input(
#         "Number of weeks to look back for retro forecast",
#         min_value=1, max_value=12, value=int(st.session_state.lookback), step=1
#     )

#st.divider()

# label = "Add retro forecast" if not st.session_state.show_retro else "Remove retro forecast"

# if st.button(label):
#    st.session_state.show_retro = not st.session_state.show_retro
