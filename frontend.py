import streamlit as st
import requests
import pandas as pd
import altair as alt

st.set_page_config(layout="wide")

# Streamlit UI Layout
st.title("Chicago Bike Sharing Forecast Application")

# Initialize session state
if "selected_interval" not in st.session_state:
    st.session_state.selected_interval = "month"  # Default interval set to monthly

if "forecast" not in st.session_state:
    st.session_state.forecast = 12  # Default forecast duration (months)

if "selected_model" not in st.session_state or st.session_state.selected_model is None:
    st.session_state.selected_model = "forecast_bikes_gam"

if "selected_forecast_date" not in st.session_state:
    st.session_state.selected_forecast_date = None

if "slider_moved" not in st.session_state:
    st.session_state.slider_moved = False  # Tracks if the slider was moved

if "show_retro" not in st.session_state:
    st.session_state.show_retro = False  # Tracks if retro forecast is run

if "lookback" not in st.session_state:
    st.session_state.lookback = 3 # default lookback


def select_interval(interval):
    if interval == "day" and st.session_state.selected_model == "forecast_bikes_sarima":
        return
    st.session_state.selected_interval = interval
    st.session_state.slider_moved = False
    maximum = {"day": 1, "week": 104, "month": 60}[interval]
    st.session_state.forecast = min(max(1, st.session_state.forecast), maximum)


def select_model(endpoint):
    if endpoint == "forecast_bikes_sarima" and st.session_state.selected_interval == "day":
        return
    st.session_state.selected_model = endpoint


# Recover an incompatible selection left in an existing browser session.
if (st.session_state.selected_interval == "day"
        and st.session_state.selected_model == "forecast_bikes_sarima"):
    st.session_state.selected_model = "forecast_bikes_gam"


# Callbacks update selection before rendering, so highlights change on one click.
for column, (label, interval) in zip(
    st.columns(3, gap="large"),
    [("Daily", "day"), ("Weekly", "week"), ("Monthly", "month")],
):
    with column:
        st.button(
            label,
            key=f"interval_{interval}",
            type="primary" if st.session_state.selected_interval == interval else "secondary",
            disabled=interval == "day" and st.session_state.selected_model == "forecast_bikes_sarima",
            on_click=select_interval,
            args=(interval,),
        )

# **Step 2: Adjust the Forecast Slider (Enforced)**
if st.session_state.selected_interval == "day":
    try:
        dates_response = requests.get("http://fastapi:8003/daily_forecast_dates", timeout=30)
    except requests.RequestException:
        dates_response = None
    if dates_response is None or dates_response.status_code != 200:
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

    try:
        response = requests.post(f"http://fastapi:8003/{endpoint}", json=payload, timeout=30)
    except requests.RequestException:
        st.error("Forecast service is unavailable. Please try again shortly.")
        return

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

# Keep all models visible; unsupported combinations are disabled.
for column, (label, endpoint) in zip(
    st.columns(3, gap="small"),
    [("SARIMA", "forecast_bikes_sarima"), ("GAM", "forecast_bikes_gam"),
     ("XGBoost", "forecast_bikes_xgboost")],
):
    with column:
        st.button(
            label,
            key=endpoint,
            type="primary" if st.session_state.selected_model == endpoint else "secondary",
            disabled=endpoint == "forecast_bikes_sarima" and st.session_state.selected_interval == "day",
            on_click=select_model,
            args=(endpoint,),
        )


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
