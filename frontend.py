import streamlit as st
import requests
import pandas as pd
import altair as alt

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

if "slider_moved" not in st.session_state:
    st.session_state.slider_moved = False  # Tracks if the slider was moved

# **Step 1: Select Interval (Weekly / Monthly)**
with col1:
    if st.button("Weekly"):
        st.session_state.selected_interval = "week"
        st.session_state.slider_moved = False  # Reset slider tracking

with col2:
    if st.button("Monthly"):
        st.session_state.selected_interval = "month"
        st.session_state.slider_moved = False  # Reset slider tracking

# **Step 2: Adjust the Forecast Slider (Enforced)**
if st.session_state.selected_interval == "week":
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

# **Step 3: Select Model & Calculate (Only When Model Button is Pressed)**
def fetch_and_display_forecast(endpoint):
    """Fetch forecast data from FastAPI backend and display Altair chart."""
    response = requests.post(f"http://fastapi:8003/{endpoint}", json={
        "steps": st.session_state.forecast,
        "interval": st.session_state.selected_interval
    })

    if response.status_code == 200:
        data = response.json()

        # Convert historical and forecast data
        historical = pd.DataFrame(data["historical"])
        forecast = pd.DataFrame(data["forecast"])

        # Combine and mark data types
        historical["type"] = "Historical"
        forecast["type"] = "Forecast"
        combined_data = pd.concat([
            historical[[st.session_state.selected_interval, "rides", "type"]],
            forecast[[st.session_state.selected_interval, "rides", "type"]]
        ])

        # Create Altair chart
        st.write(f'Forecast of rides using {endpoint.removeprefix("forecast_bikes_").upper()}')
        line_chart = alt.Chart(combined_data).mark_line().encode(
            x=alt.X(f"{st.session_state.selected_interval}:T", title="Month", axis=alt.Axis(format="%Y-%M", labelFontSize=18, titleFontSize=20)),
            y=alt.Y("rides:Q", title="Number of rides", axis=alt.Axis(labelFontSize=18, titleFontSize=20)),
            color=alt.Color("type:N", legend=alt.Legend(title="Data Type", titleFontSize=20, labelFontSize=18),
                            scale=alt.Scale(domain=["Historical", "Forecast"], range=["blue", "orange"])),
            tooltip=[f"{st.session_state.selected_interval}:T", "rides:Q", "type:N"]
        ).properties(
            width=700,
            height=400,
            title=alt.TitleParams("Bike Sharing in Chicago, Forecasts", fontSize=20)
        ).configure_legend(titleFontSize=16, labelFontSize=14).interactive()

        st.altair_chart(line_chart, use_container_width=True)

    else:
        st.error("Error fetching data. Please check the data.")

# **Step 4: Model Selection (Triggers Calculation Only When Button is Pressed)**
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


