# frontend.py
import streamlit as st
import requests
import pandas as pd
import altair as alt
import time
from fastapi import FastAPI

# Streamlit UI Layout
st.title("Chicago Bike Sharing Forecast Application")

# Input section

forecast_months = st.slider("Select Months for Forecast", min_value=1, max_value=60, value=12)

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
# Fetch data and forecast
def fetch_and_display_forecast(endpoint):

    response = requests.post(f"http://fastapi:8003/{endpoint}", json={"steps": forecast_months})
    if response.status_code == 200:
        data = response.json()

        # Historical data
        historical = pd.DataFrame(data["historical"])
        forecast = pd.DataFrame(data["forecast"])

        # Combine historical and forecast data
        historical['type'] = 'Historical'
        forecast['type'] = 'Forecast'
        combined_data = pd.concat([historical[['year_month','rides','type']],
        forecast[['year_month','rides','type']]])

        # Create the Altair chart with larger font sizes
        st.write("Forecast of monthly rides")
        line_chart = alt.Chart(combined_data).mark_line().encode(
            x=alt.X('year_month:T', title='Month', axis=alt.Axis(format='%Y-%m',
            labelFontSize=18, titleFontSize=20)),
            y=alt.Y('rides:Q', title='Number of rides',
            axis=alt.Axis(labelFontSize=18, titleFontSize=20)),
            color=alt.Color('type:N', legend=alt.Legend(title="Data Type",titleFontSize=20,
            labelFontSize=18), scale=alt.Scale(domain=['Historical','Forecast'],
            range=['blue','orange'])),
            tooltip=['year_month:T',"rides:Q",'type:N']
        ).properties(
            width=700,
            height=400,
            title=alt.TitleParams("Bike Sharing in Chicago, monthly forecats", fontSize=20)
        ).configure_legend(
            titleFontSize=16,
            labelFontSize=14
        ).interactive() # Add zoom and pan functionality
        st.altair_chart(line_chart, use_container_width=True)

    else:
        st.error("Error fetching data. Please check the data.")

# button_container = st.container()


col1, col2, col3 = st.columns(3, gap="small")

#st.markdown('<div class="button-container">', unsafe_allow_html=True)
#col1, col2 = st.columns(2)

with col1:
    if st.button("SARIMA"):
        st.session_state.selected_model = ("forecast_bikes_sarima")

with col2:
    if st.button("GAM"):
        st.session_state.selected_model = ("forecast_bikes_gam")

with col3:
    if st.button("XGBoost"):
        st.session_state.selected_model = ("forecast_bikes_xgboost")

if st.session_state.selected_model:
    endpoint = st.session_state.selected_model
    fetch_and_display_forecast(endpoint)