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

# Fetch data and forecast
if st.button("Get Forecast"):
    # API Call to FastAPI Backend

    response = requests.post("http://fastapi:8003/forecast_bikes_gam", json={"steps": forecast_months})
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
