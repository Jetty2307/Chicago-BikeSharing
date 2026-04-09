# Chicago bikes usage prediction


Description:
--------------
Within a previous decade, bike sharing became a popular way of inner-city mobility. In many cities, it has not only a leisure purpose, but acts as a full-blown public transportation mode carrying people to and from work and for their other daily activities. Development of mobile apps helped to significantly improve the bike sharing systems making the process of tracking, locking/unlocking of the bicycle and payment easy and straightforward. The growing introduction of electric bikes and e-scooters made the systems more acceptable for the broader range of customers and also facilitated the expansion of such systems to districts and cities with hilly landscapes. 

One of the examples of such systems is the bike sharing system in Chicago operated by [Divvy](https://divvybikes.com). This is the third most-populous city of the United States, which has typical big city problems of traffic jams, overloaded public transport or lack of good transport connection in some areas. Therefore Divvy offers additional way of mobility, and the system became very popular: it has around 1 000 station in Chicago and vicinities, over 15 000 vehicles, and its daily ridership amounts to over 6 million per year. Divvy also provides an app for its customers, as well as detailed data on each particular trip which took place through the service.

Goal:
------
Unlike with traditional modes of transport, the demand in bike sharing can be very flexible depending on such factors as weather, season, date, day of the week, time of the day, presence of accompanying person or persons etc. Therefore, for the operator it is important to know the potential demand at each station at a particular time instance. This knowledge may significantly improve the overall performance of the system, for example, by redistributing free bicycles among stations, charging a proper amount of electric bikes, setting reminders in the app and tuning the price of the service. The time series predictive model then must evaluate the bike usage at certain dates and at certain points with a good accuracy.

Data: 
---------------------
The Divvy data is taken from https://divvy-tripdata.s3.amazonaws.com/index.html.

Here is my [dashboard](https://app.hex.tech/019c47a9-f26e-7002-933e-b7c1d15876fa/app/Chicago-Bikes-Sharing-Stats-2020---2025-032YKwabAGzllw1oj3movx/latest) with some aggregated rides data.

Technical solution: 
---------------------
The project is close to production-ready solution and based on modern industrial standards and data pipelines.

From the analytical point of view, three different models are used for time series predictions: SARIMA, gradient boosting (XGBoost) and generalized additive model (PyGAM).

Project structure and data pipeline:

1. Data extraction, loading and transformation (ELT)
   This step is orchestrated in Airflow and run on a monthly basis, as the data from Divvy service is updated every month. It    goes as follows:

     - new data uploaded to the local host from the web with REST API requests
     - the new data from directories is inserted to the local PostgreSQL database
     - aggregated tables are formed for weekly and monthly bike usage depending on their type (electric or classic) with SQL queries in dbt
     - additionally the historical weather data are downloaded from [open-meteo](https://open-meteo.com) to form weather features (temperature, rain, snow), and they are also stored in the DB,
     and joined with time features (year, season, week/month/ lags) for the rides with dbt.
    
2. Tests for ELT procedures and resulting output
 
3. Features engineering and training the models with their evaluation, registration and feature importance control
   - extracting features from the dataframes and their transformation if needed
   - training and validation of SARIMA, XGBoost and PyGAM models for weekly and monthly predictions with a new data point(s) 
   - registration of the models with MLflow if on validation their performance does not deteriorate
   - evaluating the feature importance with SHAP values (XGBoost) and partial dependence (GAM) and saving as model artefacts in MLflow registry
  
4. AI description of the current model performance with Llama 3 (or Deepseek if the description of Llama 3 is unsatisfactory) and giving an ability to AI to prevent a new trained model from registry if it evaluates it as bad.
     
5. App embedding:
   - backend (FastAPI + Uvicorn) for taking the input for models inference and fetching the models output + SARIMA run if called
   - frontend (Streamlit) for the user's selection of the model and predictions parameters and visualization of the result
     
     The user can select a certain model, and the predictions for upcoming weeks or months for a certain period of time will be shown.
     
6. App containerization with Docker
