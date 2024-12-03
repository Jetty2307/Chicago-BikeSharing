# Chicago bikes analysis projects


Description:
--------------
Within a previous decade bike sharing became a popular way of inner-city mobility. In many cities, it has not only a leisure purpose, but acts a full-blown public transportation mode carrying people to and from work and for their other daily activities. Development of mobile apps helped to significantly improve the bike sharing systems making the process of tracking, locking/unlocking of the bicycle and payment easy and straightforward. The growing introduction of electric bikes and e-scooters made the systems more acceptable for the broader range of customers and also facilitated the expansion of such systems to districts and cities with hilly landscapes. 

One of the examples of such systems is the bike sharing system in Chicago operated by Divvy. This is the third most-populous city of the United States, which has typical big city problems of traffic jams, overloaded public transport or lack of good transport connection in some areas. Therefore Divvy offers additional way of mobility, and the system became very popular: it has around 1 000 station in Chicago and vicinities, over 15 000 vehicles, and its daily ridership amounts to over 6 million per year. Divvy also provides an app for its customers, as well as detailed data on each particular trip which took place through the service.

Goal:
------
Unlike with traditional modes of transport, the demand in bike sharing can be very flexible depending on such factors as weather, season, date, day of the week, time of the day, presence of accompanying person or persons etc. Therefore, for the operator it is important to know the potential demand at each station at a particular time instance. This knowledge may significantly improve the overall performance of the system, for example, by redistributing free bicycles among stations, charging a proper amount of electric bikes, setting reminders in the app and tuning the price of the service. The time series predictive model then must evaluate the bike usage at certain dates and at certain points with a good accuracy.

Visualization of the important data from the operator may show important trends in the system performance, highlight the problems and bottlenecks, identify the most and least popular areas. It is also useful in profiling of the most typical rides in terms of time, location, duration and length and the typical users of the systems. This information would alleviate further decision on strategic development of the system, as well as collateral activities, such as marketing, special offers and campaigns etc.

Technical solution: 
---------------------
The project consists of two parts: visualisation of most characteristic data related to usage of the bike sharing system in Chicago and construction of the predictive model for future expected usage of the system at particular dates. For both parts, the data from Divvy Tripdata are used. Information on data is available upon API requests as processed historical data (ride id, start and end station, start and end time, start and end coordinates, membership type), or as live data. 

Visualization: historical and live data for Chicago bike sharing system with various types of plots, including visualization on a geographical map. Tools: Jupyter, Tableau dashboards.
Predictive model: time series model (ARIMA), gradient boosting (xgboost) in Python, Tableau dashboards to evaluate the quality of the models by comparing predicted and actual data.
