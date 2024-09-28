# AI4LS_Challenge

## Introduction
This project involves processing multiple CSV files containing time series data from thousands of measurement water stations in Austria ![Austria](https://raw.githubusercontent.com/stevenrskelton/flag-icon/master/png/16/country-4x3/at.png "Austria"). The data is cleaned, filtered, and merged by geographic coordinates to generate forecasts for a specific geographic location using machine learning models.

The goal is to train models on this data for specific geographic locations and predict the monthly groundwater levels from **January 2022 to June 2024.**

Groundwater levels are forecasted using 11 variables under 4 headings:

**Groundwater:**
<br>
1. Groundwater level,
2. Groundwater temperature,

**Precipitation:**
<br>
3. Rainfall,
4. Snowfall,

**Water source:**
5. Source flow rate,
6. Source temperature,
7. Source sediment levels,
8. Source conductivity,

**Surface water:**
9. Surface water levels,
10. Surface water temperature, and
11. Surface water flow rate.

<br>

## Prerequisites

This project requires Python 3.8+ and the following Python libraries:

  - pandas 
  - numpy 
  - datetime 
  - scipy 
  - statsmodels
  - os

You can install the required dependencies by running:

```bash
pip install -r requirements.txt
```

<br>

## Methodology

### 1. Data Preprocessing
The four datasets from `ehyd.gv.at` related to groundwater, precipitation, water sources and surface waters in Austria are downloaded into the provided data structure. 
The folder names are renamed to their English equivalents, but no changes are made to the files themselves. 
The measurement data is extracted from the CSV files, and a dictionary is created for each variable. 
In these dictionaries, the keys represent the date (_year-month_), and the dataframes contain the measurement station IDs as their indices. 
The coordinates are obtained from the `messstellen_alle` files.

<br>

#### Processing Data for a Specific Location
487 groundwater level measurement stations are forecasted in this document, 
though the model can forecast the groundwater levels for any and all of the 3792 groundwater measurement stations (as specified in the `messstellen_alle.csv` file located in the Groundwater folder). 

To train the model for a specific geographic location, first specify the station IDs for that location in the `gw_test_empty.csv` file. 
The stations are filtered and processed using the function:

```bash
filtered_groundwater_dict, filtered_gw_coordinates = process_and_store_data(
    "datasets_ehyd/Groundwater/Grundwasserstand-Monatsmittel",
    groundwater_all_coordinates, "gw_", station_list)
```

<br>

####  Merging and Aligning Data Based on Coordinates
The counts of measurement stations per type (e.g., precipitation or surface water) are not consistent. 
The `add_nearest_coordinates_column()` function associates each measurement station with its nearest counterparts based on Euclidean distances.
For some variables, this document selects three stations for triangulation, while for others, just one is selected, depending on the number of available measurement stations. 

In this example, the 864 rain measurement stations are enough in number to be triangulated for each of the 3793 groundwater level measurement stations. Thus the three closest rain measurement station IDs are selected:


```bash
data = add_nearest_coordinates_column(rain_coord, 'nearest_rain', 3)
```

In comparison, there are only 94 water source flowrate measurement stations, and so only one nearest water source flowrate measurement station ID is taken. 


<br>

### 2. Data Imputation
Missing data is handled by the `nan_imputer()` function, which fills the NaN values using monthly averages:

```bash
filled_filtered_groundwater_dict = nan_imputer(filtered_groundwater_dict)
```

<br>

### 3. Feature Engineering
To prepare data for machine learning models, lags and rolling means are added to the datasets using the function `add_lag_and_rolling_mean()`. 
Based on our literature review (especially _Sutanto et al., 2024_), this function is defined to compute lagged values (1, 2, 3 months) and a rolling mean (over 6 months) for the dataframes.
Each dataframe in the filled dictionaries has lagged and rolling mean features added:

```bash
df = add_lag_and_rolling_mean(filled_dict['groundwater'])
```



Two dictionaries are then created:
- `new_dataframes` is a dictionary storing DataFrames of each measurement station, containing all original and engineered features monthly from January 2022 to June 2024. The keys are the measurement station IDs, and the values are dataframes.
- `monthly_dataframes` contains dataframes that are monthly snapshots of all measurement stations combined, created by processing the data from `new_dataframes`. In this dictionary, the keys represent months
                        instead of measurement station IDs.

Data is compiled monthly for the years between 1985 and 2021 in the `monthly_dict_85to21` dictionary, allowing for a consolidated view of the data. 
The data is filtered to start from 1985 instead of 1960 to reduce the amount of imputed synthetic data, which is concentrated in the earlier years.


Lastly, an average correlation feature selection method is applied to identify features that correlate significantly with the target variable, groundwater level. 
Features with an absolute correlation above a definable threshold (10% by default, 40% in this document) are selected for further analysis.


<br>

### 4. Model Training
The SARIMAX (Seasonal Autoregressive Integrated Moving Average with eXogenous variables) model is used for forecasting as it accounts for both seasonal and non-seasonal factors in the time series data.

From the `monthly_dict_85to21` dictionary, 5 years' worth of the data from `'2015_01'` to `'2019_12'` are taken as the train set, and 24 months from `'2020_01'` to `'2021_12'` as the validation set. 
Other start and cut-off dates can also be set.

After hyperparameter optimization, a **SMAPE score of 0.15** is obtained for the validation set, and this optimized model is selected for the final forecasting.

```bash
model = SARIMAX(
    station_data['Values'],
    exog=station_data.drop(columns=['Values']),
    order=(1, 0, 1),
    seasonal_order=(0, 1, 1, 12)
)
model_fit = model.fit(disp=False)
```



### 5. Forecasting Future Values
The model forecasts the next 30 time steps (months) for the given groundwater level measurement stations using the fitted model. 
The exogenous variables from the last 30 observations are used to improve forecast accuracy.

The forecasted values are compiled into a DataFrame and saved to a CSV file named `forecast_final.csv`.

<br>



## References
Sutanto, S. J., Syaehuddin, W. A., & de Graaf, I. (2024). Hydrological drought forecasts using precipitation data depend on catchment properties and human activities. Communications Earth & Environment, 5, Article 118. https://doi.org/10.1038/s43247-024-01295-w
