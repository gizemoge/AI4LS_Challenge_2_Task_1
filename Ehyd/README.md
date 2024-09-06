# AI4LS_Challenge

For each geographical location specified in the dataset, the teams are asked to forecast the monthly groundwater level from January 2022 to June 2024

## Overview

This project forecasts groundwater levels using 10 variables: **groundwater level, groundwater temperature, rainfall, snowfall, source flow rate, source temperature, source sediment levels, surface water levels, surface water temperature, and surface water flow rate.**

It processes and combines data from multiple CSV files containing time series data from different geographic stations. The goal is to train models on this data for specific geographic locations and predict the monthly groundwater levels from **January 2022 to June 2024.**

The project consists of the following steps:

1. Data Preprocessing
2. Data Imputation
3. Feature Engineering (Adding Lagged Values and Rolling Means)
4. Model Training
5. Forecasting Future Values
6. Installation



### Clone the repository:

```bash
git clone https://github.com/gizemoge/AI4LS_Challenge.git
cd AI4LS_Challenge
```


### Install the required dependencies:
This project requires Python 3.x and the following Python libraries:

- pandas
- numpy
- scikit-learn
- TensorFlow
- pickle

You can install the required dependencies by running:

```bash
pip install -r requirements.txt
```


### Prepare the dataset:
The CSV files with raw data should be placed in the appropriate folders under the Ehyd directory as follows:
- `Ehyd/datasets_ehyd/Groundwater`
- `Ehyd/datasets_ehyd/Precipitation`
- `Ehyd/datasets_ehyd/Sources`
- `Ehyd/datasets_ehyd/Surface_Water`

<br>

## Data Preprocessing

#### Step 1: Load Station Coordinates
Run the function `station_coordinates()` to load station coordinates for groundwater, precipitation, and other sources:

```bash
groundwater_all_coordinates = station_coordinates("Groundwater")
precipitation_coordinates = station_coordinates("Precipitation")
sources_coordinates = station_coordinates("Sources")
surface_water_coordinates = station_coordinates("Surface_Water")
```
#### Step 2: Process Time Series Data
For each dataset (e.g., Groundwater, Precipitation), process the data using process_and_store_data():

```bash
stations_list = [...]  # List of station IDs
filtered_groundwater_dict, filtered_gw_coordinates = process_and_store_data(
    os.path.join("Ehyd", "datasets_ehyd", "Groundwater", "Grundwasserstand-Monatsmittel"),
    groundwater_all_coordinates, "gw_", stations_list)
```

This function processes the raw CSV files into structured dataframes and stores them in a dictionary.

#### Step 3: Handle Missing Values
Missing values in the datasets are imputed using monthly means by calling the `nan_imputer()` function:

```bash
filled_filtered_groundwater_dict = nan_imputer(filtered_groundwater_dict)
```

#### Step 4: Feature Engineering
Add lagged values and rolling means to the datasets to capture temporal patterns:

```bash
filled_dict_list = [filled_gw_temp_dict, filled_filtered_groundwater_dict, filled_rain_dict, ...]
for dictionary in filled_dict_list:
    for key, df in dictionary.items():
        dictionary[key] = add_lag_and_rolling_mean(df)
```

#### Step 5: Save Processed Data
Save the processed and imputed data into pickle files for further use:

```bash
for dictionary in filled_dict_list:
    file_path = os.path.join("Ehyd", "pkl_files", f'final_{dict_name}.pkl')
    save_to_pickle(dictionary, file_path)
```
<br>

## Training the Model

To train the model on a specific geographic location, follow these steps:

**1. Select the relevant data for the station:**
Identify the nearest exogenous data points (e.g., rainfall, temperature) for the selected groundwater station using `find_nearest_coordinates()`.

**2. Prepare the input for the LSTM model:** Use the processed data with lagged values and rolling means as input features for the LSTM model.

**3. Train the model:** Load the prepared data and pass it into an LSTM model using TensorFlow:

```bash
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))
```
<br>

## Obtaining Forecasts

Once the model is trained, you can use it to forecast future groundwater levels:

```bash
forecasts = model.predict(X_test)
```

Save the forecasts to a file for later use:

```bash
np.savetxt('forecasts.csv', forecasts, delimiter=',')
```