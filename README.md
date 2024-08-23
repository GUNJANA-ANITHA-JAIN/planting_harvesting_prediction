# Rice Planting and Harvesting Prediction

This project focuses on predicting optimal planting and harvesting times for rice using historical weather data. The dataset includes daily records for the years 2020 and 2021, and various machine learning algorithms were applied to achieve the most accurate predictions. 

## Dataset

The dataset consists of daily weather parameters collected for the years 2020 and 2021 in the regions of Anantapur(Andhra Pradesh) . Each record includes the following features:

- `datetime`: Date of the record
- `tempmax`: Maximum temperature (°C)
- `tempmin`: Minimum temperature (°C)
- `temp`: Average temperature (°C)
- `feelslikemax`: Maximum feels-like temperature (°C)
- `feelslikemin`: Minimum feels-like temperature (°C)
- `feelslike`: Average feels-like temperature (°C)
- `dew`: Dew point (°C)
- `humidity`: Humidity (%)
- `precip`: Precipitation (mm)
- `precipprob`: Probability of precipitation (%)
- `precipcover`: Coverage of precipitation (%)
- `preciptype`: Type of precipitation (e.g., rain, snow)
- `snow`: Snowfall (mm)
- `snowdepth`: Snow depth (mm)
- `windgust`: Maximum wind gust speed (km/h)
- `windspeed`: Average wind speed (km/h)
- `winddir`: Wind direction (°)
- `sealevelpressure`: Sea level pressure (hPa)
- `cloudcover`: Cloud cover (%)
- `visibility`: Visibility (km)
- `solarradiation`: Solar radiation (W/m²)
- `solarenergy`: Solar energy (kWh/m²)
- `uvindex`: UV index
- `severerisk`: Severe weather risk
- `sunrise`: Sunrise time
- `sunset`: Sunset time
- `moonphase`: Moon phase
- `conditions`: General weather conditions
- `description`: Detailed weather description
- `icon`: Weather icon
- `stations`: Weather stations

## Methodology

### Data Preparation

1. Combined datasets from 2020 and 2021.
2. Converted 'datetime' column to datetime format.
3. Forward filled missing values.
4. Calculated mean temperature.
5. Converted sunrise and sunset times to datetime format and calculated daylight hours.
6. Defined conditions for planting and harvesting based on ideal weather parameters.

### Model Training and Evaluation

Nine different machine learning algorithms were tested to determine the most accurate model for predicting planting and harvesting times. The XGBoost algorithm was selected based on its performance.

### Results
#### Planting Time Prediction

- **Model Used**: XGBoost
- **Evaluation Metrics**: Accuracy, Classification Report
  ![planting](https://github.com/user-attachments/assets/affaa8d1-b18a-4d0f-ab8b-741d62cd8f00)

#### Harvesting Time Prediction

- **Model Used**: XGBoost
- **Evaluation Metrics**: Accuracy, Classification Report
  ![metric](https://github.com/user-attachments/assets/dd07a29d-4aeb-4d32-bf60-a9ee93017ba7)

### Visualizations

Graphs and plots showing the following:
- Temperature trends over time
  ![temp](https://github.com/user-attachments/assets/d15780ec-c93a-4058-8d62-5c7d95de0581)

- Humidity trends over time
  ![humidity](https://github.com/user-attachments/assets/a0772ca3-b884-497e-87a2-b84df2d709ce)

- Predicted planting months for 2021
  ![predicted](https://github.com/user-attachments/assets/8b10f31b-32d6-40a1-a1e2-17a3ca33f673)

- Predicted harvesting months for 2021
  ![harvested](https://github.com/user-attachments/assets/16771b7f-1c1f-420e-a5e5-39f8ecede628)


### Code

The Python code provided includes:

1. Data loading and preprocessing
2. Model training and evaluation
3. Visualization of results

### How to Use

1. Ensure you have the required libraries installed: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `xgboost`.
2. Load the datasets and combine them as shown in the code.
3. Execute the code to train the models and generate predictions and visualizations.

## License

This project is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/). 

You are free to:
- **Share** — copy and redistribute the material in any medium or format
- **Adapt** — remix, transform, and build upon the material

Under the following terms:
- **Attribution** — You must give appropriate credit, provide a link to the license, and indicate if changes were made.
