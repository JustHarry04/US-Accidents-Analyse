# US-Accidents Analysis Dashboard

## Description

The **US-Accidents Analysis Dashboard** is a Streamlit application designed to provide interactive visualizations and insights into traffic accident data across the United States. This application helps users explore and analyze accident trends, severity, and contributing factors using various charts and maps.

## Features

- **Dataset Preview**: View the first few rows of the dataset and understand its structure.
- **Accident Maps**: Interactive maps showing accident locations, total accidents by state, and severity levels.
- **Time-Based Analysis**: Charts displaying accident counts over time, including by year, month, day of the week, and hour of the day.
- **City-Specific Insights**: Visualizations of monthly accident data for the top 10 cities and low severity accidents in cities with low visibility.
- **Temperature and Road Length Analysis**: Histograms of temperature distribution and affected road lengths.

## Installation

To run this application locally, you need to have Python and Streamlit installed. You can install the required packages using pip:

```bash
pip install streamlit pandas seaborn matplotlib
## Files

- **`app.py`**: Main Streamlit application file.
- **`data/`**: Directory containing the dataset and any additional data files.
- **`assets/`**: Directory for images or static files used in the application.

## Dataset

The application uses a dataset containing US traffic accidents from 2016 to 2022. Make sure to place the dataset in the `data/` directory or modify the file path in `app.py` accordingly.

## Contributing

Contributions are welcome! If you have suggestions for improvements or encounter any issues, please open an issue or submit a pull request.


## Acknowledgments

- [Streamlit](https://streamlit.io) for creating an easy-to-use framework for building interactive web applications.
- [Seaborn](https://seaborn.pydata.org) for beautiful statistical visualizations.
