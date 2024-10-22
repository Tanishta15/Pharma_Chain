import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
from datetime import datetime
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
import os

# Path to your CSV file
results_path = '/Users/tanishta/Desktop/GitHub/CPI/Dataset/salesdaily.csv'

# Load the CSV file
data = pd.read_csv(results_path)

# Convert the 'date' column to datetime format and set it as the index
data['date'] = pd.to_datetime(data['date'], format='mixed', dayfirst=False)
data.set_index('date', inplace=True)

# List of product categories to forecast
product_categories = ['M01AB', 'M01AE', 'N02BA', 'N02BE', 'N05B', 'N05C', 'R03', 'R06']

# Dictionary to hold results
forecast_results = {}
mse_results = {}

# Define parameters for EOQ
annual_demand_estimate = 365
holding_cost_per_unit_per_year = 2
reorder_point = 50
safety_stock = 20
order_cost = 100
supplier_lead_time = 10

# Function to calculate distance between two cities
def get_distance(city1, city2):
    # Example coordinates for cities (latitude, longitude)
    coords = {
        'AstraZeneca': (13.126673959848691, 77.61428680623204),
        'R L Fine Chem Pvt. Ltd.': (13.108225922813784, 77.58094729480572),
        'Alnes Pharma India Pvt Ltd': (13.102719090767396, 77.579687830924232),
        'Astrazeneca Research Foundation India': (13.067815244607441, 77.62255325143731),
        'DFE Pharma India Private Limited': (12.985390114416418, 77.7219294455769),
        'Novartis India Limited': (12.976184878196241, 77.60543296999586),
        'Abbvie India Private Limited': (12.981809528609254, 77.59755134001882),
        'Bal Pharma Limited': (12.994772922241175, 77.59025573144308),
        'Pharma Corporation of India': (13.001128919453665, 77.54528045249025),
        'Group Pharma': (13.01304597560203, 77.55600928888111),
        'Sun Pharmaceutical Industries Limited': (13.03859254062895, 77.5901438352728),
        'Med-India': (13.036585704621686, 77.59117380349309),
        'Daphne Pharmaceuticals Private Limited': (13.030063375029227, 77.5604295701835),
        'Senses pharmaceuticals private Limited': (13.031202704580592, 77.53034591385175),
        'Peenya Pharmaceuticals Pvt Ltd': (13.023331816332165, 77.51754101958883),
        'Celest Pharma Labs Private Limited': (13.02363054417523, 77.51788302242679),
        'KARNATAKA ANTIBIOTICS AND PHARMACEUTICALS LTD': (13.026702844939722, 77.51381367326027),
        'Madhur Pharma & Research Laboratories Private Limited': (13.023942002043398, 77.51354590021863),
        'BPREX pharma packaging india pvt ltd': (13.02311230584062, 77.50428271296745),
        'Sami-Sabinsa Group Limited': (13.035488208965813, 77.51080584502931),
        'Sundia MediPharma India Private Limited': (13.035488208965813,77.51080584502931),

        'Sri Ganesh medical & general storechemist': (13.123350789119192, 77.61076861589498),
    }
    
    if city1 in coords and city2 in coords:
        distance = geodesic(coords[city1], coords[city2]).kilometers
        print(f"Distance from {city1} to {city2}: {distance:.2f} km")
    else:
        print("Coordinates for the entered cities are not available.")

# Get user input for city names
print("Enter pharma factory from where you want to get medicines delivered:")
place_of_production = input("Enter the city of production: ")
pharma_store = input("Enter the city of plant: ")

# Get distance between the cities
get_distance(place_of_production, pharma_store)

# Iterate over each product category
for category in product_categories:
    # Filter data for the category (quantity sold per month)
    product_data = data[category]

    # EOQ calculation
    eoq = np.sqrt((2 * annual_demand_estimate * order_cost) / holding_cost_per_unit_per_year)
    print(f'EOQ for Product {category}: {eoq}')
    print(f'Safety Stock for Product {category}: {safety_stock}')

    # Split the data into training and testing sets
    train_size = int(len(product_data) * 0.8)
    train, test = product_data[:train_size], product_data[train_size:]

    # Fit the SARIMAX model for forecasting
    model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
    model_fit = model.fit(disp=False)

    # Forecasting
    forecast = model_fit.forecast(steps=len(test))
    forecast_results[category] = forecast

    # Evaluate the forecast with MSE
    if test.isna().sum() == 0 and forecast.isna().sum() == 0:
        mse = mean_absolute_error(test, forecast)
        mse_results[category] = mse
        print(f'Product Category: {category}, MSE: {mse}')

    # Plotting the forecast against actual values
    plt.figure(figsize=(10, 4))
    plt.plot(test.index, test, label='Actual')
    plt.plot(test.index, forecast, label='Forecast')
    plt.title(f'Forecast vs Actual for {category}')
    plt.xlabel('Date')
    plt.ylabel('Quantity Sold')
    plt.legend()
    plt.show()

# Save the forecast results to CSV
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
forecast_results_df = pd.DataFrame.from_dict(forecast_results, orient='index').transpose()
forecast_results_df.to_csv(f'/Users/tanishta/Desktop/GitHub/CPI/forecast_results_{timestamp}.csv', index=False)

# Save the MSE results to CSV
mse_results_df = pd.DataFrame(list(mse_results.items()), columns=['Product_Category', 'MSE'])
mse_results_df.to_csv(f'/Users/tanishta/Desktop/GitHub/CPI/mse_results_{timestamp}.csv', index=False)

print("Forecasting and MSE results saved successfully.")