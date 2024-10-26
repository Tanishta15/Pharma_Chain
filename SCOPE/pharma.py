import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
from datetime import datetime
import http.client
import json
import os

# Directory to save API results
output_dir = '/Users/tanishta/Desktop/GitHub/SCOPE/Directions'
os.makedirs(output_dir, exist_ok=True)  # Create directory if it doesn't exist

# Store results in lists for each API
directions_data = []  # Stores each route's information as a dictionary
places_data = []      # Stores each place's information as a dictionary
geocode_data = []     # Stores geocode result as a dictionary
matrix_data = []      # Stores driving matrix information

# Function to save API results to CSV
def save_to_csv(data, filename):
    if isinstance(data, list) and len(data) > 0:
        df = pd.DataFrame(data)
        file_path = os.path.join(output_dir, filename)
        df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}")
    else:
        print(f"No data to save for {filename}")

# Function for TrueWay Directions API to calculate route distance
def get_route_distance(stops):
    conn = http.client.HTTPSConnection("trueway-directions2.p.rapidapi.com")
    headers = {
        'x-rapidapi-key': "5b18cb5cb1msh75d91bed144ff14p17e377jsnd7ae035bedb5",
        'x-rapidapi-host': "trueway-directions2.p.rapidapi.com"
    }
    try:
        conn.request("GET", f"/FindDrivingRoute?stops={stops}", headers=headers)
        res = conn.getresponse()
        data = res.read()
        route_info = json.loads(data.decode("utf-8"))

        if route_info.get("route"):
            distance_meters = route_info["route"]["distance"]
            distance_km = distance_meters / 1000  # Convert to kilometers
            directions_data.append({"Stops": stops, "Distance (km)": distance_km})
            save_to_csv(directions_data, 'directions_results.csv')
            return distance_km
        else:
            print("No route found.")
            return None
    except Exception as e:
        print(f"Error retrieving distance: {e}")
        return None

# Function for TrueWay Places API to find nearby places
def get_nearby_places(location, place_type="cafe", radius=180):
    conn = http.client.HTTPSConnection("trueway-places.p.rapidapi.com")
    headers = {
        'x-rapidapi-key': "5b18cb5cb1msh75d91bed144ff14p17e377jsnd7ae035bedb5",
        'x-rapidapi-host': "trueway-places.p.rapidapi.com"
    }
    try:
        conn.request("GET", f"/FindPlacesNearby?location={location}&type={place_type}&radius={radius}&language=en", headers=headers)
        res = conn.getresponse()
        data = res.read()
        places_info = json.loads(data.decode("utf-8"))
        if isinstance(places_info, dict) and "results" in places_info:
            places_data.extend(places_info["results"])
            save_to_csv(places_data, 'places_results.csv')
        else:
            print("No places found.")
    except Exception as e:
        print(f"Error retrieving places: {e}")

# Function for TrueWay Geocoding API for reverse geocoding
def reverse_geocode(location):
    conn = http.client.HTTPSConnection("trueway-geocoding.p.rapidapi.com")
    headers = {
        'x-rapidapi-key': "5b18cb5cb1msh75d91bed144ff14p17e377jsnd7ae035bedb5",
        'x-rapidapi-host': "trueway-geocoding.p.rapidapi.com"
    }
    try:
        conn.request("GET", f"/ReverseGeocode?location={location}&language=en", headers=headers)
        res = conn.getresponse()
        data = res.read()
        geocode_info = json.loads(data.decode("utf-8"))
        if isinstance(geocode_info, dict):
            geocode_data.append(geocode_info)
            save_to_csv(geocode_data, 'geocode_results.csv')
        else:
            print("No geocode information found.")
    except Exception as e:
        print(f"Error retrieving geocode information: {e}")

# Function for TrueWay Matrix API for driving distance matrix
def calculate_driving_matrix(origins, destinations):
    conn = http.client.HTTPSConnection("trueway-matrix.p.rapidapi.com")
    headers = {
        'x-rapidapi-key': "5b18cb5cb1msh75d91bed144ff14p17e377jsnd7ae035bedb5",
        'x-rapidapi-host': "trueway-matrix.p.rapidapi.com"
    }
    try:
        conn.request("GET", f"/CalculateDrivingMatrix?origins={origins}&destinations={destinations}", headers=headers)
        res = conn.getresponse()
        data = res.read()
        matrix_info = json.loads(data.decode("utf-8"))
        if isinstance(matrix_info, dict) and "rows" in matrix_info:
            matrix_data.extend(matrix_info["rows"])
            save_to_csv(matrix_data, 'matrix_results.csv')
        else:
            print("No driving matrix information found.")
    except Exception as e:
        print(f"Error retrieving driving matrix: {e}")

# Example usages for the APIs
stops = "40.629041,-74.025606;40.630099,-73.993521;40.644895,-74.013818;40.627177,-73.980853"
distance_km = get_route_distance(stops)
if distance_km:
    print(f"Distance for route: {distance_km} km")

get_nearby_places("37.783366,-122.402325")
reverse_geocode("37.7879493,-122.3961974")
calculate_driving_matrix("40.629041,-74.025606;40.630099,-73.993521", "40.644895,-74.013818;40.627177,-73.980853")

# Load the sales data
results_path = '/Users/tanishta/Desktop/GitHub/SCOPE/Dataset/salesdaily.csv'
data = pd.read_csv(results_path)

# Convert the 'date' column to datetime format and set it as the index
data['date'] = pd.to_datetime(data['date'], format='mixed', dayfirst=False)
data.set_index('date', inplace=True)

# Set the frequency explicitly to daily
data = data.asfreq('D')  # Change 'D' to the appropriate frequency for your data if needed

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
    plt.title(f'Forecast vs Actual for Product Category: {category}')
    plt.legend()
    plt.show()

# Save the forecast results to CSV
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
forecast_results_df = pd.DataFrame.from_dict(forecast_results, orient='index').transpose()
forecast_results_df.to_csv(f'/Users/tanishta/Desktop/GitHub/SCOPE/forecast_results_{timestamp}.csv', index=False)

# Save the MSE results to CSV
mse_results_df = pd.DataFrame(list(mse_results.items()), columns=['Product_Category', 'MSE'])
mse_results_df.to_csv(f'/Users/tanishta/Desktop/GitHub/SCOPE/mse_results_{timestamp}.csv', index=False)

print("Forecasting and MSE results saved successfully.")