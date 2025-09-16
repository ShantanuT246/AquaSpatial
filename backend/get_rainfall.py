import imdlib as im
import numpy as np
import os

def get_rainfall_data(lat, lon, start_year=2023, data_dir='datasets/Rainfall_ind2023_rfp25.grd'):
    """
    Retrieves and processes IMD rainfall data for a specific location and year.

    This function downloads the data if not already present, finds the nearest
    grid point to the given latitude and longitude, and calculates the average
    rainfall for each of the four quarters and the full year.

    Args:
        lat (float): The latitude of the location.
        lon (float): The longitude of the location.
        start_year (int): The year for which to fetch the data. Defaults to 2023.
        data_dir (str): The directory to store the downloaded data files.
                        Defaults to 'backend'.

    Returns:
        tuple: A tuple in the format (q1_avg, q2_avg, q3_avg, q4_avg, annual_avg),
               with each value rounded to two decimal places.
               Returns None if an error occurs during data processing.
    """
    try:
        # Create the data directory if it doesn't exist to avoid errors
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
            print(f"Created directory: {data_dir}")

        # Open (and download if necessary) the rainfall data for the specified year
        # Using open_data is efficient as it won't re-download if the file exists
        data = im.open_data('rain', start_year, start_year, file_dir=data_dir)
        ds = data.get_xarray()

        # Get the latitude and longitude arrays from the dataset
        lats = ds['lat'].values
        lons = ds['lon'].values

        # Find the index of the grid point closest to the input coordinates
        lat_idx = (np.abs(lats - lat)).argmin()
        lon_idx = (np.abs(lons - lon)).argmin()

        # Extract the time series data for the identified grid point
        rain_da = ds['rain']
        rain_series = rain_da.isel(lat=lat_idx, lon=lon_idx).to_numpy()

        # # Define the number of days in each month for a non-leap year
        # # The imdlib data is daily, so we can split it by day counts.
        # days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

        # # Calculate the cumulative day number at the end of each quarter
        # q1_end = sum(days_in_months[:3])  # End of Q1 (Mar)
        # q2_end = sum(days_in_months[:6])  # End of Q2 (Jun)
        # q3_end = sum(days_in_months[:9])  # End of Q3 (Sep)

        # # Calculate the mean rainfall for each quarter and the entire year
        # # [0:q1_end] -> Jan, Feb, Mar
        # # [q1_end:q2_end] -> Apr, May, Jun
        # # [q2_end:q3_end] -> Jul, Aug, Sep
        # # [q3_end:] -> Oct, Nov, Dec
        # q1_avg = np.mean(rain_series[0:q1_end])
        # q2_avg = np.mean(rain_series[q1_end:q2_end])
        # q3_avg = np.mean(rain_series[q2_end:q3_end])
        # q4_avg = np.mean(rain_series[q3_end:])
        # # annual_avg = np.mean(rain_series)

        # # Format the result as a tuple of rounded values
        # result = (
        #     round(q1_avg, 2),
        #     round(q2_avg, 2),
        #     round(q3_avg, 2),
        #     round(q4_avg, 2),
        #     round(annual_avg, 2)
        # )
        #total anual rainafall
        annual_avg = np.sum(rain_series)
        result = round(annual_avg,2)
        return result

    except Exception as e:
        print(f"An error occurred while processing data for lat={lat}, lon={lon}: {e}")
        return None