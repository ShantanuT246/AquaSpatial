import imdlib as im
import numpy as np


lat = 12.9165
lon = 79.1325
file_dir = 'backend'


data = im.open_data('rain', 2023, 2023, file_dir=file_dir)  # Adjust years as needed
ds = data.get_xarray()


lats = ds['lat'].values
lons = ds['lon'].values


lat_idx = (np.abs(lats - lat)).argmin()
lon_idx = (np.abs(lons - lon)).argmin()


rain_da = ds['rain']
rain_series = rain_da.isel(lat=lat_idx, lon=lon_idx).to_numpy()


# print(type(rain_series))  # <class 'numpy.ndarray'>
# print(rain_series[:5])    # sample values from time series


days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
q1 = sum(days_in_months[:3])
q2 = sum(days_in_months[:6])
q3 = sum(days_in_months[:9])


q1_avg = np.mean(rain_series[:q1])
q2_avg = np.mean(rain_series[q1:q2])
q3_avg = np.mean(rain_series[q2:q3])
q4_avg = np.mean(rain_series[q3:])
annual_avg = np.mean(rain_series)


result = (round(q1_avg, 2), round(q2_avg, 2), round(q3_avg, 2), round(q4_avg, 2), round(annual_avg, 2))
print(result)


# # data = im.get_data('rain',start_yr=2023,end_yr=2023)
# # command to download yearwise data
