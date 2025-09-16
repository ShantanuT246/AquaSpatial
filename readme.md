#backend#
------rain
import get_rainfall as rain

rain.get_rainfall_data(latitude,longitude)

make sure lat and long is in india


-----soil texture
import get_soil as s

lat = 13.0827
lon = 80.2707

print(s.get_soil_type(lat, lon))