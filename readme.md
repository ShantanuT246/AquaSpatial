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

---example code (test)
import get_soil as s
import get_rainfall as r

lat = 13.0827
lon = 79.2707

print(s.get_soil_type(lat, lon))
print(r.get_rainfall_data(lat, lon))
print(s.get_soil_slope(lat,lon)[1])