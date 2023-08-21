import pandas as pd
import requests
import json
from tqdm import tqdm

def get_weather_api(lat, lon, year):
    mod_params = f"latitude={lat}&longitude={lon}&start_date={year}-01-01&end_date={year}-12-31&"
    url_base = f"https://archive-api.open-meteo.com/v1/archive?{mod_params}"\
                "daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,apparent_temperature_max,"\
                "apparent_temperature_min,apparent_temperature_mean,precipitation_sum,rain_sum,snowfall_sum,"\
                "precipitation_hours,windspeed_10m_max,windgusts_10m_max,winddirection_10m_dominant,"\
                "shortwave_radiation_sum,et0_fao_evapotranspiration&windspeed_unit=ms&timezone=auto"
    response = requests.get(url_base).json()
    return response

df_fc = pd.read_csv('data/Biomass_History.csv', index_col=0)
df_fc.head()

ls_years = [2018, 2019]
ls_coords = df_fc[["Latitude", "Longitude"]].values.tolist()

for year in ls_years:
    ls_weather = []
    print(f"Getting weather for year {year}")
    for i, coord in tqdm(enumerate(ls_coords), desc=str(year), unit="item"):
        lat, lon = coord
        # response = get_weather_api(lat, lon, year)
        try:
            response = get_weather_api(lat, lon, year)["daily"]
            # d_weather = response["daily"]
        except KeyError:
            response = {}

        ls_weather.append(response)

    with open(f"data/weather_{year}.json", "a") as f:
        json.dump(ls_weather, f)
    print("\n")
