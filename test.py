import requests

url = "http://localhost:9696/predict"

client = {
    "square_footage": 3000,
    "num_bedrooms": 5,
    "num_bathrooms": 4,
    "year_built": 2010,
    "lot_size": 0.7,
    "garage_size": 3,
    "neighborhood_quality": 9
}

response = requests.post(url, json=client).json()

print(response)