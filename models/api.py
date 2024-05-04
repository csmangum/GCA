# import json
# import os
# from datetime import datetime

# import requests
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # API Key from environment
# COIN_MARKET_API_KEY = os.getenv("COIN_MARKET_API")

# # API endpoint for cryptocurrency details
# url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
# headers = {
#     "Accept": "application/json",
#     "X-CMC_PRO_API_KEY": COIN_MARKET_API_KEY,
# }

# # Making the request
# response = requests.get(url, headers=headers)
# data = response.json()

# # Filter for BTC data
# btc_data = next(item for item in data["data"] if item["symbol"] == "BTC")

# # Append a timestamp to the data
# btc_data["timestamp"] = datetime.now().isoformat()

# # Path to the JSON file
# file_path = "coinmarketcap.json"

# # Append data to the JSON file
# if os.path.exists(file_path):
#     with open(file_path, "r+") as file:
#         # Read existing data
#         file_data = json.load(file)
#         # Append new data
#         file_data.append(btc_data)
#         # Set file's current position at offset
#         file.seek(0)
#         # Convert back to json and write to file
#         json.dump(file_data, file, indent=4)
# else:
#     with open(file_path, "w") as file:
#         # Write new data as a list
#         json.dump([btc_data], file, indent=4)


import os
from datetime import datetime
import requests
from google.cloud import firestore
import firebase_admin

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Key from environment
COIN_MARKET_API_KEY = os.getenv("COIN_MARKET_API")


def fetch_and_store_crypto_data(request):
    """
    Fetches cryptocurrency data from CoinMarketCap and stores it in Google Firestore.
    Triggered by HTTP request.
    """
    project_id = "likeendeavor"

    # Application Default credentials are automatically created.
    app = firebase_admin.initialize_app()
    # Initialize Firestore client
    db = firestore.Client(project=project_id, database="cryptocurrency")

    # API endpoint for cryptocurrency details
    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    headers = {
        "Accept": "application/json",
        "X-CMC_PRO_API_KEY": COIN_MARKET_API_KEY,
    }

    # Making the request to the CoinMarketCap API
    response = requests.get(url, headers=headers)
    data = response.json()

    # Filter for BTC data
    btc_data = next(item for item in data["data"] if item["symbol"] == "BTC")

    # Append a timestamp to the data
    btc_data["timestamp"] = datetime.now().isoformat()

    # Store data in Firestore
    db.collection("BTC").add(btc_data)


if __name__ == "__main__":
    fetch_and_store_crypto_data("request")
