
from urllib.request import urlopen
from zipfile import ZipFile
import json
import os
import re
import requests, io
import pandas as pd
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from dotenv import load_dotenv
load_dotenv()

url = 'https://divvy-tripdata.s3.amazonaws.com'

response = requests.get(url)
if response.status_code != 200:
    print(f"Failed to retrieve XML page: {response.status_code}")
    exit()

zip_urls = re.findall(r'([\w\-_]+\.zip)', response.text)

download_dir = os.environ["ZIP_DIR"]
extracted_dir = os.environ["CSV_DIR"]

os.makedirs(download_dir, exist_ok=True)
os.makedirs(extracted_dir, exist_ok=True)

new_files = []

for zip_filename in zip_urls:
    local_path = os.path.join(download_dir, zip_filename)

    if os.path.exists(local_path):
        print(f"Already exists, skipping : {zip_filename}")
        continue

    full_url = f"{url}/{zip_filename}"

    print(f"Downloading: {full_url}")
    zip_response = requests.get(full_url)

    if zip_response.status_code == 200:
        with open(local_path, 'wb') as f:
            f.write(zip_response.content)
        new_files.append(zip_filename)
        # zip_filename = os.path.join(download_dir, zip_url.split('/')[-1])

        # Save the ZIP file to the local directory
        with ZipFile(local_path, 'r') as zip_ref:
            print(f"Extracting: {zip_filename}")
            zip_ref.extractall(extracted_dir)
    else:
        print(f"Failed to download: {full_url} - Status: {zip_response.status_code}")