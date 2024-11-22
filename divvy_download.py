# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


from urllib.request import urlopen
from zipfile import ZipFile
import json
import os
import re
import requests, io
import pandas as pd
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup

url = 'https://divvy-tripdata.s3.amazonaws.com'

response = requests.get(url)
if response.status_code != 200:
    print(f"Failed to retrieve XML page: {response.status_code}")
    exit()

zip_urls = re.findall(r'([\w\-_]+\.zip)', response.text)

download_dir = 'downloaded_zips'
extracted_dir = 'extracted_files'

os.makedirs(download_dir, exist_ok=True)
os.makedirs(extracted_dir, exist_ok=True)

for zip_url in zip_urls:
    # The links on the page are relative, so prepend the base URL
    full_zip_url = f"https://divvy-tripdata.s3.amazonaws.com/{zip_url}"

    print(f"Downloading: {full_zip_url}")
    zip_response = requests.get(full_zip_url)

    if zip_response.status_code == 200:
        zip_filename = os.path.join(download_dir, zip_url.split('/')[-1])

        # Save the ZIP file to the local directory
        with open(zip_filename, 'wb') as zip_file:
            zip_file.write(zip_response.content)

        # Step 5: Extract the ZIP file
        with ZipFile(zip_filename, 'r') as zip_ref:
            print(f"Extracting: {zip_filename}")
            zip_ref.extractall(extracted_dir)
    else:
        print(f"Failed to download: {full_zip_url} - Status code: {zip_response.status_code}")