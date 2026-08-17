# Each row in the CSV represents a district in California

from pathlib import Path

import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt

def load_housing_data():
    # Declare a path where you want to store the data
    # Tgz file - compressed archive file
    tarball_path = Path("datasets/housing.tgz")

    # Checks if path exists
    if not tarball_path.is_file():
        # If not, create the directory
        Path("datasets").mkdir(parents=True, exist_ok=True)

        # Fetch data from GitHub
        url = "https://github.com/ageron/data/raw/main/housing.tgz"

        # Download the file from the web address
        urllib.request.urlretrieve(url, tarball_path)

        # Open a compressed archive file (a tarball) and extracts its entire
        # contents into the datasets folder
        with tarfile.open(tarball_path) as housing_tarball:
            housing_tarball.extractall(path="datasets", filter="data")

    # Returns render of a table of values from the file
    return pd.read_csv("datasets/housing/housing.csv")

housing_full = load_housing_data()

print(housing_full.info()) # Get info on all columns, number of non-null
# values, and data type
print(housing_full["ocean_proximity"].value_counts()) # Get specific count of
# certain values or categories
pd.set_option('display.max_columns', None)

# Prevent horizontal line wrapping in the console (optional)
pd.set_option('display.expand_frame_repr', False)
print(housing_full.describe(include="all")) # Describes count, mean, standard

housing_full.hist(bins=50, figsize=(12, 8)) # Displays histograms of each
# column in the dataset
# bins=50 - divides value range of each column into 50 intervals
# figsize=(12,8) - sets width to 12 inches and height to 8 inches
plt.show()