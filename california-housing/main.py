# Each row in the CSV represents a district in California

from pathlib import Path

import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

# This function determines whether a specific data row belongs in the test
# dataset by checking its unique ID number using the CRC32 hash function
def is_id_in_test_set(identifier, test_ratio):
    # This line hashes the row ID, feeds it into the hashing algorithm
    # Outputs an integer then applies the test_ratio
    # If the number fals below the boundary, it returns True (put in test
    # set). Otherwise, it returns False (put it in training set)
    return crc32(np.int64(identifier)) < test_ratio * 2**32

# Builds on the previous function as
def split_data_with_id_hash(data, test_ratio, id_column):
    # Extracts all unique IDs from the data
    ids = data[id_column]

    # Extracts all unique IDs that are in the test set
    in_test_set = ids.apply(lambda id_: is_id_in_test_set(id_, test_ratio))

    # Splits original DataFrame into two distinct sets
    # ~ - means bitwise NOT
    return data.loc[~in_test_set], data.loc[in_test_set]

# This function takes a dataset and splits the data into two parts: a testing
# set and training set
# data - pandas DataFrame
# test_ratio - a decimal number between 0 and 1 for an x% test split
# rng - A NumPy random generator instance used to handle the randomness of
# data assortment
def shuffle_and_split_data(data, test_ratio, rng):
    # Creates a randomly shuffled list of row indices
    shuffled_indices = rng.permutation(len(data))

    # Calculate how many rows belong in the test set
    test_set_size = int(len(data) * test_ratio)

    # Extracts the random row indices designated for testing data
    test_indices = shuffled_indices[:test_set_size]

    # Extracts the random row indices designated for training data
    train_indices = shuffled_indices[test_set_size:]

    # Uses the selected random row indices to extract the actual data rows
    # and returns both datasets
    return data.iloc[train_indices], data.iloc[test_indices]

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

# Creates NumPy's random number generator
rng = np.random.default_rng()

# Takes the dataset and turns index of each row into a brand new, permenant
# column named index
# This is not ideal because if we split our data into two sets, the IDs will
# change as the order will be changed
# If IDs change, rows will accidentally jump between the testing and training
# sets, ruining our machine learning results
housing_with_id = housing_full.reset_index()

# Because a house's physical location never changes, this ensures the IDs
# will stay constant forever, which is why this would be utilized for IDs
# instead
housing_with_id["id"] = (housing_full["longitude"] * 1000 + housing_full[
    "latitude"])

# Execute the hash splitting function, where 20% of the data is allocated to
# the test set and 80% to the train set
# train_set, test_set = split_data_with_id_hash(housing_with_id, 0.2, "index")

# Scikit-learn's train_test_split method does exactly what our own shuffle_and_
# split_data method does with some additional features.
# The random_state parameter allows you to set the random generator seed
# As well, you can pass in multiple datasets
train_set, test_set = train_test_split(housing_full, test_size=0.2,
                                       random_state=42)

# Since we are told that median income is a very important attribute to
# predict median housing prices, we must organize each median income into a
# category

# Groups income into 5 distinct buckets utilizing boundaries (or bins) and
# labels each category from 1 to 5
housing_full["income_cat"] = pd.cut(housing_full["median_income"], bins=[0.,
                                                                         1.5,
                                                                         3.0,
                                                                         4.5,
                                                                         6.,
                                                                         np.inf], labels=[1, 2, 3, 4, 5])

# Counts how many districts fall into each of the 5 categories
cat_counts = housing_full["income_cat"].value_counts().sort_index()

# Builds a vertical bar chart using the counts we calculated
cat_counts.plot.bar(rot=0, grid=True)

# Labels x and y axes
plt.xlabel("Income Category")
plt.ylabel("Number of Districts")

# Displays graph
plt.show()

# Stratified Sampling - a research method where you split a large group of
# data into smaller, non-overlapping groups based on shared attributes

splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
strat_splits = []
for train_index, test_index in splitter.split(housing_full, housing_full[
    "income_cat"]):
    strat_train_set_n = housing_full.iloc[train_index]
    strat_test_set_n = housing_full.iloc[test_index]
    strat_splits.append([strat_train_set_n, strat_test_set_n])

strat_train_set, start_test_set = strat_splits[0]