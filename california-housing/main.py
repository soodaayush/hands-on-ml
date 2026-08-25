# Each row in the CSV represents a district in California
from math import gamma
from pathlib import Path

import pandas as pd
import tarfile
import urllib.request
import matplotlib.pyplot as plt
import numpy as np
from zlib import crc32
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (OrdinalEncoder, OneHotEncoder,
                                   MinMaxScaler, StandardScaler, FunctionTransformer)
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.linear_model import LinearRegression
from sklearn.compose import TransformedTargetRegressor

imputer = SimpleImputer(strategy="median")

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


# A tool that splits data into train/test, keeping income_cat proportions the
# same in both. Makes 10 versions

# The reason why we want equal proportions is so that the data isn't skewed
# to a specific income category

splitter = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42)

strat_splits = [] # Empty list to store the 10 splits

for train_index, test_index in splitter.split(housing_full, housing_full[
    "income_cat"]):
    strat_train_set_n = housing_full.iloc[train_index] # Pick out train rows
    strat_test_set_n = housing_full.iloc[test_index] # Pick out test rows
    strat_splits.append([strat_train_set_n, strat_test_set_n]) # Save the pair

# strat_train_set, strat_test_set = strat_splits[0]

# Easier way of doing the stratified split (compared to the loop above),
# no loop needed
strat_train_set, strat_test_set = train_test_split(housing_full,
                                                   test_size=0.2,
                                                   stratify=housing_full[
                                                       "income_cat"],
                                                   random_state=42)

# This checks the income category proportions in our test set to confirm if
# the stratified split kept the proportions balanced and matched to the full
# dataset
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

# Drop the entire income_cat column as we are done utilizing it
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()

# Displays a scatter plot shaped like the state of California, where each dot
# denotes a district
# Setting an alpha of 0.2 allows us to differentiate high density districts
# and low density districts
# Radius of each circle represents the district's population and color
# represents the price
housing.plot(kind="scatter", x="longitude", y="latitude", grid=True, alpha=0.1)
plt.show()

# In this plot, you can see where homes are located (position), how populated
# their area is (dot size), and how expensive homes are there (dot color)

# s - size of each dot
# label - label each dot in accordance to their population
# c - controls the color of each dot based on the district's median house value
# cmap - the color scheme to differentiate ranges of median house values
# colorbar - shows a colorbar on the side for reference
# legend - shows legend explaining what each dot size means
# sharex - a technical fix for a quirk in Pandas/Matplotlib where the x-axis
# label will sometimes not show up correctly

housing.plot(kind="scatter", x="longitude", y="latitude", grid=True,
             s=housing["population"] / 100, label="population",
             c="median_house_value", cmap="jet", colorbar=True, legend=True,
             sharex=False, figsize=(10, 7))

plt.show()

# These lines calculate the standard correlation coefficient (also called
# Pearson's r) between every pair of numerical attributes (attribute,
# median_house_value)

# In simple words, these lines show how much correlation each attribute has
# with median house prices
# Example: population doesn't have much correlation while median_income does

# Close to 1 - strong positive correlation
# Close to -1 - strong negative correlation
# Close to 0 - no linear correlation

# Positive Example: when median_income increases, median_house_value increases
# Negative Example: when latitude decreases, median_house_values increases

corr_matrix = housing.corr(numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# This creates a grid of scatter plots showing how every pair of these four
# features relates to each other, all at once

# Picking out columns to inspect
attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]

# The function takes the four columns and builds a grid of plots; every
# attribute plotted against every other attribute, creating a 4x4 grid
scatter_matrix(housing[attributes], figsize=(12, 8))

plt.show()

# This shows a graph of median income vs. median house value, as it is the
# most promising attribute of the 16 graphs that has a positive correlation
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1, grid=True)

plt.show()

# Since we would like to determine other attributes such as rooms per house,
# bedrooms ratio, and people per house, we can calculate those from our
# available data
housing["rooms_per_house"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_ratio"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["people_per_house"] = housing["population"] / housing["households"]

# We are recalculating the correlations between the attributes and median
# house prices because we added new columns for the three new attributes above

corr_matrix = housing.corr(numeric_only=True)
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# We are separating the features (population, latitude, etc.) and the labels (
# median house value) in order to conduct supervised learning

housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# We need to clean up our data as we have some missing features
# There are three options
# Option 1 - drop all districts that do not have a total_bedrooms value
# Option 2 - remove the entire attribute from the dataset
# Option 3 - set the missing values to some value (zero, the mean,
# the median, etc.)

# inplace - edits the dataset directly rather than making a copy
# housing.dropna(subset=["total_bedrooms"], inplace=True) # Option 1

# axis=1 - means column
# housing.drop("total_bedrooms", axis=1, inplace=True) # Option 2

median = housing["total_bedrooms"].median() # Option 3

# fillna - fills all missing values (NaN) in the total_bedrooms column with
# the median total bedroom value
housing["total_bedrooms"] = housing["total_bedrooms"].fillna(median)

# Filters out all non-numerical labels from the dataset as median
# calculations could be inaccurate due to non-numerical values
housing_num = housing.select_dtypes(include=[np.number])

# Calculates medians for all numerical columns in the dataset
# We apply this to each column because we may not be aware of other missing
# data in other columns, so a blanket calculation rids us of this worry
imputer.fit(housing_num)

# Displays medians calculated for each column
print(imputer.statistics_)

# Replaces all missing values with the learned medians
X = imputer.transform(housing_num)

# This line takes a NumPy array generated by the imputer and converts it back
# into a clean and labeled Pandas DataFrame
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)

# Since machine learning models work best with numerical attributes, we can
# convert each ocean proximity category to a number
housing_cat = housing[["ocean_proximity"]]
print(housing_cat.head(8))

# Sci-kit learn's OrdinalEncoder is a tool that does exactly that
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoded[:8])
print(ordinal_encoder.categories_)

# When we assign simple numbers (e.g. 1, 2, 3) to categories, machine
# learning models can get confused by math
# For an example: It could think that NEAR OCEAN (3) is bigger and "better"
# than INLAND (1), when there inherently is no number better than the other

# To solve this, we use One-Hot Encoding - instead of one column with numbers
# 1, 2, or 3, we split it into three separate columns (one for each
# location). The computer answers Yes (1) or No (0) for each column
# 1 - is it inland?
# 0 - is it near bay?
# 0 - is it near ocean?

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
print(housing_cat_1hot)

# This converts housing_cat_1hot, which was a sparse matrix, to a dense array
cat_encoder = OneHotEncoder(sparse_output=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

# Prints a truth table for specified categories below
df_test_unknown = pd.DataFrame({"ocean_proximity": ["INLAND", "NEAR BAY"]})
print(pd.get_dummies(df_test_unknown))
print(cat_encoder.transform(df_test_unknown))

# Takes our encoded matrix filled with 0's and 1's and wraps it back into a
# readable Pandas DataFrame with correct column names and row labels
df_output = pd.DataFrame(cat_encoder.transform(df_test_unknown),
                         columns=cat_encoder.get_feature_names_out(),
                         index=df_test_unknown.index)

# Machine learning algorithms do not perform well when there is a really
# large scale of numbers for a given category (e.g. the total number of rooms
# ranging from 6 to 39,320)
# We need to scale down the values to a smaller range. Without that,
# most models would be biased toward ignoring the median income and focusing
# more on the number of rooms

# We find the lowest value in a column and call it -1, and the highest being
# 1. Scale eveything else proportionally in between
min_max_scaler = MinMaxScaler(feature_range=(-1, 1))

# We apply min-max formula to all columns
housing_num_min_max_scaled = min_max_scaler.fit_transform(housing_num)

# Another way of scaling these ranges is by using a type of feature scaling
# called Standardization (or Z-score normalization)
# Instead of squishing your numbers into a strict range like min-max,
# this method centers our data based on statistics

std_scalar = StandardScaler()
housing_num_std_scaled = std_scalar.fit_transform(housing_num)

age_simil_35 = rbf_kernel(housing[["housing_median_age"]], [[35]], gamma=0.1)

# Scale down values for target values (median_house_value)
target_scaler = StandardScaler()
scaled_labels = target_scaler.fit_transform(housing_labels.to_frame())

# Train a plain linear regression model, where we are predicting the scaled
# labels
model = LinearRegression()
model.fit(housing_labels[["median_income"]], scaled_labels)
some_new_data = housing_labels[["median_income"]].iloc[:5]

# Model predicts based of off 5 example rows of median_income
scaled_predictions = model.predict(some_new_data)

# Convert the scaled predictions back into real units (actual house dollar
# prices)
predictions = target_scaler.inverse_transform(scaled_predictions)

# A less tedious, more automated way of doing the same thing as above
model = TransformedTargetRegressor(LinearRegression(),
                                   transformer=StandardScaler())
model.fit(housing[["median_income"]], housing_labels)
predictions = model.predict(some_new_data)

# Transforming - to take your data and reshape/rescale it into a different
# form that is easier for the model to learn from, without changing what the
# data actually represents underneath

# In this case, population has a problem: Most districts have small
# populations, but a few have exceptionally huge populations, stretching out
# the data distribution unevenly. This lopsided shape is harder for a model
# to learn patterns form

# To fix this, we take the log of each value, squishing big numbers down a
# lot more than the small numbers, pulling the bigger population closer to
# the smaller ones

# inverse_func allows us to undo our changes after our model is done
# predicting/training using the log values

log_transformer = FunctionTransformer(np.log, inverse_func=np.exp)
log_pop = log_transformer.transform(housing_labels[["population"]])

# We create a new feature measuring "how close is this house's age to 35
# years old?" - we turn this raw number into a score
rbf_transformer = FunctionTransformer(rbf_kernel, kw_args=dict(Y=[[35.]],
                                                               gamma=0.1))
age_simil_35 = rbf_transformer.transform(housing[["housing_median_age"]])