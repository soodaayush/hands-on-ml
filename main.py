import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Download and prepare data
data_root = "https://github.com/ageron/data/raw/main/"
lifesat = pd.read_csv(data_root + "lifesat/lifesat.csv")
X = lifesat[["GDP per capita (USD)"]].values # Set x-axis to measure GDP per
# capita
y = lifesat[["Life satisfaction"]].values # Set y-axis to measure life
# satisfaction

# Visualize the data
lifesat.plot(kind="scatter", grid=True, x="GDP per capita (USD)", y="Life "
                                                                    "satisfaction")
plt.axis([23_500, 62_500, 4, 9]) # States x-axis starts at $23,500 and ends
# at $62,500; y-axis begins at a score of 4 and ends at 9
plt.show() # Shows all plots on the screen

# Select a linear model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Make a prediction for Puerto Rico
X_new = [[33_422.8]] # Puerto Rico's GDP per capita in 2020
print(model.predict(X_new)) # Predicts a life satisfaction score based on
# Puerto Rico's GDP per capita