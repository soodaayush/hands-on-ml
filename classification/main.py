from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt

# This imports a Stochastic Gradient Descent classifier

# Gradient descent: a model's error is like a landscape, you find the lowest
# point (least error) by repeatedly checking the slope and stepping downhill.
# Basically a model is racing to the bottomest part of the land it can find
# to achieve the least error
# Stochastic Gradient Descent checks one random example per step to determine
# if its on the right tracker rather than measuring slope. This is much
# faster, but results in a zaggy path downhill
# Ideal for classification as there are many images to process

from sklearn.linear_model import SGDClassifier

# Cross validation splits our training data into several chunks and trains
# the model on some chunks and tests on the rest. It repeats this process
# several times and averages the scores of how accurate your model is at
# predicting based off of the data

# cross_val_score runs cross-validation for us and gives back the accuracy
from sklearn.model_selection import cross_val_score

# Function for rendering images of digits
def plot_digit(image_data):
    # We resize the image to be 28x28 pixels
    image = image_data.reshape(28, 28)
    # Rendering binary data as a 2D image
    plt.imshow(image, cmap="binary")
    # We disable the axis
    plt.axis("off")

# We are importing the MNIST dataseet, a set of 70,000 small images of digits
# handwritten by high school students and employees of the US Census Bureau
mnist = fetch_openml("mnist_784", as_frame=False)

# Features and labels are being defined
X, y = mnist.data, mnist.target
print(X)
# Prints 70,000 images, each containing 784 features
print(X.shape)
# Contains labels (e.g. 1, 2, 3, 4, 5)
print(y)
# Prints number of labels (70,000)
print(y.shape)

some_digit = X[0] # Binary representation of 5
print(y[0]) # Prints 5
plot_digit(some_digit)
plt.show() # Renders the image

# Since we are machine learning engineers, we must split our data into a
# training and testing set. Fortunately, it is already organized for us,
# as the training set is the first 60,000 images and the test set is the last
# 10,000 images
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# We are now going to create something called a binary classifier. This
# allows us to solely identify one digit. In this case, the classifier
# distinguishes between two categories (or classes): 5, and non-5
y_train_5 = (y_train == '5')
y_test_5 = (y_test == '5')

# We initialze our SGD classifier; remember that random_state ensures that a
# specific random order is the same every time you run this code
# We train the SGD using our training data and the binary classifier
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)

# Since we inputted some_digit, which is 5, this outputs true
print(sgd_clf.predict([some_digit]))

# We are running a cross-validation on our SGD Classifier, inputting our
# training features and labels, splitting the data into three folds (cv) and
# measuring for accuracy

# We consistently have at least 95% accuracy, which is good!
print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring="accuracy"))