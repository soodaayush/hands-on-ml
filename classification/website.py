from sklearn.datasets import fetch_openml
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray
import cv2 as cv

from sklearn.svm import SVC

if "clicked" not in st.session_state:
    st.session_state.clicked = False

def run_model():
    st.session_state.clicked = True

def plot_digit(image_data):
    # We resize the image to be 28x28 pixels
    image = image_data.reshape(28, 28)
    # Rendering binary data as a 2D image
    plt.imshow(image, cmap="binary")
    # We disable the axis
    plt.axis("off")

st.title("Hello World!")

uploaded_file = st.file_uploader("Choose an image...")
st.button("Upload!", on_click=run_model)

if st.session_state.clicked:
    # Get byte data of image
    bytes_data = uploaded_file.getvalue()

    # Convert image into a grayscale image
    uploaded_file = Image.open(uploaded_file).convert('L')

    # Resize to a 28x28 px image
    resized_img = uploaded_file.resize((28, 28))
    st.image(resized_img)

    # Converts image into a NumPy array, where each number is a pixel
    a = asarray(resized_img)

    ret2, th2 = cv.threshold(a, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Inverts colors
    # th2 = 255 - th2

    # Turns 2D 28x28 grid into 1D array containing 784 values
    th2 = th2.flatten()
    print(th2)
    plot_digit(th2)

    # a = remove(a)

    # We are importing the MNIST dataseet, a set of 70,000 small images of digits
    # handwritten by high school students and employees of the US Census Bureau
    mnist = fetch_openml("mnist_784", as_frame=False)

    # Features and labels are being defined
    X, y = mnist.data, mnist.target

    digit = 20

    # Since we are machine learning engineers, we must split our data into a
    # training and testing set. Fortunately, it is already organized for us,
    # as the training set is the first 60,000 images and the test set is the last
    # 10,000 images
    X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[
        60000:]

    # We trained the SVM using the original target classes from 0 to 9 (y_train)
    # instead of the 5 vs. the rest target classes (y_train_5). Since there are
    # 10 classes (now more than 2), scikit learn uses OVO to train 45 binary
    # classifiers. We can now make predictions on digits!
    svm_clf = SVC(random_state=42)
    svm_clf.fit(X_train[:60000], y_train[:60000])

    # Instead of one model choosing directly among all 10 digits, Scikit-Learn
    # trains 45 mini-classifiers, each an expert at telling apart just one specific
    # pair of digits (like "3 or 7?"). A new digit gets judged by all 45 of them,
    # and whichever digit wins the most of these one-on-one matchups becomes the
    # final prediction.

    st.write(svm_clf.predict([th2]))
