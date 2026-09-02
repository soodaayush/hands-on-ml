import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray, ones, uint8
import cv2 as cv
import pickle

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

    return plt

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

    # Converts image into a NumPy array, where each number is a pixel
    a = asarray(resized_img)

    # Turns our image from greyscale to black and white
    # ret2 - threshold needed to produce black or white
    # th2 - image binary
    ret2, th2 = cv.threshold(a, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Inverts colors
    th2 = 255 - th2

    dist = cv.distanceTransform(th2, cv.DIST_L2, 5)
    max = cv.minMaxLoc(dist)
    stroke_width = max * 2

    st.write(max)

    kernel = ones((2, 2), uint8)

    th2 = cv.dilate(th2, kernel, anchor=(0, 0), iterations=1)

    # Displays an image of the adjusted binary
    plt = plot_digit(th2)
    st.pyplot(plt)

    # Turns 2D 28x28 grid into 1D array containing 784 values
    th2 = th2.flatten()
    plot_digit(th2)

    with open("model.pkl", "rb") as f:
        svm_clf = pickle.load(f)

    st.write(svm_clf.predict([th2]))
