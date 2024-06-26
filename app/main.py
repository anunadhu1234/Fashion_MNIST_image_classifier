import streamlit as st #used to create  webapp 
import tensorflow as tf
from PIL import Image
import numpy as np
import os

working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = f"{working_dir}/trained_model/fashion_mnist_image_classifier.h5"
# Load the pre trained model
model = tf.keras.models.load_model(model_path)

#Define class labels for Fashion MNIST dataset
class_name = ['T-Shirt/top', 'Trouser', 'Pullover', 'Dress', 'Cast', 
              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

#fucntion to preprocess the uploaded image

def preprocess_image(image):
    img = Image.open(image)
    img = img.resize((28,28))
    img = img.convert('L') #convert to grayscale
    img_array = np.array(img)
    img_array = img_array.reshape((1,28,28,1))
    return img_array

#Streamlit App
st.title('Fashion Item Classifier')

uploaded_image = st.file_uploader("Upload an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((100,100))
        st.image(resized_img)
    
    with col2:
        if st.button('Classify'):
        #Preprocess the uploaded image
            img_array = preprocess_image(uploaded_image)

        #Make a prediction using the pre-trained model
            result = model.predict(img_array)
        # st.write(str(result))
            predicted_class = np.argmax(result) #returns the index
            prediction = class_name[predicted_class]

            st.success(f'Prediction: {prediction}')