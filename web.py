import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import tensorflow as tf

from tempfile import NamedTemporaryFile
from tensorflow.keras.preprocessing import image
import streamlit as st
import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np
import time



# Just making sure we are not bothered by File Encoding warnings
st.set_option('deprecation.showfileUploaderEncoding', False)


def main():
    menu = ['Home', 'Contact']
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        # Let's set the title of our awesome web app
        st.title('Auto Vaidya')
        # Now setting up a header text
        st.subheader("Automating Healthcare one problem at a time")

        def your_image_classifier(image):
            '''
            Function that takes the path of the image as input and returns the closest predicted label as output
            '''
            # Disable scientific notation for clarity
            np.set_printoptions(suppress=True)
            # Load the model
            model = tensorflow.keras.models.load_model('model/name_of_the_keras_model.h5')
            # Determined by the first position in the shape tuple, in this case 1.
            data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
            # Resizing the image to be at least 224x224 and then cropping from the center
            size = (224, 224)
            image = ImageOps.fit(image, size, Image.ANTIALIAS)
            # Turn the image into a numpy array
            image_array = np.asarray(image)
            # Normalize the image
            normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
            # Load the image into the array
            data[0] = normalized_image_array
            labels = {0: "Class 0", 1: "Class 1", 2: "Class 2",3: "Class 3", 4: "Class 4", 5: "Class 5"}
            # Run the inference
            predictions = model.predict(data).tolist()
            best_outcome = predictions[0].index(max(predictions[0]))
            print(labels[best_outcome])
            return labels[best_outcome]
        # Option to upload an image file with jpg,jpeg or png extensions
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
        # When the user clicks the predict button
        if st.button("Predict"):
        # If the user uploads an image
            if uploaded_file is not None:
                # Opening our image
                image = Image.open(uploaded_file)
                # Let's see what we got
                st.image(image,use_column_width=True)
                st.write("")
                try:
                    with st.spinner("The magic of our AI has started...."):
                        label = your_image_classifier(image)
                        time.sleep(8)
                    st.success("We predict this image to be: "+label)
                    rating = st.slider("Do you mind rating our service?",1,10)
                except:
                    st.error("We apologize something went wrong 🙇🏽‍♂️")
            else:
                st.error("Can you please upload an image 🙇🏽‍♂️")
    elif choice == "Contact":
        # Let's set the title of our Contact Page
        st.title('Get in touch')
        def display_team(name,path,affiliation="",email=""):
            '''
            Function to display picture,name,affiliation and name of creators
            '''
            team_img = Image.open(path)
            st.image(team_img, width=350, use_column_width=False)
            st.markdown(f"## {name}")
            st.markdown(f"#### {affiliation}")
            st.markdown(f"###### Email {email}")
            st.write("------")
        display_team("Your Awesome Name", "./assets/profile_pic.png","Your Awesome Affliation","hello@youareawesome.com")

st.set_option('deprecation.showfileUploaderEncoding', False)


@st.cache(allow_output_mutation=True)
def loading_model():
    menu = ['Home', 'Contact']
    choice = st.sidebar.selectbox("Menu", menu)

    
    fp = "./model/model.h5"
    model_loader = load_model(fp)
    return model_loader


cnn = loading_model()
st.write("""
# Jayes Tuberculosis X-Ray Classification Model
by Jayes and Team
""")


temp = st.file_uploader("Upload X-Ray Image")
#temp = temp.decode()

buffer = temp
temp_file = NamedTemporaryFile(delete=False)
if buffer:
    temp_file.write(buffer.getvalue())
    st.write(image.load_img(temp_file.name))


if buffer is None:
    st.text("Oops! that doesn't look like an image. Try again.")

else:

    img = image.load_img(temp_file.name, target_size=(
        500, 500), color_mode='grayscale')

    # Preprocessing the image
    pp_img = image.img_to_array(img)
    pp_img = pp_img/255
    pp_img = np.expand_dims(pp_img, axis=0)

    # predict
    preds = cnn.predict(pp_img)
    if preds >= 0.5:
        out = ('I am {:.2%} percent confirmed that this is a Tuberculosis case. You may need to get advice from the doctor'.format(
            preds[0][0]))

    else:
        out = ('I am {:.2%} percent confirmed that this is a Normal case'.format(
            1-preds[0][0]))

    st.success(out)

    image = Image.open(temp)
    st.image(image, use_column_width=True)
