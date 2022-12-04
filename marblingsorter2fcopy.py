import streamlit as st
import tensorflow as tf
import streamlit as st
from pathlib import Path

@st.cache(allow_output_mutation=True)
def load_model():

    save_dest = Path('model')
    save_dest.mkdir(exist_ok=True)
    
    f_checkpoint = Path("model/content/drive/MyDrive/code/model_save/my_model1.hdf5")

    if not f_checkpoint.exists():
        with st.spinner("Downloading model... this may take awhile! \n Don't stop it!"):
            from GD_download import download_file_from_google_drive
            download_file_from_google_drive(cloud_model_location, f_checkpoint)
    
    model = torch.load(f_checkpoint, map_location=device)
    model.eval()
    return model
model=load_model()
from PIL import Image, ImageOps
st.write("""
         # Beef Classification
         """
         )
image = Image.open('/content/download.jpg')
st.image(image, caption='Made in waya')
file = st.file_uploader("Please upload an brain scan file", type=["jpg", "png"])
class_names=['Group1-Select','Group2-Select','Group3-Choice','Group4-Choice','Group5-Prime','Group6-Prime']
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
    
        size = (180,180)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img[np.newaxis,...]
    
        predictions = model.predict(img_reshape)
        
        return predictions
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    st.write(predictions)
    st.write(score)
    pred_class=class_names[np.argmax(predictions)]
    st.write("Predicted Class:",pred_class)
    
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
)
