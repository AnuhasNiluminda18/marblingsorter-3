import streamlit as st
from PIL import Image,ImageOps
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow.keras import preprocessing
from tensorflow.keras.activations import softmax
from tensorflow.keras.models import load_model 
import os 
import h5py 

st.header("marbling Sorter")

def main():
    file_uploaded=st.file_uploader("choose the file", type= ['jpg','png','jpeg'])
    if file_uploaded is not None:  
        image=Image.open(file_uploaded)
        figure=plt.figure()
        plt.imshow(image)
        plt.axis('off')
        results= predict_class(image)
        st.write(result)
        st.write(figure)
        
@st.cache
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

    test_image=image.resize((128,128))
    test_image=preprocessing.image.img_to_array(test_image)
    test_image=test_image/255.0
    test_image= np.expand_dims(test_image, axis=0)
    class_names=['G1','G2','G3','G4','G5','G6','G7','G8']
    predictions= model.predict(test_image)
    scores=tf.nn.softmax(predictions[0])
    scores=scores.numpy()
    image_class=class_names[np.argmax(scores)]
    results= "The photo you have uploaded is:{}".format(image_class)
    return results 
if __name__=="__main__":
    main()

