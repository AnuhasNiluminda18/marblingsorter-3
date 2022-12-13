import streamlit as st
import tensorflow as tf
import streamlit as st
import pickle
from streamlit_option_menu import option_menu
from  PIL import Image
import webbrowser


with st.sidebar:
    choose = option_menu("Main menu", ["About", "Beef Marbling Sorter", "Beef price analysis",],
                         icons=['house', 'camera fill', 'kanban', 'book','person lines fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )
if choose == "About":
    col1, col2 = st.columns( [0.8, 0.2])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:35px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">About the Creator</p>', unsafe_allow_html=True)    
    with col2:               # To display brand log
       # st.image(logo, width=130 )
    
        st.write("Meat marbling is the pattern of distribution of intramuscular fat in beef steaks obtain from Rib-eye area (REA).")    
    #st.image(profile, width=700 )
    
st.markdown(""" <style> .font {
font-size:50px ; font-family: 'Cooper Black'; color: #FF9633;} 
</style> """, unsafe_allow_html=True)
st.markdown('<p class="font">Prime-Choice-Select</p>', unsafe_allow_html=True)


            
if choose == "Beef Marbling Sorter":
    @st.cache(allow_output_mutation=True)
    def load_model():
        picklefile = open("emp-model.pkl", "rb")
        model = pickle.load(picklefile)
        return model

    with st.spinner('Model is being loaded..'):
        model=load_model()
    from PIL import Image, ImageOps
   # st.write("""
           #  # Beef Marbling classifier
            # """
             #)
    from PIL import Image
    image = Image.open('beefgradingcomparison.png')

    st.image(image, caption='Made for your convenience')
    file = st.file_uploader("You can see the beef marbling status of your beef steak by uploading here", type=["jpg", "png"])
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
        #st.write(predictions)
        #st.write(score)
        pred_class=class_names[np.argmax(predictions)]
        st.write("Predicted Class:",pred_class)
        st.write("Place your feedback here [link](https://docs.google.com/forms/d/e/1FAIpQLSez6MK1CuUisH-j1rBjx1Bpoe1JwgA1bAIlV5MMD1rmbkJ1Bg/viewform?usp=sf_link)")
        print(
        #"This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100 * np.max(score))
)
if choose == "Beef price analysis":
    url ='https://www.streamlit.io/'

    if st.button('Open browser'):
         webbrowser.open_new_tab(url)
    link = '[GitHub](http://github.com)'
    st.markdown(link, unsafe_allow_html=True)
    link = '[More about marbling](https://www.google.com/url?sa=i&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=0CAQQw7AJahcKEwjIqursnff7AhUAAAAAHQAAAAAQAw&url=https%3A%2F%2Fwww.masterclass.com%2Farticles%2Fwhat-is-marbling-in-meat-learn-about-the-different-types-of-marbling-and-what-factors-impact-marbling&psig=AOvVaw1gyAbF4erVKoA8j2rbis89&ust=1671042845137472)'
    st.markdown(link, unsafe_allow_html=True)
