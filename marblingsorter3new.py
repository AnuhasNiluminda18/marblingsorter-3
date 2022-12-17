import streamlit as st
import tensorflow as tf
import streamlit as st
import pickle
from streamlit_option_menu import option_menu
from  PIL import Image
import webbrowser
import numpy as np
import cv2

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
  
    col1, col2 = st.columns( [0.6, 0.4])
    with col1:               # To display the header text using css style
        st.markdown(""" <style> .font {
        font-size:20px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<p class="font">Marbling is the visible unsaturated (healthy) intramuscular fat that accumulates within the muscle and between the muscle fibre bundles. Visually, marbling is soft intramuscular (between the muscle fibre) fat made up of polyunsaturated, monounsaturated and saturated fats</p>', unsafe_allow_html=True)    
        
    with col2:               # To display brand log
       # st.image(logo, width=130 )
    
        st.write("Web app – Web applications (web app) are popular in these days because anyone who has a device can be accessed to the internet easily than previously. Web app is hosted on a web server and it is delivered over the Internet through a browser interface. Conventional applications have to be installed on the device and then only it can be accessed. The Web app is convince in that situation; any browser you use ex- chrome, Mozilla Firefox or Safari can be used to access the web app(Postma & Goedhart, 2019) . ")    
    #st.image(profile, width=700 )
        image = Image.open('beefgradingcomparison.png')
  

          
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
    import streamlit as st

    from PIL import Image
    import numpy as np
    import cv2
    
    DEMO_IMAGE = 'beefgradingcomparison.png'

   #title of the web-app
st.title('QR Code Decoding with OpenCV')

@st.cache
def show_qr_detection(img,pts):
    
    pts = np.int32(pts).reshape(-1, 2)
    
    for j in range(pts.shape[0]):
        
        cv2.line(img, tuple(pts[j]), tuple(pts[(j + 1) % pts.shape[0]]), (255, 0, 0), 5)
        
    for j in range(pts.shape[0]):
        cv2.circle(img, tuple(pts[j]), 10, (255, 0, 255), -1)
 



st.markdown("**Warning** Only add QR-code Images, other images will give out an error")

#uploading the imges
img_file_buffer = st.file_uploader("Upload an image which you want to Decode", type=[ "jpg", "jpeg",'png'])

if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))

else:
    demo_image = DEMO_IMAGE
    image = np.array(Image.open(demo_image))


st.subheader('Orginal Image')

#display the image
st.image(
    image, caption=f"Original Image", use_column_width=True
) 
    
st.subheader('Decoded data')

@st.cache
def qr_code_dec(image):
    
    decoder = cv2.QRCodeDetector()
   
    data, vertices, rectified_qr_code = decoder.detectAndDecode(image)
    
    if len(data) > 0:
        print("Decoded Data: '{}'".format(data))
        
    # Show the detection in the image:
        show_qr_detection(image, vertices)
        decoded_data = 'Decoded data: '+ data
        rectified_image = np.uint8(rectified_qr_code)
        
   
        
        rectified_image = cv2.putText(rectified_image,decoded_data,(50,350),fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale = 2,
            color = (250,225,100),thickness =  3, lineType=cv2.LINE_AA)
        
        
    return decoded_data

decoded_data = qr_code_dec(image)
st.markdown(decoded_data)
