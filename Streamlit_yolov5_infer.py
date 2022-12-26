import streamlit as st
from Yolov5_infer_opencv import detectv5
from yolov5_onnxinfer import ort_v5
import shutil
from PIL import Image

st.set_page_config(
    page_title="YOLOV5 Inference App",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.header('YOLOV5 Inference with ONNX model ')

st.sidebar.header('Inference Method')
option=st.sidebar.selectbox('Select inference method:',('None','ONNXRuntime','OpenCV DNN'))
if option== 'None':
    pass
elif option=='ONNXRuntime':
    st.sidebar.subheader('Confidence Threshold')
    conf=st.sidebar.slider('Specify confidence threshold:',min_value=0.0, max_value=1.0, step=0.1, value=0.7)
    st.sidebar.subheader('IOU Threshold')
    iou=st.sidebar.slider('Specify IOU threshold:',min_value=0.0, max_value=1.0, step=0.1, value=0.5)
else:
    st.sidebar.subheader('Confidence Threshold')
    conf_thres=st.sidebar.slider('Specify confidence threshold:',min_value=0.0, max_value=1.0, step=0.1, value=0.7)
    st.sidebar.subheader('Score Threshold')
    score_thres=st.sidebar.slider('Specify score threshold:',min_value=0.0, max_value=1.0, step=0.1, value=0.5)
    st.sidebar.subheader('NMS Threshold')
    NMS_thres=st.sidebar.slider('Specify NMS threshold:',min_value=0.0, max_value=1.0, step=0.1, value=0.5)
col1, col2 = st.columns([2,1])
upload=col2.file_uploader('Upload image', type= ['jpg', 'png'])
if upload is None:
   col2.write('Upload image.')
else:
    col1.image(upload) 
    type= upload.type.split('/')[1] 
    in_file= open('upload.{}'.format(type), 'wb') 
    shutil.copyfileobj(upload, in_file)
    submit= st.button('Submit')
    if submit and option== 'ONNXRuntime':
        infer=ort_v5('upload.{}'.format(type), './yolov5s.onnx', conf, iou, 640, './classes.txt')
        infer()
        st.subheader('Output Prediction')
        st.image(Image.open('result.jpg'))
    elif submit and option== 'None':        
        st.write('Select inference method.')
    elif submit and option== 'OpenCV DNN':
        infer=detectv5('upload.{}'.format(type), './yolov5s.onnx',640, 640, conf_thres, score_thres, NMS_thres, './classes.txt')
        infer()
        st.subheader('Output Prediction')
        st.image(Image.open('result.jpg'))
