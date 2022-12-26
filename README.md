# Yolov5-inference-on-ONNXRuntime-and-Opencv-DNN

* To convert yolov5 weights to onnx format follow this tutorial link: https://github.com/ultralytics/yolov5/issues/251.

* yolov5.onnx model in Netron
<img width="941" alt="v5onnx_nms" src="https://user-images.githubusercontent.com/64680838/207949434-6cb25740-57d1-420b-8384-f81cb33fd284.PNG">

**ONNXRuntime**

* For GPU system install ONNXRuntime-GPU library and ONNXRuntime for CPU system.
```
!pip install -r requirements.txt
```

* Command to run the code:

```
!python yolov5_onnxinfer.py  --image 'bus.jpg'  --weights 'yolov5s.onnx' --conf_thres 0.7 \
      --iou_thres 0.5 --imgs 640 --classes 'classes.txt'
```
* Arguments Details:
1. Input image
2. Weight file
3. Confidence threshold value
4. IOU threshold value
5. Image size
6. Classes.txt file

**Opencv DNN**

* Command to run code:

```
!python Yolov5_infer_opencv.py --image ./bus.jpg --weights ./yolov5s.onnx \
        --classes ./classes.txt --imgs_w 640 --imgs_h 640 \
        --conf_thres 0.7 --score_thres 0.5 --nms_thres 0.5
```
* Arguments Details:
1. Input image
2. Weight file
3. Image width
4. Image Height
5. Confidence threshold value
6. score threshold value
7. nms threshold value
8. Classes.txt file

* Comparison of inference time:

For image 'bus.jpg', inference time of ONNXRuntime and opencv DNN module are:

  1. opencv DNN: 0.29987263679504395
 
 2. ONNXRuntime: 0.13161110877990723

**Streamlit yolov5 Inference App**

* Install streamlit.

```
!pip install streamlit
```

* Run stramlit code.

```
streamlit run Streamlit_yolov5_infer.py
```


* References
1. https://github.com/ultralytics/yolov5
2. https://github.com/WongKinYiu/yolov7
3. https://learnopencv.com/object-detection-using-yolov5-and-opencv-dnn-in-c-and-python/
