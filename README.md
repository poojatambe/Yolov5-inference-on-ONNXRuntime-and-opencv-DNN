# Yolov5-inference-on-ONNXRuntime

* To convert yolov5 weights to onnx format follow this tutorial link: https://github.com/ultralytics/yolov5/issues/251.

* yolov5.onnx model in Netron
<img width="941" alt="v5onnx_nms" src="https://user-images.githubusercontent.com/64680838/207949434-6cb25740-57d1-420b-8384-f81cb33fd284.PNG">


* For GPU system install ONNXRuntime-GPU library and ONNXRuntime for CPU system.
```
!pip install -r requirements.txt
```

* Command to run the code:

```
!python yolov5_onnxinfer.py  --image 'bus.jpg'  --weights 'yolov5s.onnx' --conf_thres 0.7 --iou_thres 0.5 --imgs 640 --classes 'classes.txt'
```
* Arguments Details:
1. Input image
2. Weight file
3. Confidence threshold value
4. IOU threshold value
5. Image size
6. Classes.txt file

* References
1. https://github.com/ultralytics/yolov5
2. https://github.com/WongKinYiu/yolov7
