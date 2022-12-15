# Yolov5-inference-on-ONNXRuntime

* To convert yolov5 weights to onnx format follow this tutorial link: https://github.com/ultralytics/yolov5/issues/251.

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