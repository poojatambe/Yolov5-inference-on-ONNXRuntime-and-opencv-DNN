# Yolov5-inference-on-ONNXRuntime

* To convert yolov5 weights to onnx format follow this tutorial link: https://github.com/ultralytics/yolov5/issues/251.


* Command to run the code:

```
!python yolov5_onnxinfer.py  --image 'bus.jpg'  --weights 'yolov5s.onnx' --conf_thres 0.5 --iou_thres 0.5 --imgs 640 --cuda False 
```