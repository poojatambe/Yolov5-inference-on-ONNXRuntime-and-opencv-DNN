# Yolov5-inference-on-ONNXRuntime

Command to run the code

```
!python yolov5_onnxinfer.py  --image '/content/bus.jpg'  --weights '/content/yolov5s.onnx' --conf_thres 0.5 --iou_thres 0.5 --imgs 640 --cuda False 
```