import cv2
import numpy as np
import argparse
import time


class detectv5:
    def __init__(self, img_path, model, imgs_w,imgs_h, conf_threshold,score_threshold,nms_threshold, classes_txt):
        self.conf= conf_threshold
        self.score=score_threshold
        self.nms=nms_threshold
        self.img= img_path
        self.model= model
        self.img_w= imgs_w
        self.img_h = imgs_h
        self.classes_file= classes_txt

    def __call__(self):
        img= cv2.imread(self.img)
        net = cv2.dnn.readNetFromONNX(self.model)
        classes= self.class_name()
        self.detection(img, net, classes)
        
         

    def class_name(self):
        classes=[]
        file= open(self.classes_file,'r')
        while True:
            name=file.readline().strip('\n')
            classes.append(name)
            if not name:
                break
        return classes

    def detection(self, img, net, classes): 
        blob = cv2.dnn.blobFromImage(img, 1/255 , (self.img_w, self.img_h), swapRB=True, mean=(0,0,0), crop= False)
        net.setInput(blob)
        t1= time.time()
        outputs= net.forward(net.getUnconnectedOutLayersNames())
        t2= time.time()
        out= outputs[0]
        print('Opencv dnn yolov5 inference time: ', t2- t1)
        # print(out.shape)
        n_detections= out.shape[1]
        height, width= img.shape[:2]
        x_scale= width/self.img_w
        y_scale= height/self.img_h
        conf_threshold= self.conf
        score_threshold= self.score
        nms_threshold=self.nms

        class_ids=[]
        score=[]
        boxes=[]



        for i in range(n_detections):
            detect=out[0][i]
            confidence= detect[4]
            if confidence >= conf_threshold:
                class_score= detect[5:]
                class_id= np.argmax(class_score)
                if (class_score[class_id]> score_threshold):
                    score.append(confidence)
                    class_ids.append(class_id)
                    x, y, w, h = detect[0], detect[1], detect[2], detect[3]
                    left= int((x - w/2)* x_scale )
                    top= int((y - h/2)*y_scale)
                    width = int(w * x_scale)
                    height = int(h *y_scale)
                    box= np.array([left, top, width, height])
                    boxes.append(box)


        indices = cv2.dnn.NMSBoxes(boxes, np.array(score), conf_threshold, nms_threshold)
        # print(indices)
        for i in indices:
            box = boxes[i[0]]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3] 
            cv2.rectangle(img, (left, top), (left + width, top + height), (0, 0, 255), 3)
            label = "{}:{:.2f}".format(classes[class_ids[i[0]]], score[i[0]])
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)
            dim, baseline = text_size[0], text_size[1]
            cv2.rectangle(img, (left, top), (left + dim[0], top + dim[1] + baseline), (0,0,0), cv2.FILLED)
            cv2.putText(img, label, (left, top + dim[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1, cv2.LINE_AA)
            print('predictions: ', box, class_ids[i[0]], score[i[0]] )

        cv2.imwrite('result.jpg', img)
        # cv2.imshow('output',img)    
        # cv2.waitKey(0)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image',help='Specify input image', default= '', type=str)
    parser.add_argument('--weights', default='./yolov5s.onnx', type=str, help='model weights path')
    parser.add_argument('--imgs_w', default=640,type= int, help='image size')
    parser.add_argument('--imgs_h', default=640,type= int, help='image size')
    parser.add_argument('--conf_thres',default= 0.7, type=float, help='confidence threshold')
    parser.add_argument('--score_thres',type= float, default= 0.5, help='iou threshold')
    parser.add_argument('--nms_thres',type= float, default= 0.5, help='nms threshold')
    parser.add_argument('--classes',type=str,default='', help='class names')
    opt= parser.parse_args()
    instance= detectv5( opt.image, opt.weights, opt.imgs_w, opt.imgs_h, opt.conf_thres, opt.score_thres, opt.nms_thres, opt.classes)
    instance()
    
