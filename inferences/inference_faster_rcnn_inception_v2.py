#!/usr/bin/env python3
"""
 Copyright (c) 2018 Intel Corporation.

 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit persons to whom the Software is furnished to do so, subject to
 the following conditions:

 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from openvino.inference_engine import IENetwork, IECore
import os
import cv2
import numpy as np

BASE_DIR = "/".join(os.path.abspath(__file__).split('/')[:-2])+'/'

class Network:
    """
    Load and configure inference plugins for the specified target devices 
    and performs synchronous and asynchronous modes for the specified infer requests.
    """
    classes = {1:"PERSON"}

    def __init__(self,model_path_xml,cpu_extension=None,device='CPU'):
        self.cpu_extension = cpu_extension
        self.device = device
        self.name = "Faster RCNN Inception V2 Dataset COCO"
        self.model_xml = model_path_xml#base_dir + "models_files/faster_rcnn_inception_v2_coco/converted/frozen_inference_graph_600_600.xml"
        self.model_bin = os.path.splitext(self.model_xml)[0] + ".bin"

    def get_network_name(self):
        return self.name


    def load_model(self):
        self.plugin = IECore()
        # Add a CPU extension, if applicable
        if self.cpu_extension and "CPU" in self.device:
            self.plugin.add_extension(self.cpu_extension, self.device)
        net = IENetwork(model=self.model_xml, weights=self.model_bin)
        supported_layers = self.plugin.query_network(network=net, device_name="CPU")
        unsupported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(unsupported_layers) != 0:
            print("Unsupported layers found: {}".format(unsupported_layers))
            print("Check whether extensions are available to add to IECore.")
            exit(1)
        self.exec_net = self.plugin.load_network(net, "CPU")
        self.input_blob = next(iter(net.inputs))
        self.input_shape = net.inputs[self.input_blob].shape

    def get_input_shape(self):
        return self.input_shape

    def execute_net(self,frame,dets_confidence):

        def put_in_frame(filtered_dets, image):
            for det in filtered_dets:
                x1, y1, x2, y2 = det[:4]
                cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
            return image

        def preprocessing(input_image, height, width):
            image = np.copy(input_image)
            image = cv2.resize(image, (width, height))
            image = image.transpose((2, 0, 1))
            image = image.reshape(1, 3, height, width)
            return image

        def perform_inference(exec_net, image):
            preprocessed_image = preprocessing(image, 600,600)
            output = exec_net.infer(inputs={'image_tensor' : preprocessed_image,'image_info': (600, 600, 1) } )
            return output

        def nms(dets, score_threshold=0.8, beta=3):
            """
            Las detecciones deben venir en formato:
            [x1,y1,x2,y2,acc,class_to_be_detected]
            """

            def iou(bb_test, bb_gt):
                """
                Computes IUO between two bboxes in the form [x1,y1,x2,y2]
                """
                xx1 = np.maximum(bb_test[0], bb_gt[0])
                yy1 = np.maximum(bb_test[1], bb_gt[1])
                xx2 = np.minimum(bb_test[2], bb_gt[2])
                yy2 = np.minimum(bb_test[3], bb_gt[3])
                w = np.maximum(0., xx2 - xx1)
                h = np.maximum(0., yy2 - yy1)
                wh = w * h
                o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) +
                          (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
                return (o)

            filtered_dets = []
            total_dets = sorted(dets.copy(), key=lambda x: x[4], reverse=True)
            while len(total_dets) > 0:
                for i in range(1, len(total_dets)):
                    IOU = iou(total_dets[i][:4], total_dets[0][:4])
                    total_dets[i][4] *= np.exp(-beta * IOU)  # (1-IOU)
                temp = []
                for i in total_dets:
                    if i[4] >= score_threshold:
                        temp.append(i)
                if len(temp) > 0:
                    filtered_dets.append(temp[0])
                total_dets = sorted(temp[1:].copy(), key=lambda x: x[4], reverse=True)
                del temp
            return filtered_dets

        def filter(result, h, w,classes):
            dets = result['detection_output'][0][0]
            dets_fil = []
            index  = 0
            while dets[index][0]!=-1:
                det = dets[index]
                if det[1] in list(classes.keys()) and float(det[2]) >= dets_confidence:
                    x1, y1, x2, y2 = int(det[3] * w), int(det[4] * h), int(det[5] * w), int(det[6] * h)
                    if x1 < 0: x1 = 0
                    if x2 < 0: x2 = 0
                    if y1 < 0: y1 = 0
                    if y2 < 0: y2 = 0
                    dets_fil.append([x1, y1, x2, y2, round(float(det[2]), 4), int(det[1])])
                index+=1
            dets_fil = nms(dets_fil)
            return dets_fil

        img_h, img_w = frame.shape[:2]
        iamge_preprocessed = perform_inference(self.exec_net, frame)
        detections = filter(iamge_preprocessed, img_h, img_w,self.classes)
        return detections

    def execute_net_asyn(self,frame,request_id):
        def preprocessing(input_image, height, width):
            image = np.copy(input_image)
            image = cv2.resize(image, (width, height))
            image = image.transpose((2, 0, 1))
            image = image.reshape(1, 3, height, width)
            return image
        preprocessed_image = preprocessing(frame, 600,600)
        self.exec_net.start_async(request_id = request_id, inputs={'image_tensor' : preprocessed_image,'image_info': (600, 600, 1) })

    def wait(self,request_id):
        while True:
            status = self.exec_net.requests[request_id].wait(-1)
            if status == 0:
                break

    def get_output(self,request_id):
        outputs = self.exec_net.requests[request_id].outputs
        return outputs

    def parse_outputs(self,output,frame,dets_confidence):
        def put_in_frame(filtered_dets, image):
            for det in filtered_dets:
                x1, y1, x2, y2 = det[:4]
                cv2.rectangle(image.copy(), (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
            return image
        def nms(dets, score_threshold=0.3, beta=3):
            """
            Las detecciones deben venir en formato:
            [x1,y1,x2,y2,acc,class_to_be_detected]
            """

            def iou(bb_test, bb_gt):
                """
                Computes IUO between two bboxes in the form [x1,y1,x2,y2]
                """
                xx1 = np.maximum(bb_test[0], bb_gt[0])
                yy1 = np.maximum(bb_test[1], bb_gt[1])
                xx2 = np.minimum(bb_test[2], bb_gt[2])
                yy2 = np.minimum(bb_test[3], bb_gt[3])
                w = np.maximum(0., xx2 - xx1)
                h = np.maximum(0., yy2 - yy1)
                wh = w * h
                o = wh / ((bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1]) +
                          (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1]) - wh)
                return (o)

            filtered_dets = []
            total_dets = sorted(dets.copy(), key=lambda x: x[4], reverse=True)
            while len(total_dets) > 0:
                for i in range(1, len(total_dets)):
                    IOU = iou(total_dets[i][:4], total_dets[0][:4])
                    total_dets[i][4] *= np.exp(-beta * IOU)  # (1-IOU)
                temp = []
                for i in total_dets:
                    if i[4] >= score_threshold:
                        temp.append(i)
                if len(temp) > 0:
                    filtered_dets.append(temp[0])
                total_dets = sorted(temp[1:].copy(), key=lambda x: x[4], reverse=True)
                del temp
            return filtered_dets
        def filter(result, h, w,classes,dets_confidence):
            dets = result['detection_output'][0][0]
            dets_fil = []
            index  = 0
            while dets[index][0]!=-1:
                det = dets[index]
                if det[1] in list(classes.keys()) and float(det[2]) >= dets_confidence:
                    x1, y1, x2, y2 = int(det[3] * w), int(det[4] * h), int(det[5] * w), int(det[6] * h)
                    if x1 < 0: x1 = 0
                    if x2 < 0: x2 = 0
                    if y1 < 0: y1 = 0
                    if y2 < 0: y2 = 0
                    dets_fil.append([x1, y1, x2, y2, round(float(det[2]), 4), int(det[1])])
                index+=1
            dets_fil = nms(dets_fil)
            return dets_fil
        img_h, img_w = frame.shape[:2]
        out = filter(output, img_h, img_w, self.classes,dets_confidence)
        return out
