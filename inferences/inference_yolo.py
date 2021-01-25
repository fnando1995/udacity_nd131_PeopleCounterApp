# !/usr/bin/env python3
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
import math

BASE_DIR = "/".join(os.path.abspath(__file__).split('/')[:-2]) + '/'


def EntryIndex( side, lcoords, lclasses, location, entry):
    n = int(location / (side * side))
    loc = location % (side * side)
    return int(n * side * side * (lcoords + lclasses + 1) + entry * side * side + loc)

class DetectionObject():
    xmin = 0
    ymin = 0
    xmax = 0
    ymax = 0
    class_id = 0
    confidence = 0.0

    def __init__(self, x, y, h, w, class_id, confidence, h_scale, w_scale):
        self.xmin = int((x - w / 2) * w_scale)
        self.ymin = int((y - h / 2) * h_scale)
        self.xmax = int(self.xmin + w * w_scale)
        self.ymax = int(self.ymin + h * h_scale)
        self.class_id = class_id
        self.confidence = confidence


class Network:
    m_input_size = 416

    yolo_scale_13 = 13
    yolo_scale_26 = 26
    yolo_scale_52 = 52

    classes = 80
    coords = 4
    num = 3
    anchors_yolov3_tiny = [10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]
    anchors_yolov3 = [10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326]

    LABELS = ("person", "bicycle", "car", "motorbike", "aeroplane",
              "bus", "train", "truck", "boat", "traffic light",
              "fire hydrant", "stop sign", "parking meter", "bench", "bird",
              "cat", "dog", "horse", "sheep", "cow",
              "elephant", "bear", "zebra", "giraffe", "backpack",
              "umbrella", "handbag", "tie", "suitcase", "frisbee",
              "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
              "wine glass", "cup", "fork", "knife", "spoon",
              "bowl", "banana", "apple", "sandwich", "orange",
              "broccoli", "carrot", "hot dog", "pizza", "donut",
              "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
              "remote", "keyboard", "cell phone", "microwave", "oven",
              "toaster", "sink", "refrigerator", "book", "clock",
              "vase", "scissors", "teddy bear", "hair drier", "toothbrush")

    label_text_color = (255, 255, 255)
    label_background_color = (125, 175, 75)
    box_color = (255, 128, 0)
    box_thickness = 1

    def __init__(self,model_path_xml,cpu_extension=None,device='CPU'):
        self.cpu_extension = cpu_extension
        self.device = device
        self.model_xml = model_path_xml#base_dir + "models_files/yolov3/converted/frozen_darknet_yolov3_model_full.xml"
        self.model_bin = os.path.splitext(self.model_xml)[0] + ".bin"


        if 'tiny' in self.model_xml:
            self.anchors_selected = self.anchors_yolov3_tiny
            self.name = "Tiny Yolo v3 by mystic123"
        else:
            self.anchors_selected = self.anchors_yolov3
            self.name = "Yolo v3 by mystic123"


    def get_network_name(self):
        return self.name

    def IntersectionOverUnion(self,box_1, box_2):
        width_of_overlap_area = min(box_1.xmax, box_2.xmax) - max(box_1.xmin, box_2.xmin)
        height_of_overlap_area = min(box_1.ymax, box_2.ymax) - max(box_1.ymin, box_2.ymin)
        area_of_overlap = 0.0
        if (width_of_overlap_area < 0.0 or height_of_overlap_area < 0.0):
            area_of_overlap = 0.0
        else:
            area_of_overlap = width_of_overlap_area * height_of_overlap_area
        box_1_area = (box_1.ymax - box_1.ymin) * (box_1.xmax - box_1.xmin)
        box_2_area = (box_2.ymax - box_2.ymin) * (box_2.xmax - box_2.xmin)
        area_of_union = box_1_area + box_2_area - area_of_overlap
        retval = 0.0
        if area_of_union <= 0.0:
            retval = 0.0
        else:
            retval = (area_of_overlap / area_of_union)
        return retval

    # objects = ParseYOLOV3Output(output, new_h, new_w, camera_height, camera_width, 0.4, objects)
    def ParseYOLOV3Output(self,blob, resized_im_h, resized_im_w, original_im_h, original_im_w, threshold, objects, anchors):
        # print("BLOB SHAPE:", blob.shape)
        out_blob_h = blob.shape[2]
        out_blob_w = blob.shape[3]

        side = out_blob_h
        anchor_offset = 0

        if len(anchors) == 18:  ## YoloV3
            if side == self.yolo_scale_13:
                anchor_offset = 2 * 6
            elif side == self.yolo_scale_26:
                anchor_offset = 2 * 3
            elif side == self.yolo_scale_52:
                anchor_offset = 2 * 0

        elif len(anchors) == 12:  ## tiny-YoloV3
            if side == self.yolo_scale_13:
                anchor_offset = 2 * 3
            elif side == self.yolo_scale_26:
                anchor_offset = 2 * 0

        else:  ## ???
            if side == self.yolo_scale_13:
                anchor_offset = 2 * 6
            elif side == self.yolo_scale_26:
                anchor_offset = 2 * 3
            elif side == self.yolo_scale_52:
                anchor_offset = 2 * 0

        side_square = side * side
        output_blob = blob.flatten()

        for i in range(side_square):
            row = int(i / side)
            col = int(i % side)
            for n in range(self.num):
                obj_index = EntryIndex(side, self.coords, self.classes, n * side * side + i, self.coords)
                box_index = EntryIndex(side, self.coords, self.classes, n * side * side + i, 0)
                scale = output_blob[obj_index]
                if (scale < threshold):
                    continue
                x = (col + output_blob[box_index + 0 * side_square]) / side * resized_im_w
                y = (row + output_blob[box_index + 1 * side_square]) / side * resized_im_h
                height = math.exp(output_blob[box_index + 3 * side_square]) * anchors[anchor_offset + 2 * n + 1]
                width = math.exp(output_blob[box_index + 2 * side_square]) * anchors[anchor_offset + 2 * n]
                for j in range(self.classes):
                    class_index = EntryIndex(side, self.coords, self.classes, n * side_square + i, self.coords + 1 + j)
                    prob = scale * output_blob[class_index]
                    if prob < threshold:
                        continue
                    obj = DetectionObject(x, y, height, width, j, prob, (original_im_h / resized_im_h),
                                          (original_im_w / resized_im_w))
                    objects.append(obj)
        return objects

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
        def preprocessing(input_image):
            height, width = self.input_shape[2:]
            image = np.copy(input_image)
            image = cv2.resize(image, (width, height),interpolation=cv2.INTER_CUBIC)
            canvas = np.full((self.m_input_size, self.m_input_size, 3), 128)
            canvas[(self.m_input_size - height) // 2:(self.m_input_size - height) // 2 + height,
            (self.m_input_size - width) // 2:(self.m_input_size - width) // 2 + width, :] = image
            prepimg = canvas
            prepimg = prepimg[np.newaxis, :, :, :]  # Batch size axis add
            prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW
            return prepimg

        def perform_inference(input_image):
            return self.exec_net.infer(inputs={self.input_blob: preprocessing(frame)})

        def parse_outputs(outputs, frame, dets_confidence):
            new_h, new_w = self.input_shape[2:]
            h, w = frame.shape[:2]
            objects = []
            acc = []
            for output in outputs.values():
                objects = self.ParseYOLOV3Output(output, new_h, new_w, h, w, 0.4, objects, self.anchors_selected)
            # Filtering overlapping boxes
            objlen = len(objects)
            for i in range(objlen):
                if (objects[i].confidence == 0.0):
                    continue
                for j in range(i + 1, objlen):
                    if (self.IntersectionOverUnion(objects[i], objects[j]) >= dets_confidence):
                        if objects[i].confidence < objects[j].confidence:
                            objects[i], objects[j] = objects[j], objects[i]
                        objects[j].confidence = 0.0
            detections = []
            for obj in objects:
                detections.append([obj.xmin, obj.ymin, obj.xmax, obj.ymax,obj.confidence,obj.class_id])
            return detections


        detections = parse_outputs(perform_inference(frame),frame,dets_confidence)
        return detections

    def execute_net_asyn(self, frame, request_id):
        def preprocessing(input_image):
            height, width = self.input_shape[2:]
            image = np.copy(input_image)
            image = cv2.resize(image, (width, height),interpolation=cv2.INTER_CUBIC)
            canvas = np.full((self.m_input_size, self.m_input_size, 3), 128)
            canvas[(self.m_input_size - height) // 2:(self.m_input_size - height) // 2 + height,
            (self.m_input_size - width) // 2:(self.m_input_size - width) // 2 + width, :] = image
            prepimg = canvas
            prepimg = prepimg[np.newaxis, :, :, :]  # Batch size axis add
            prepimg = prepimg.transpose((0, 3, 1, 2))  # NHWC to NCHW
            return prepimg

        self.exec_net.start_async(request_id=request_id, inputs={self.input_blob: preprocessing(frame)})

    def wait(self, request_id):
        while True:
            status = self.exec_net.requests[request_id].wait(-1)
            if status == 0:
                break

    def get_output(self, request_id):
        outputs = self.exec_net.requests[request_id].outputs
        return outputs

    def parse_outputs(self, outputs, frame, dets_confidence):
        new_h, new_w = self.input_shape[2:]
        h,w = frame.shape[:2]
        objects = []
        acc = []
        for output in outputs.values():
            objects = self.ParseYOLOV3Output(output, new_h, new_w, h, w, 0.4, objects, self.anchors_selected)
        # Filtering overlapping boxes
        objlen = len(objects)
        for i in range(objlen):
            if (objects[i].confidence == 0.0):
                continue
            for j in range(i + 1, objlen):
                if (self.IntersectionOverUnion(objects[i], objects[j]) >= dets_confidence):
                    if objects[i].confidence < objects[j].confidence:
                        objects[i], objects[j] = objects[j], objects[i]
                    objects[j].confidence = 0.0

        # Drawing boxes
        for obj in objects:
            if obj.confidence < dets_confidence:
                continue
            label = obj.class_id
            confidence = obj.confidence
            acc.append(confidence)
            # if confidence >= 0.2:
            label_text = self.LABELS[label] + " (" + "{:.1f}".format(confidence * 100) + "%)"
            cv2.rectangle(frame, (obj.xmin, obj.ymin), (obj.xmax, obj.ymax), self.box_color, self.box_thickness)
            cv2.putText(frame, label_text, (obj.xmin, obj.ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.label_text_color, 1)
        # return
        return frame,[]
