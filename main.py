"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
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
import os
import sys
import socket
import cv2

from argparse import ArgumentParser
import paho.mqtt.client as mqtt
from tracking.tracker import Tracker


from inferences.inferences import get_network


def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    BASE_DIR = "/".join(os.path.abspath(__file__).split('/')[:-1])+'/'
    parser = ArgumentParser()
    parser.add_argument("-m", "--model_type", required=True, type=str,default='intel',
                        help="Path to an xml file with a trained model. Posibilities: smb1 smb2 yolov3 yolov3tiny fasterrcnn intel")
    parser.add_argument("-i", "--input", required=True, type=str, default=BASE_DIR + 'resources/Pedestrian_Detect_2_1_1.mp4',
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str, default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, required=False, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", required=False, type=float, default=0.4,
                        help="Probability threshold for detections filtering"
                        "(0.4 by default)")
    return parser

def get_input_stream_data(input_stream,is_image):
    ### TODO: Read from the video capture ###
    if is_image:
        frame = input_stream
        flag = False
    else:
        _, frame = input_stream.read()
        if not _:
            flag = False
        else:
            flag = True
    return frame,flag

def infer_on_stream(args, client):
    # time_a = dt.now()
    # Initialise the class

    infer_network = get_network(args.model_type,args.cpu_extension,args.device)
    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold
    # Load the model through `infer_network`
    infer_network.load_model()
    # Handle the input stream
    input_stream = args.input
    if input_stream.endswith(".jpg") or input_stream.endswith(".png"):
        is_image = True
        input_stream = cv2.imread(input_stream)
    elif input_stream.endswith(".mp4") or input_stream.endswith(".avi"):
        is_image = False
        input_stream = cv2.VideoCapture(input_stream)
        FPS = input_stream.get(cv2.CAP_PROP_FPS)
    else:
        print("input extesion not available...")
        exit()
    frame_count = 0
    # Get first frame (or the image)
    frame,flag = get_input_stream_data(input_stream,is_image)
    if flag:
        frame_count+=1
    # initialize parameters for the tracker
    h,w = frame.shape[:2]
    MatrixOPeration = "DIST"# "IOU" #
    THRESH = int(min(h, w) / 5) if MatrixOPeration == "DIST" else  0.5
    AssigProblemSolver = "lapsolver_solvedense"
    #initialize tracker
    if is_image:
        tracker = Tracker(THRESH, MatrixOPeration, AssigProblemSolver)
    else:
        tracker = Tracker(THRESH, MatrixOPeration, AssigProblemSolver,FPS = FPS)

    # First Inference note: to make async async.
    infer_network.execute_net_asyn(frame, 0)
    while flag:
        # Async handled as:
        #     Each frame is sent to inference after
        #     the outputs of the last inference were
        #     obtained (get_output())
        #     After this frame is sent, outputs are managed,
        #     this means, parsed (filters, shapes, etc) and sent to
        #     sort tracking algorithm (selected for making the tracking
        #     as this is need for a high level of what is happening in
        #     the input.
        last_frame = frame
        frame, flag = get_input_stream_data(input_stream, is_image)
        if flag:
            frame_count += 1
        else:
            continue

        infer_network.wait(0)
        output = infer_network.get_output(0)
        infer_network.execute_net_asyn(frame, 0)
        detections_parsed = infer_network.parse_outputs(output,last_frame,prob_threshold)
        # track_dets manage the messages to the client itself.
        image_output_tracked = tracker.track_dets(detections_parsed,last_frame,client)
        # send images to the ffmpeg server
        # cv2.imshow("win",image_output_tracked)
        # cv2.waitKey(1)
        sys.stdout.buffer.write(image_output_tracked)
        sys.stdout.flush()
    # cv2.destroyAllWindows()
    # print(frame_count)


def connect_mqtt():
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT,MQTT_KEEPALIVE_INTERVAL)
    return client

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60


def main():
    # BASE_DIR = "/".join(os.path.abspath(__file__).split('/')[:-1])+'/'
    args = build_argparser().parse_args()
    # args = {'base_dir'      :BASE_DIR,
    #         'input_stream'  :BASE_DIR + 'resources/Pedestrian_Detect_2_1_1.mp4',
    #         'probability'   :0.4,
    #         }
    client = connect_mqtt()
    infer_on_stream(args, client)

if __name__ == '__main__':
    main()
