import os
import cv2
from datetime import datetime as dt
from argparse import ArgumentParser



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
    parser.add_argument("-i", "--input", required=False, type=str, default=BASE_DIR + 'resources/Pedestrian_Detect_2_1_1.mp4',
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

def infer_on_stream(args):

    infer_network = get_network(args.model_type,args.cpu_extension,args.device)


    prob_threshold = args.prob_threshold
    infer_network.load_model()

    input_stream = args.input
    if input_stream.endswith(".jpg") or input_stream.endswith(".png"):
        is_image = True
        input_stream = cv2.imread(input_stream)
    elif input_stream.endswith(".mp4") or input_stream.endswith(".avi"):
        is_image = False
        input_stream = cv2.VideoCapture(input_stream)
    else:
        print("input extesion not available...")
        exit()

    frame_count ={'with_det':0,'without_det':0}
    acc = 0
    flag = True
    inicio = dt.now()
    while flag:

        frame, flag = get_input_stream_data(input_stream, is_image)
        if not flag:
            break
        detections = infer_network.execute_net(frame,prob_threshold)
        if len(detections) == 0:
            frame_count['without_det']+=1
        else:
            frame_count['with_det']+=1
            for det in detections:
                acc += det[4]
        print("frame_count = ", frame_count['with_det']+frame_count['without_det'])
    fin = dt.now()
    Total_time_seconds = (fin-inicio).total_seconds()
    Total_frames = frame_count['with_det']+frame_count['without_det']
    video_seconds = 139
    print("""
    ###########################################################################################
    Model Name: {}
    Video total number of frame: {}
    Video seconds: {}
    Video FPS: {}
    VideoInference total seconds taken: {}
    Average Inference time: {}
    Number of frames with detections: {}
    Recall: {}
    Accuracy: {}
    ###########################################################################################
    """.format(infer_network.get_network_name(),
               Total_frames,
               video_seconds,
               Total_frames/video_seconds,
               Total_time_seconds,
               Total_time_seconds/Total_frames,
               frame_count['with_det'],
               frame_count['with_det']/Total_frames,
               acc /frame_count['with_det'],
               )

          )




def main():
    args = build_argparser().parse_args()
    infer_on_stream(args)

if __name__ == '__main__':
    main()
