from inferences.inference_pdr_0013 import Network as NetworkIntel
from inferences.inference_ssd_mobilenet_v2 import Network as NetworkSMB2
from inferences.inference_yolo import Network as NetworkYolo
from inferences.inference_faster_rcnn_inception_v2 import Network as NetworkFasterRCNN


intel       = "models_files/intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml"
smb1        = "models_files/ssd_mobilenet_v1/converted/onnx_model.xml"
smb2        = "models_files/ssd_mobilenet_v2/converted/frozen_inference_graph.xml"
yolov3      = "models_files/yolov3/converted/frozen_darknet_yolov3_model_full.xml"
yolov3tiny  = "models_files/yolov3/converted/frozen_darknet_yolov3_model_tiny.xml"
fasterrcnn  = "models_files/faster_rcnn_inception_v2_coco/converted/frozen_inference_graph.xml"

def get_network(modelType,extension,device):
    if modelType=='intel':
        return NetworkIntel(intel,extension,device)
    elif modelType=='smb2':
        return NetworkSMB2(smb2,extension,device)
    elif modelType=='yolov3':
        return NetworkYolo(yolov3,extension,device)
    elif modelType=='yolov3tiny':
        return NetworkYolo(yolov3tiny,extension,device)
    elif modelType=='fasterrcnn':
        return NetworkFasterRCNN(fasterrcnn,extension,device)
    elif modelType=='smb1':
        print('Model get outputs that are not recognized')
    else:
        print('model_type not recognized. Please check arguments.')