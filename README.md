# Project Write-Up

This is the project write-up to introduce my experiences in the development of a people counter
using openvino.

In investigating potential people counter models, I tried these models:

###### Note:Links for models are in the link of each model or in the steps to convert to IR model

- Model 1/2: [Yolov3/TinyYolov3](https://docs.openvinotoolkit.org/2020.1/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_YOLO_From_Tensorflow.html)
  - I converted the models to an Intermediate Representation with the following lines: 
        
        ```
        mkdir -p ./models_files/yolov3
        cd models_files/yolov3
        mkdir original
        mkdir converted
        cd original
        git clone https://github.com/mystic123/tensorflow-yolo-v3.git
        cd tensorflow-yolo-v3/
        wget https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names
        wget https://pjreddie.com/media/files/yolov3.weights
        wget https://pjreddie.com/media/files/yolov3-tiny.weights
        python3 convert_weights_pb.py --class_names coco.names --data_format NHWC --weights_file yolov3.weights
        mv ./frozen_darknet_yolov3_model.pb ./frozen_darknet_yolov3_model_full.pb
        python3 convert_weights_pb.py --class_names coco.names --data_format NHWC --weights_file yolov3-tiny.weights --tiny
        mv ./frozen_darknet_yolov3_model.pb ./frozen_darknet_yolov3_model_tiny.pb
        MO_ROOT=/opt/intel/openvino/deployment_tools/model_optimizer
        python3 $MO_ROOT/mo_tf.py --input_model frozen_darknet_yolov3_model_full.pb --tensorflow_use_custom_operations_config $MO_ROOT/extensions/front/tf/yolo_v3.json --batch 1
        python3 $MO_ROOT/mo_tf.py --input_model frozen_darknet_yolov3_model_tiny.pb --tensorflow_use_custom_operations_config $MO_ROOT/extensions/front/tf/yolo_v3_tiny.json --batch 1
        mv frozen_darknet_yolov3_model_full.* ./../../converted/
        mv frozen_darknet_yolov3_model_tiny.* ./../../converted/
        ```

  - The models were insufficient for the app because:
        
        - tiny yolo v3: recall insuficient enough
        - yolo v3: inference time too high
    
    ![tiny_yolo_v3_records]     
    
    ![yolo_v3_records]
    
    ###### you can replicate values. check "how to test" section.
    
    
  - I tried to improve the model USAGE with the Asynchronous inferences. Speeds increases 
  but not enough (this for Yolov3).
  
- Model 3: [SSD_MobileNet_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz)
  - I converted the model to an Intermediate Representation with the following lines:
      ```
        mkdir -p ./models_files/ssd_mobilenet_v2
        cd models_files/ssd_mobilenet_v2
        mkdir original
        mkdir converted
        cd original
        wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz
        tar -xvf ssd_mobilenet_v2_coco_2018_03_29.tar.gz
        cd ssd_mobilenet_v2_coco_2018_03_29/
        MO_ROOT=/opt/intel/openvino/deployment_tools/model_optimizer
        python3 $MO_ROOT/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config $MO_ROOT/extensions/front/tf/ssd_v2_support.json
        mv frozen_inference_graph.* ./../../converted/
      ```
  - The model was insufficient for the app because recall was too low:
  
    ![ssd_mobilenet_records]

    ###### you can replicate values. check "how to test" section.
    
    
- Model 4: [SSD_Mobilenet_v1](https://github.com/qfgaohao/pytorch-ssd)
  - I converted the model to an Intermediate Representation with the following lines:
      ```
        mkdir -p models_files/ssd_mobilenet_v1
        cd models_files/ssd_mobilenet_v1/
        mkdir original
        mkdir converted
        cd original
        git clone https://github.com/qfgaohao/pytorch-ssd  #obviously
        cd pytorch-ssd
        wget -P models https://storage.googleapis.com/models-hao/mobilenet-v1-ssd-mp-0_675.pth
        wget -P models https://storage.googleapis.com/models-hao/voc-model-labels.txt
      ```
        
       since the model is pytorch, lets frist save it as .onnx 
        
      ```
        nano run_ssd_live_demo.py
        # add 5 lines after line 41 of the file:
        41   net.load(model_path)
        42   import torch
        43   dummy_input = torch.randn(1, 3, 300, 300)
        44   torch.onnx.export(net, dummy_input, "onnx_model.onnx")
        45   print('model saved as .onnx')
        46   exit()
      ```
      
      then run the demo. If no camera you can add a video path at the end.
      
      ```
        python3 run_ssd_live_demo.py mb1-ssd models/mobilenet-v1-ssd-mp-0_675.pth models/voc-model-labels.txt
      ```
    
       Model is now as .onnx file, lets convert it to IR model.
    
      ```
        MO_ROOT=/opt/intel/openvino/deployment_tools/model_optimizer    
        python3 $MO_ROOT/mo.py --input_model onnx_model.onnx    
            # Execution - DIRs changed to keep example
            Model Optimizer arguments:
            Common parameters:
                - Path to the Input Model: 	<DIR>/pytorch-ssd/onnx_model.onnx
                - Path for generated IR: 	<DIR>/pytorch-ssd/.
                - IR output name: 	onnx_model
                - Log level: 	ERROR
                - Batch: 	Not specified, inherited from the model
                - Input layers: 	Not specified, inherited from the model
                - Output layers: 	Not specified, inherited from the model
                - Input shapes: 	Not specified, inherited from the model
                - Mean values: 	Not specified
                - Scale values: 	Not specified
                - Scale factor: 	Not specified
                - Precision of IR: 	FP32
                - Enable fusing: 	True
                - Enable grouped convolutions fusing: 	True
                - Move mean values to preprocess section: 	False
                - Reverse input channels: 	False
            ONNX specific parameters:
            Model Optimizer version: 	2020.1.0-61-gd349c3ba4a
            
            [ SUCCESS ] Generated IR version 10 model.
            [ SUCCESS ] XML file: <DIR>/pytorch-ssd/./onnx_model.xml
            [ SUCCESS ] BIN file: <DIR>/pytorch-ssd/./onnx_model.bin
            [ SUCCESS ] Total execution time: 9.11 seconds. 
            [ SUCCESS ] Memory consumed: 215 MB.
        mv onnx_model.* ./../../converted/
      ```    
               
  - The model was insufficient for the app because there were no way to extract 
  "detections" from the outputs. Outputs keys: 
      ```
        dict_keys(['Concat_220', 'Softmax_195'])
      ```
  - Model were no able for further improvements.

- Model 5: [Faster R-CNN inception v2 COCO](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz)
   - I converted the model to an Intermediate Representation with the following lines:
      ```
        mkdir -p ./models_files/faster_rcnn_inception_v2_coco
        cd models_files/faster_rcnn_inception_v2_coco
        wget http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz
        mkdir original 
        mkdir converted
        tar -xvf faster_rcnn_inception_v2_coco_2018_01_28.tar.gz -C ./original
        cd original/faster_rcnn_inception_v2_coco_2018_01_28/
        MO_ROOT=/opt/intel/openvino_2020.1.023/deployment_tools/model_optimizer
        python3 $MO_ROOT/mo_tf.py --input_model frozen_inference_graph.pb --tensorflow_object_detection_api_pipeline_config pipeline.config --reverse_input_channels --tensorflow_use_custom_operations_config $MO_ROOT/extensions/front/tf/faster_rcnn_support.json --input_shape [1,600,600,3]
        mv frozen_inference_graph.* ./../../converted/
      ```
  - The models were insufficient for the app because inference time is too high:
  
    ![faster_rcnn_records]
  
    ###### you can replicate values. check "how to test" section.
    
    
  - I tried to improve the model USAGE with the Asynchronous inferences. Speeds increases 
  but not enough.

- Model 6: person-detection-retail-0013:

    - This model was selected at the end to keep going with the project. How to download model:
    
    ```
        python3 /opt/intel/openvino/deployment_tools/open_model_zoo/tools/downloader/downloader.py --name person-detection-retail-0013 -o ./models_files/
    ```
  
    ###### Note: values of testing model are described at "Solution for People Counter" Section.
    
    
# Solution for People Counter

The solution is related to the Trackiong by detections method.

This method's pipeline is about 2 big boxes. Detections and Tracking.

Due to previous section, I chose the model "person_detection_retail_0013" from the intel zoo 
model for the detection Box. This model shows great accuracy (90%+), good recall (70%+, aprox 
30% of the images do not have objects to detect), and excellent inference time (19ms).

![person_detection_0013_records]

For the tracking Box, I chose the SORT algorithm. This Algorithm is fed with the detections
and solve a linear assigment problem to assign detections of frame N+1 to tracks done until
frame N. This iteration gives a basic but powerfull object tracking.
 

The tracks (objects being tracked) are analized after they die (this means the object was
unsuccessfully reassigned during the tracking due to normal computer vision problems; or 
they are out of the visual scene).

![tracking_by_detection]

## How to test

There is a python file "test.py"

This file is used to get information about a full video iteration

```
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
```

You can use the following parse to test each of the models. 

```
python3 test.py -m "intel"
```

This file will print frame by frame a counter that increase by 1 each time. At the end it shows the
parameters mentioned before.

###### Notice: ssd_mobilinet_v1 is not working due to problems presented before.

## How to use

##### Instal Openvino

Check openvino documentation.

##### Install the following packages:

```
pip3 install -r resources/requirements.txt
```

##### Start servers and main:

```
# run MQTT server node (terminal_1)
cd webservice/server/node-server
node ./server.js

# run UI (terminal_2)
cd webservice/ui
npm run dev

# run ffmpeg server (terminal_3)
sudo ffserver -f ./ffmpeg/server.conf

# open browser at http://0.0.0.0:3000/

# run project (terminal_4)
python3 main.py -m 'intel' -i 'resources/Pedestrian_Detect_2_1_1.mp4' -pt 0.4 | ffmpeg -v warning -f rawvideo -pixel_format bgr24 -video_size 768x432 -framerate 24 -i - http://0.0.0.0:3004/fac.ffm

# Check project in the browser at 0.0.0.0:3000
```

## Results

- People in Frame change everything that is a change in the number of tracks tracked.
- Average Duration is sent to update everything there a person goes out of the scene. the 
method for obtain time is: 

```
Number of frames track was alive / (video FPS * number of people counted)
```

![results]



## Assess Model Use Cases

Since Covid-19 forces us to keep social distancing some use cases for people counter could be:

- Number of person entering to a closed zone like bathrooms and showing the maximum of person
 so they can wait for someone to come out.
- Alert when a capacity of people is overpassed for a scene (covered by the camera and 
parametrized).
- Alerts in places where people are not able to be during quarantine time.
- Understand people traffic in street.



## Assess Effects on End User Needs

Some variable to consider when you have a model to deploy:

Lighting: Light in scene is crucial for detections. A big amount of light in a
2D scene may confuse the model, and a low amount may give no detections.

Model Accuracy: Essential for take decision about which model is better for a determinate
use case.

Image size: A bigger image (more resolution) will have better pattern to detect, but this also
makes the network to be slower since it has to go through all the pixels.

Model input size: While this is faster if short, it also needs the input to resize and the 
patterns can be lost.

FPS: While this is visually better if the FPS is high, this can make the results slow
due to the detections bottleneck. Consider use video sampling.

Streaming method: Some project may be easy with a deployment with a local camera. But if not,
like a CCTV camera environment, you may need integrate for make streaming of the video frames 
to make them you input stream.





[ssd_mobilenet_records]: images/ssd_mobilenet_v2_coco.png
[tiny_yolo_v3_records]: images/tiny_yolo_v3.png
[yolo_v3_records]: images/Yolo_v3.png
[faster_rcnn_records]: images/faster_rcnn_inception_v2.png
[person_detection_0013_records]: images/person_detection_retail_0013.png
[tracking_by_detection]: images/pipeline.png
[results]: images/results.png


