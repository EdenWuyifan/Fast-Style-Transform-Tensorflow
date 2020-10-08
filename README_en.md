# Fast Style Transformation Model and Its Inference on Intel openVINO

###### *author：Eden Yifan Wu*  

## About Model Training and Storing  

### Model Training

* Firstly using Jupyter notebook to open train_model.ipynb. 

* Edit training parameters under the third column, following are the status that I used for training my model: 

>content directory : "content/"  
>content layer : ['relu4_2']  
>style image : "style/Shinagawa.jpg" (可更改)  
>style layers : ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']  
>style layer weights : [.05, .1, .15, .3, .4]  
>number of epoches : 1000  
>batch size : 1  
>content weight : 10  
>style weight : 1000  
>total variable weight : 200  
>learning rate : 1e-4  
>save path : "models/"  
>check period : 500  
>test image : "content/0801.png"  


* Run train_model.ipynb. Apart from monitering log, trainers can access the output images for each check period (under save_path). 

* The process is going to last more than 2 hours, training platforms are highly recommended (donnot use CPU). 

* The final .ckpt file will be saved under save_path. 

### Using Pretrained Model

* Use Jupyter notbook to open make_prediction.ipynb（better close train_model.ipynb process, or there could be a memory overflow. 

* Edit CONTENT_NAME, CONTENT_IMAGE, MODEL_PATH（.ckpt that just stored）, OUTPUT_PATH, then we can start our prediction. 

* If you are going to use openVINO Inference Engine for further inference please uncomment code line 27 and 28, define save_path and restore ckpt (single-batched for testing). 

### Freezing Model

* Use openVINO Inference Engine required first generating frozen model(.pb), to do so you need to open freeze_graph.ipynb. 

* Edit model_dir to the prediction ckpt directory that we just saved. 

* output node need to be defined here. If you want to target an output node, you can uncomment code line 38 when make prediction, and using tensorboard --logdir logs/, which visualize the model to tensorboard (here I use 'clip_by_value' as output layer). 

* When 'xxx ops in the final graph' shows, the model is successfully freezed. The frozen_model.pb is saved under the same directory.  

## Using openVINO to Inference Model

### Generating IR Model

* Firstly you should install openVINO toolkit under root directory. Then initialize the environment using source /opt/intel/openvino/bin/setupvar.sh.  

* Enter /opt/intel/openvino/deployment_tools/model_optimizer, and install prerequisites by running install_prerequisites/install_prerequisites.sh  

* Use model optimizer towards tensorflow convert the frozen model to IR model：sudo python3 mo_tf.py --input_model {MODEL_DIR/frozen_model.pb} --input_shape [4,512,512,3] --output_dir {OUTPUT_DIR} (note here I declear input shape [4,512,512,3] for my freezing prediction model with batch=4).  

* The generated IR model includes .xml, .mapping, and .bin, which are stored under OUTPUT_DIR.  

### Using Model to Infer

* First change parameters in openVINO inference.py: 

>model_xml = "frozen_model.xml"  
>input_pic = "../content/"  
>output_dir = "output/"  
>mean_val_r, mean_val_g, mean_val_b = (0, 0, 0)  

* Then run python3 openVINO inference.py CPU get the image being inferenced, stored under output_dir. 

* Here we can change CPU to HETERO:FPGA,CPU for heterogenous computation，note that we should run aocl program acl0 2020-3_RC_FP16_InceptionV1_SqueezeNet_TinyYolo_VGG.aocx to load bitstreams for Intel a10 FPGA(differs between different chips) before doing inference（bitstream are stored under /opt/intel/openvino/bitstreams/).  

  
----  

###### Bug Reports towards FPGA Inference：

1. Running with multiple batches sometimes caused image color mixing. (Solved: Checking if the same input shapes are used when freezing model and generating IR model). 
