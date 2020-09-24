# 快速风格迁移及其在Intel openVINO平台上的推理

###### *作者：吴一凡*  

## 关于模型的训练及存储  

### 模型的训练

* 首先使用Jupyter编辑器打开train_model.ipynb

* 在第三个column中调整训练参数，以下是我的训练参数：

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


* 运行train_model.ipynb，除监控log之外还可以观察每个check period的输出图片（位于save path下）

* 该过程将持续2小时以上，建议借助KML等训练平台的帮助

* 最终的ckpt文件将在训练完成后被保存在save path下

### 使用训练好的模组

* 使用Jupyter编辑器打开make_prediction.ipynb（最好关闭train_model的进程，否则可能出现内存溢出的情况）

* 修改CONTENT_NAME, CONTENT_IMAGE, MODEL_PATH（刚才存储的ckpt）, OUTPUT_PATH即可进行预测

* 如需使用openVINO推理机进行推理需要uncomment 27, 28行的代码，定义save_path并重新存储针对content图片的单batch ckpt

### 冻结模型

* 使用openVINO推理机需要首先生成冻结的模型（.pb），首先打开freeze_graph.ipynb

* 修改model_dir为刚才存储的prediction ckpt

* 此处需要自定义output node。如需锁定output node，可以在运行make prediction时uncomment 38行存储log的代码，并使用tensorboard --logdir logs/加载tensorboard查看从何处输出（此处我使用'clip_by_value'作为输出层）

* 最后出现 xxx ops in the final graph. 字样，模型冻结成功，冻结完的模型frozen_model.pb存储在相同的文档下  

## 使用openVINO推理模型

### 生成IR模型

* 首先你需要在根目录安装openVINO库并使用source /opt/intel/openvino/bin/setupvar.sh  

* 进入/opt/intel/openvino/deployment_tools/model_optimizer，并安装支持组件sh install_prerequisites/install_prerequisites.sh  

* 使用model optimizer tensorflow支持转换模组为IR模组：sudo python3 mo_tf.py --input_model {MODEL_DIR/frozen_model.pb} --input_shape [4,512,512,3] --output_dir {OUTPUT_DIR} （注意此处需要明确输入图形[4,512,512,3],此处我是用的是batch=4运行prediction模型）  

* 生成的IR模型分为.xml, .mapping, .bin三部分，被存储于OUTPUT_DIR中  

### 使用模型进行推理

* 首先修改openVINO inference.py中的数据

>model_xml = "frozen_model.xml"  
>input_pic = "../content/"  
>output_dir = "output/"  
>mean_val_r, mean_val_g, mean_val_b = (0, 0, 0)  

* 然后运行python3 openVINO inference.py CPU得到推理后的生成的图片，存储在output_dir中

* 此处可以将CPU更改为HETERO:FPGA,CPU进行异构推理，注意推理前需要使用aocl program acl0 2020-3_RC_FP16_InceptionV1_SqueezeNet_TinyYolo_VGG.aocx指令加载bitstreams（bitstream存储于/opt/intel/openvino/bitstreams/）

  
----  

###### 关于FPGA推理的bug整合：

1. 现在运行多batch时会出现batch中图像强制压缩的问题，导致batch图像混色

2. 

