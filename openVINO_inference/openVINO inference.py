from __future__ import print_function
import sys
import os
import cv2
import numpy as np
import logging as log
from PIL import Image, ImageOps
import time


sys.path.append('/opt/intel/openvino/python/python3.7/')
from openvino.inference_engine import IECore


def main():

    model_xml = "frozen_model.xml"
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    #cpu_extension = 
    device = [sys.argv[1]]
    input_dir = "../content/"
    output_dir = "output/"
    mean_val_r, mean_val_g, mean_val_b = (0, 0, 0)
    input_list = os.listdir(input_dir)
    input_num = len(input_list)
    starting_time = time.time()

    print("---------------------------Creating Inference Engine---------------------------")
    ie = IECore()

    print("---------------------------Loading Network Files-------------------------------")

    net = ie.read_network(model=model_xml, weights=model_bin)

    if "CPU" in device:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            print("[--!!!--]Following layers are not supported by the plugin for specified device {}:\n {}".format(device, ', '.join(not_supported_layers)))
            print("[--!!!--]Please try to specify cpu extensions library path in sample's command line parameters using -l")
            sys.exit(1)

    print("---------------------------Preparing Input Blobs-------------------------------")

    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    net.batch_size = 4

    n, c, h, w = net.inputs[input_blob].shape
    #print(n,h,w,c)
    #image = np.ndarray(shape=(c, h, w))
    
    for epoch in range(len(input_list) // n):
        image_batch = np.zeros((n, c, h, w),np.float32)
        for index, input_pic in enumerate(input_list[epoch*n:(epoch+1)*n]):
            print(input_pic)
            image = Image.open(input_dir+input_pic)	
            #if image.shape[-1] != (h, w):
            print("Input image is resized to (512, 512)")
            #image = cv2.resize(image, (w, h))
            image = ImageOps.fit(image, (w, h), Image.ANTIALIAS)
            
            image = np.asarray(image, np.float32)
            image = np.swapaxes(image,0,2)
            image_batch[index] = image
        #image_batch = np.swapaxes(image_batch, 1, 3)
        #print(image_batch.shape)
        #print("Batch size is {}".format(n))

        print("---------------------------Loading Model to Plugin------------------------------")
        exec_net = ie.load_network(network=net, device_name=device[0])


        print("---------------------------Start Sync Inference---------------------------------")
        res = exec_net.infer(inputs={input_blob: image_batch})


        print("---------------------------Processing Output Blob-------------------------------")
        data = res[out_blob]

        print("---------------------------Post Process Output----------------------------------")

        data = np.swapaxes(data, 1, 3)
        #data = np.swapaxes(data, 0, 1)
        print("Output shape: ", data.shape)
        for i, out_img in enumerate(data):
            #print(out_img.shape)
            out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2RGB)
            out_img[out_img < 0] = 0
            out_img[out_img > 255] = 255
            #out_img = out_img[::] - (mean_val_r, mean_val_g, mean_val_b)
            out_path = os.path.join(os.path.dirname(__file__), "output/out_{}.jpg".format(epoch*n+i))
            cv2.imwrite(out_path, out_img)

            print("Result image was saved to {}".format(out_path))
   
    ending_time = time.time()
    latency = ending_time - starting_time
    print("Latency: "+str(latency*1000)+" ms")
    print("Throughput: "+str(input_num / latency)+" fps") 
if __name__ == '__main__':
    sys.exit(main() or 0)
