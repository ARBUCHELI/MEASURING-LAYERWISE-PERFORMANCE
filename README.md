# MEASURING-LAYERWISE-PERFORMANCE

## This Content was Created by Intel Edge AI for IoT Developers UDACITY Nanodegree.

Use the get_perf_counts API available in OpenVINO to get the performance of each layer in our model. By identifying the bottlenecks in our models performance, we can then remove,
or change those layers to make our model inference time faster.


## Measuring Layerwise Performance

So far, you have seen the effect that using efficient layers has on the overall inference time of the model. Since the networks we have been working on so far were small, it was easy for us to calculate the FLOPs for each layer and seewhich layers were the bottleneck in our model.

However, in case of larger models, calculating the FLOPs for each model quickly becomes very tedious and difficult. Moreover, just measuring the inference time of the model does not give us much information about which layer might be taking more time to compute in our model.

In this concept, we will use the <code>get_perf_counts</code> API available in OpenVINO to get the performance of each layer in our model. By identifying the bottlenecks in our models performance, we can then remove, or change those layers to make our model inference time faster.

Follow along with the video and try to run the code

## Something important to notice is that the fully connected layer takes the most time to execute, because since the flattened output from the previous convolutional layer is quite large, the total FLOPs in the fully connected layer layer is much more than the convolutional layers. This is why it took more time to execute.

Before running the code, make sure you source the OpenVINO environment.

<pre><code>
import numpy as np
from openvino.inference_engine import IENetwork
from openvino.inference_engine import IEPlugin

import pprint
import os
import cv2
import argparse
import sys

def main(args):
    pp = pprint.PrettyPrinter(indent=4)
    model=args.model
    device=args.device
    image_path=args.image

    # Loading model
    model_weights=model+'.bin'
    model_structure=model+'.xml'
    
    model=IENetwork(model_structure, model_weights)
    plugin = IEPlugin(device=device)

    # Loading network to device
    net = plugin.load(network=model, num_requests=1)

    # Get the name of the input node
    input_name=next(iter(model.inputs))

    # Reading and Preprocessing Image
    input_img=np.load(image_path)
    input_img=input_img.reshape(1, 28, 28)

    # Running Inference
    net.requests[0].infer({input_name:input_img})
    pp.pprint(net.requests[0].get_perf_counts())

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='CPU')
    parser.add_argument('--image', default=None)
    
    args=parser.parse_args()
    sys.exit(main(args) or 0)
</code></pre>

## To run the application please enter the following command in the terminal: python3 perf_counts.py --image image.npy --model base_cnn/base_cnn

## Adaptation as a Repository: Andrés R. Bucheli.
