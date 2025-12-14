import os
import cv2
import torch
import numpy as np
import json
import mmcv
import gc
import argparse
import pickle
from mmdet.registry import VISUALIZERS
from mmengine.registry import init_default_scope
from mmdet.apis import inference_detector, init_detector
from multiprocessing import Pool, cpu_count
from functools import partial
from mmcv.image import imread
from mmcv.transforms import Compose
from tqdm import tqdm
#https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmdet_nano_8xb32-100e_coco-obj365-person-05d8511e.pth
#rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth
bt_size_det = 30

det_config = 'demo/mmdetection_cfg/rtmdet_m_640-8xb32_coco-person.py'
det_checkpoint = 'rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth'

# process Det
def preprocess_det(img, test_pipeline,img_id):

    data_ = dict(img = img,
                 img_id = img_id)
    data_ = test_pipeline(data_)
    return data_

# do det
def det_frames(video_path,det_model,output,test_pipeline):
    #compose batch for det
    det_inputs_list =[]
    det_data_sample_list =[]
    det_data_img_id = []
    # store result
    results_list = []

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file:{video_path}")
        return None #if None, continue
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    index = 0

    print('Start reading video and preprocessing......\n')
    with tqdm(total=total_frames, desc="Processing Video") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            data_ = preprocess_det(frame,test_pipeline,index)
            det_inputs_list.append(data_['inputs'])
            det_data_sample_list.append(data_['data_samples'])
            det_data_img_id.append(data_['data_samples'].img_id)

            # Check if batch size is reached or it is the last element
            if (index + 1) % bt_size_det == 0 or (index + 1) == total_frames:
                batch_data =  {'inputs': det_inputs_list, 'data_samples': det_data_sample_list, 'img_id':det_data_img_id}
                # detection in batch
                results = det_model.test_step(batch_data)
                # import pdb
                # pdb.set_trace()
                results_list.extend(results)

                det_inputs_list =[]
                det_data_sample_list =[]
                det_data_img_id = []
            
            index += 1
            pbar.update(1)

    print(f'Save to pickle file: {output}\n')
    with open(output, 'wb') as f: 
        pickle.dump(results_list, f)
    cap.release()
    return True


def func(video_path, output,test_pipeline,det_model,txt_path):

    if(det_frames(video_path, det_model,output,test_pipeline)):
        print(f'success prcocessed {video_path}')
        if not os.path.isfile(txt_path):
            with open(txt_path, 'w') as f:
                f.write(output + '\n')
        else:
            with open(txt_path, 'a') as f:
                f.write(output + '\n')


def process_videos_in_directory(base_directory, output_directory,device,start,end,txt_path):
    #init det model
    det_model = init_detector(det_config, det_checkpoint, device=device)

    # build test pipeline
    det_model.cfg.test_dataloader.dataset.pipeline[
        0].type = 'mmdet.LoadImageFromNDArray'
    test_pipeline = Compose(det_model.cfg.test_dataloader.dataset.pipeline)
    print(len(sorted(os.walk(base_directory))))
    #reading video file
    for root, dirs, files in tqdm(sorted(os.walk(base_directory))[start:end]):
        for file in files:
            if file.endswith((".mp4", ".avi", ".mkv", ".mov")):  # Add more video formats based on actual situation
                video_path = os.path.join(root, file)
                output_file_name = os.path.splitext(file)[0] + "_det.pickle"
                output_path = os.path.join(output_directory, output_file_name)
                if  os.path.exists(output_path):
                    print(f'The {output_path} file exist')
                    continue
                func(video_path, output_path,test_pipeline,det_model,txt_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('start', type=int, default=0,help='Start index')
    parser.add_argument('end', type=int, default=-1,help='End index')
    parser.add_argument('--device', type=str, default='cuda:0', help='CUDA device to use (default: cuda:0)')
    args = parser.parse_args()
    device = args.device
    start = args.start
    end = args.end

    base_directory = "YOUR_BASE_DIRECTORY/1_scene/cliped_videos"
    output_directory = "YOUR_OUTPUT_DIRECTORY/2_det"
    txt_path = "YOUR_TXT_PATH/det_success.txt"