import os
import cv2
import torch
import numpy as np
import json
import mmcv
import gc
import argparse
import pickle

from mmengine.registry import init_default_scope
from mmpose.apis import init_model as init_pose_estimator
from mmdet.apis import inference_detector, init_detector
from multiprocessing import Pool, cpu_count
from functools import partial
from mmcv.image import imread
from mmengine.dataset import Compose, pseudo_collate
from mmcv.transforms import Compose as Det_Compose
from mmpose.evaluation.functional import nms
from tqdm import tqdm
from mmdet.utils import get_test_pipeline_cfg
from mmpose.utils import adapt_mmdet_pipeline

batch_size = 200

def preprocess_pose(frame, bbox, img_id, pose_test_pipeline,pose_dataset_meta):
    data_info = dict(img=frame,
                     img_id = img_id)
    data_info['bbox'] = bbox[0][:4][None]
    data_info['bbox_score'] = np.array(bbox[0][4:])
    # import pdb
    # pdb.set_trace()
    data_info.update(pose_dataset_meta)
    data =  pose_test_pipeline(data_info)
    return data


def func(pickle_data, mp4_file_path, pose_test_pipeline, pose_dataset_meta):
    # Open video file
    cap = cv2.VideoCapture(mp4_file_path)

    if not cap.isOpened():
        print(f"Error opening video file: {mp4_file_path}")
        return

    # Get total frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Build a dictionary to quickly find bboxes corresponding to img_id
    frame_data = {item['img_id']: item['bboxes'] for item in pickle_data}

    # Build batch data
    batch = []
    all_batches = []
    
    # Read video frames sequentially
    for frame_idx in tqdm(range(total_frames), desc="Reading frames"):
        ret, frame = cap.read()

        if not ret:
            print(f"Failed to read frame {frame_idx} from video {mp4_file_path}")
            break

        # Process if current frame has corresponding bboxes
        if frame_idx in frame_data:
            bboxes = frame_data[frame_idx]
            processed_data = preprocess_pose(frame, bboxes, frame_idx, pose_test_pipeline, pose_dataset_meta)
            
            # Add processed frame to batch
            batch.append(processed_data)

            if len(batch) == batch_size:
                all_batches.append(pseudo_collate(batch))
                batch = []
    
    # Store remaining incomplete batch in all_batches
    if batch:
        all_batches.append(pseudo_collate(batch))
            
    # Release video capture object
    cap.release()
    cv2.destroyAllWindows()

    return all_batches




def find_mp4_file(base_path, filename):
    for root, _, files in os.walk(base_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

def process_files(pickle_path, mp4_base_path,output_path,pose_model, pose_test_pipeline, pose_dataset_meta ,start,end):
    # Get all pickle files in the pickle folder
    pickle_files = [f for f in os.listdir(pickle_path) if f.endswith('.pickle')]
    print(len(pickle_files))
    for pickle_file in tqdm(pickle_files[start:end], desc="Processing files"):
        base_name_json = pickle_file.split('.')[0]
        json_file = f'{base_name_json}.pickle'
        json_file_path = os.path.join(output_path, json_file)
        if os.path.exists(json_file_path):
            print(f'{json_file_path} exist, skip it')
            continue
        # Build full path for pickle file
        pickle_file_path = os.path.join(pickle_path, pickle_file)

        # Load pickle file data
        with open(pickle_file_path, 'rb') as f:
            pickle_data = pickle.load(f)

        # Corresponding mp4 filename
        mp4_file_name = pickle_file.replace('.pickle', '.mp4')

        # Find corresponding mp4 file in mp4_base_path
        mp4_file_path = find_mp4_file(mp4_base_path, mp4_file_name)

        if mp4_file_path:
            # Call func to process pickle data and mp4 file
            process_batch = func(pickle_data, mp4_file_path, pose_test_pipeline, pose_dataset_meta)
        else:
            print(f"MP4 file {mp4_file_name} not found in {mp4_base_path}")
            continue
        # for i, batch_data in enumerate(process_batch):

        pose_results = []
        for i in tqdm(range(len(process_batch)), desc= 'Det human pose ...'):
            with torch.no_grad():
                results = pose_model.test_step(process_batch[i])
                pose_results.append(results)
            torch.cuda.empty_cache()
        
        json_result = {}

        for i in tqdm(range(len(pose_results)), desc='Reading and Saving the Pose and Box ...'):
            for j in range(len(pose_results[i])):
                pred_instance = pose_results[i][j].pred_instances.cpu().numpy()
                img_id = pose_results[i][j].img_id
                base_name = img_id
                new_bboxes = pred_instance.bboxes
                new_bbox_scores = pred_instance.bbox_scores
                new_keypoints = pred_instance.keypoints
                new_keypoint_scores = pred_instance.keypoint_scores
                
                if base_name not in json_result:
                    json_result[base_name] = {
                    'img_index': img_id,
                    'bboxes': new_bboxes.tolist(),
                    'bbox_scores': new_bbox_scores.tolist(),
                    'keypoints': new_keypoints.tolist(),
                    'keypoint_scores': new_keypoint_scores.tolist()
                    }

        # save json file
        with open(json_file_path, 'wb') as file:
            pickle.dump(json_result, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('start', type=int, help='Start index')
    parser.add_argument('end', type=int, help='End index')
    parser.add_argument('--device', type=str, default='cuda:0', help='CUDA device to use (default: cuda:0)')
    args = parser.parse_args()
    

    pickle_path = 'YOUR_BASE_PATH/2_det_clean'
    mp4_base_path = 'YOUR_BASE_PATH/1_scene/cliped_videos'
    output_path = 'YOUR_BASE_PATH/3_pose'
    
    device = args.device
    start = args.start
    end = args.end

    #load cfg
    pose_config = 'DWPose/mmpose/configs/wholebody_2d_keypoint/rtmpose/ubody/rtmpose-l_8xb32-270e_coco-ubody-wholebody-384x288.py'
    pose_checkpoint = 'dw-ll_ucoco_384.pth'

    # build pose estimator
    pose_estimator = init_pose_estimator(
    pose_config,
    pose_checkpoint,
    device=device,
    cfg_options=dict(
        model=dict(test_cfg=dict(output_heatmaps=False))))

    # pose data process
    scope = pose_estimator.cfg.get('default_scope', 'mmpose')
    if scope is not None:
        init_default_scope(scope)

    pose_test_pipeline = Compose(pose_estimator.cfg.test_dataloader.dataset.pipeline)
    pose_dataset_meta = pose_estimator.dataset_meta

    # Process files
    process_files(pickle_path, mp4_base_path, output_path, pose_estimator, pose_test_pipeline, pose_dataset_meta,start,end)
