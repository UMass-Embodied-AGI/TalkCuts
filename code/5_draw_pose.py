import sys
import os
import cv2
import pickle
from PIL import Image
from tqdm import tqdm
import numpy as np
import av
from concurrent.futures import ThreadPoolExecutor, as_completed
import shutil
import util


def resize_image(input_image, resolution):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    k = float(resolution) / min(H, W)
    H *= k
    W *= k
    H = int(np.round(H / 64.0)) * 64
    W = int(np.round(W / 64.0)) * 64
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if k > 1 else cv2.INTER_AREA)
    return img


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y

def fy_draw_pose(pose, H, W):
    # pose 133*3
    # H, W = RGBimage.shape
    # add neck (L-shoulder & R-shoulder)
    neck = np.mean(pose[5:7],axis=0,keepdims=True) # [1*3]
    new_pose = np.insert(pose, 17, neck, axis=0) #[134*3]

    # body transform
    # transform mmpose to openpose
    mmpose_idx = [
        17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3
    ]
    openpose_idx = [
        1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17
    ]
    new_pose[openpose_idx] = new_pose[mmpose_idx]
    
    # split corrdinates and score
    new_pose_only = new_pose[:,:2] # [134*2] 
    new_pose_score = new_pose[:,2] #[134]
    
    # normalization with RGB shape
    new_pose_only[...,0] /= float(W)
    new_pose_only[...,1] /= float(H)


    # split human parts

    # body
    body_pose = new_pose_only[:18].copy()
    body_score = new_pose_score[:18]
    body_score = body_score[np.newaxis,:]

    for i in range(len(body_score)):
        for j in range(len(body_score[i])):
            if body_score[i][j] > 0.3:
                body_score[i][j] = int(18*i+j)
            else:
                body_score[i][j] = -1

    # !!! vis or un_vis (thresohld = 0.3) !!!
    un_visible = new_pose_score<0.3
    new_pose_only[un_visible] = -1

    # face
    face_pose = new_pose_only[24:92].copy()
    face_pose = face_pose[np.newaxis,:]
    face_score =  new_pose_score[24:92]


    # hand
    hands_pose  = new_pose_only[92:113].copy()
    hands_pose =  np.vstack([hands_pose[np.newaxis,:],  new_pose_only[np.newaxis,113:]])
    hand_score = new_pose_score[92:113].copy()
    hand_score = np.vstack([hand_score,  new_pose_score[113:]])

    # draw
    # basic image
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)
    # draw body
    canvas = util.draw_bodypose(canvas, body_pose, body_score)
    # draw hands
    canvas = util.draw_handpose(canvas, hands_pose)

    canvas = util.draw_facepose(canvas, face_pose)


    return canvas











def get_dict_slice(input_dict, start, end):
    # Convert dictionary items to a list
    items = list(input_dict.items())
    
    # Slice the list
    sliced_items = items[start:end]
    
    # Convert the sliced list back to a dictionary
    sliced_dict = dict(sliced_items)
    
    return sliced_dict


def get_video_dimensions(video_path):
    container = av.open(video_path)
    stream = container.streams.video[0]
    width = stream.width
    height = stream.height
    fps = stream.average_rate
    container.close()
    return height, width, fps




def main(video_root, pose_root, save_root, target_videos_path):
    mp4_filenames = [f for f in os.listdir(target_videos_path) if f.endswith('.mp4')]

    print(f'total: {len(mp4_filenames)} videos...')
    for mp4_filename in tqdm(mp4_filenames, desc="Processing videos"):
        video_path = os.path.join(video_root, mp4_filename)
        try:
            H, W, fps = get_video_dimensions(video_path)
        except FileNotFoundError as e:
            print(e)
            continue

        pkl_filename = os.path.splitext(mp4_filename)[0] + '.pkl'
        pkl_path = os.path.join(pose_root, pkl_filename)

        if not os.path.exists(pkl_path):
            continue

        with open(pkl_path, 'rb') as pkl_file:
            pose_data = pickle.load(pkl_file)

        save_path = os.path.join(save_root, os.path.splitext(mp4_filename)[0] + '.mp4')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        if os.path.isfile(save_path):
            print('video exists')
            continue

        with av.open(video_path) as input_container:
            input_video = input_container.streams.video[0]
            input_audio = next((s for s in input_container.streams if s.type == 'audio'), None)

            with av.open(save_path, 'w') as output_container:
                output_video = output_container.add_stream('libx264', rate=input_video.average_rate)
                output_video.width = W
                output_video.height = H
                output_video.pix_fmt = 'yuv420p'

                if input_audio:
                    output_audio = output_container.add_stream(template=input_audio)

                for frame_idx, frame in enumerate(input_container.decode(input_video)):
                    if frame_idx >= len(pose_data):
                        break

                    # Convert frame to numpy array
                    rgb_frame = frame.to_ndarray(format="rgb24")

                    # Draw pose on the RGB frame
                    pose_frame = fy_draw_pose(pose_data[frame_idx], H, W)
                    pose_frame = HWC3(pose_frame)

                    # Combine RGB frame and pose frame
                    combined_frame = cv2.addWeighted(rgb_frame, 1, pose_frame, 0.5, 0)

                    # Convert to PIL Image and then to av.VideoFrame
                    combined_frame_pil = Image.fromarray(combined_frame)
                    av_frame = av.VideoFrame.from_image(combined_frame_pil)

                    # Encode and write video frame
                    for packet in output_video.encode(av_frame):
                        output_container.mux(packet)

                # Flush video stream
                for packet in output_video.encode():
                    output_container.mux(packet)

                # Copy audio packets if audio stream exists
                if input_audio:
                    for packet in input_container.demux(input_audio):
                        packet.stream = output_audio
                        output_container.mux(packet)

if __name__ == "__main__":
    base = 'YOUR_BASE_PATH'
    pose_root = base + 'data/merge_all_videos/video_dataset_pkl/youtube1_ted100'
    video_root = base + 'data/merge_all_videos/video_dataset/youtube1_ted100'
    save_root = base + 'code/00_draw_ouput'
    target_videos_path = 'YOUR_BASE_PATH/video_out'

    # Output folder path
    os.makedirs(save_root, exist_ok=True)
    main( video_root, pose_root, save_root,target_videos_path)
