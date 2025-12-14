import os
import pickle
import math

from scenedetect import SceneManager,open_video
from scenedetect.detectors import ContentDetector
from scenedetect.video_splitter import split_video_ffmpeg
from tqdm import tqdm

def find_scenes_save_videos(video_folder, output_folder,visualization,start, end):

    if not os.path.exists(os.path.join(output_folder, 'index_pkl_file')):
        os.makedirs(os.path.join(output_folder, 'index_pkl_file'))
    if visualization and (not os.path.exists(os.path.join(output_folder, 'cliped_videos'))):
        os.makedirs(os.path.join(output_folder, 'cliped_videos'))

    for root, dirs, files in os.walk(video_folder):
        video_files = [os.path.join(root, file) for file in files if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        total_videos = len(video_files)
        partition_size = math.ceil(total_videos / 20)
        start_idx = start * partition_size
        end_idx = min(end * partition_size, total_videos)

        for video_path in tqdm(video_files[start_idx:end_idx],desc='Process Videos'):
            base_filename = os.path.splitext(os.path.basename(video_path))[0]
            pkl_file_path = os.path.join(output_folder, 'index_pkl_file',f"{base_filename}_scene.pkl")
            video_file_path = os.path.join(output_folder, 'cliped_videos',f"{base_filename}")
            if os.path.exists(video_file_path):
                if len(os.listdir(video_file_path))!=0:
                    print(f'Skip the {base_filename}.mp4, which has been processed. ')
                    continue
            try:
                video = open_video(video_path)
                print(f'read {video_path}... ')
            except:
                print(f'read {video_path} failed... ')
                continue

            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector())
            scene_manager.detect_scenes(video, show_progress=True)

            
            scene_list = scene_manager.get_scene_list()
            #store the scenes index
            scene_list_output = []
            for index, scene in tqdm(enumerate(scene_list),desc='Generate scenes index pkl'):
                scene_list_output.append(scene[0].get_frames())
            
            with open(pkl_file_path, 'wb') as file:
                pickle.dump(scene_list_output, file)
            print(f'save {base_filename}_scene.pkl')

            #generate the clips
            if visualization:
                output_folder_video = os.path.join(output_folder,'cliped_videos',base_filename)
                if not os.path.exists(output_folder_video):
                    os.makedirs(output_folder_video)
                split_video_ffmpeg(video_path, scene_list, output_dir = output_folder_video,
                                   output_file_template ='$VIDEO_NAME-Scene-$SCENE_NUMBER-$START_FRAME.mp4', arg_override='-map 0',
                                    show_progress = True)

import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="index of the start and end")
    parser.add_argument("start", type=int, help="start index")
    parser.add_argument("end", type=int, help="end index")
    args = parser.parse_args()
    started_index = args.start
    end_index = args.end

    video_folder = 'YOUR VIDEO FOLDER PATH'  
    output_folder = 'YOUR OUTPUT FOLDER PATH'  

    print('start det scenes')
    find_scenes_save_videos(video_folder, output_folder, visualization=True, start=started_index, end=end_index)



