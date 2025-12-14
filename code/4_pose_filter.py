import os
import pickle
import glob
import json
import numpy as np
from tqdm import tqdm


def func(pickle_data,out_pose_path):
    # Initialize an empty list to store all keypoint_scores
    all_keypoint_scores = []
    all_keypoints = []
    # Iterate through each sub-dict in pickle_data

    for key, sub_dict in pickle_data.items():
        # Extract keypoint_scores[0][:17] from each sub-dict and append to the list
        if 'keypoint_scores' in sub_dict:
            all_keypoint_scores.append(sub_dict['keypoint_scores'][0][:17])
            # import pdb
            # pdb.set_trace()
            all_keypoints.append(np.concatenate((np.array(sub_dict['keypoints'][0]), np.array(sub_dict['keypoint_scores'][0])[:, np.newaxis]),axis=-1))
    
    with open(out_pose_path, 'wb') as f:
        pickle.dump(np.array(all_keypoints), f)
    
    # Convert list to numpy array for easier calculation
    keypoint_scores_array = np.array(all_keypoint_scores)

    # Calculate the mean of each column, i.e., the mean along the T dimension
    mean_scores = np.mean(keypoint_scores_array, axis=0)
    
    return mean_scores

def process_files(path1, output_file,out_pose):

    threshold = 0.5
    # Read all pickle files in path1
    pickle_files = sorted(glob.glob(os.path.join(path1, "*.pickle")))
    

    whole_body = []
    half_body = []
    head_body = []
    low_q = []

    for pickle_file in tqdm(pickle_files):
        # Get the filename of the pickle file (without extension)
        base_name = os.path.splitext(os.path.basename(pickle_file))[0]
        out_pose_path = os.path.join(out_pose,f'{base_name}.pkl')
        try:
            with open(pickle_file, 'rb') as pf:
                pickle_data = pickle.load(pf)
                mean_score = func(pickle_data,out_pose_path)
                mp4_path =f"{base_name}.mp4"
                    # If quality is very low / detection error


                if all(score >= threshold for score in mean_score):
                    whole_body.append(mp4_path)
                elif (mean_score[9]>=threshold or  mean_score[10]>=threshold or mean_score[11]>=threshold or mean_score[12]>=threshold):
                    half_body.append(mp4_path)
                else:
                    if np.mean(mean_score[:5])<threshold:
                        low_q.append(mp4_path)
                    else:
                        head_body.append(mp4_path)
        except:
            print('error in load file')
            continue


    # Save results to the specified output files
    with open(f'{output_file}/whole_body.txt', 'w') as f:
        f.write("\n".join(whole_body))

    with open(f'{output_file}/half_body.txt', 'w') as f:
        f.write("\n".join(half_body))

    with open(f'{output_file}/head_body.txt', 'w') as f:
        f.write("\n".join(head_body))

    with open(f'{output_file}/low_quality.txt', 'w') as f:
        f.write("\n".join(low_q))

# Specify folder paths
path = "YOUR_BASE_PATH/3_pose"
output_file = "YOUR_BASE_PATH/3_pose_class"
out_pose = 'YOUR_BASE_PATH/3_pose_clean'
# Call the processing function
process_files(path, output_file,out_pose)
