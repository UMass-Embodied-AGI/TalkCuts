import os
import pickle
import pdb
from tqdm import tqdm
import numpy as np
from mmpose.evaluation.functional import nms

def det_post(data,file_path):
    filter_res = []
    check_list = []
    for det_result in data:
        pred_instance = det_result.pred_instances.cpu().numpy()
        img_id = det_result.img_id
        bboxes = np.concatenate(
            (pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        # person & score
        bboxes = bboxes[np.logical_and(pred_instance.labels == 0,
                                    pred_instance.scores >0.5)]
        # bboxes
        bboxes = bboxes[nms(bboxes, 0.3), :]

        # store, if only one bbox
        if len(bboxes)==1:
            new_dict = {
                'img_id':img_id,
                'bboxes':bboxes
            }
        # skip, if there is no bbox
        elif len(bboxes)==0:
            print('No bbox')
            return None
        # select the largest bbox
        else: 
            areas = [(y2 - y1) * (x2 - x1) for x1, y1, x2, y2,_ in bboxes]
            max_area_index = areas.index(max(areas))
            new_bbox = bboxes[max_area_index]
            new_dict = {
                'img_id':img_id,
                'bboxes':new_bbox[None]
            }
        # add this frame
        filter_res.append(new_dict)
        refer_bboxes  = new_dict['bboxes']

        # check if the bbox area is satisfied and single person
        if min(abs(refer_bboxes[0][2]-refer_bboxes[0][0]),abs(refer_bboxes[0][3]-refer_bboxes[0][1]))>=192 and len(bboxes)==1:
            check_list.append(1)
        else:
            check_list.append(0)
    
    if sum(check_list)/len(check_list)>=0.8:
        return filter_res
    else:
        print('no fit')
        return None



def read_pickle_files(directory_path,out_path):
    """
    Read all pickle files from the given directory and return a list of their contents.

    Parameters:
    directory_path (str): The path to the directory containing pickle files.

    Returns:
    list: A list containing the contents of all pickle files.
    """
    pickle_contents = []
    for root, dirs, files in os.walk(directory_path):
        for file in tqdm(sorted(files)):
            if file.endswith('.pickle') or file.endswith('.pkl'):
                file_path = os.path.join(root, file)

                out_file_name = file.split('.')[0].split('_det')[0] + '.pickle'
                # import pdb
                # pdb.set_trace()
                out_file = os.path.join(out_path,out_file_name)
                if os.path.isfile(out_file):
                    print(f'skip {out_file}')
                    continue
                try:
                    with open(file_path, 'rb') as f:
                        content = pickle.load(f)
                except:
                    print('error open file')
                    continue
                filter_res = det_post(content,file_path)
                if filter_res:
                    with open(out_file, 'wb') as file:
                        pickle.dump(filter_res, file)

            
    
    return pickle_contents


in_path = 'YOUR_INPUT_PATH/2_det'
out_path = 'YOUR_OUTPUT_PATH/2_det_clean'

read_pickle_files(in_path, out_path)