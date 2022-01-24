import os, sys
import cv2
import numpy as np
from glob import glob
from pose_utils import xinzhu_gen_poses
from normalize import xinzhu_get_normalization

_project_folder_ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(os.path.abspath(__file__)))))
if _project_folder_ not in sys.path:
    sys.path.insert(0, _project_folder_)
os.chdir(_project_folder_)

source_dir = 'F:/FaceIdRecon'
target_dir = './public_data'

def gen_image_and_mask_dir(case_name, 
                           interval=10, 
                           min_threshold=10, 
                           max_threshold=254, 
                           kernel_size=20):
    
    i_src_paths = sorted(glob(os.path.join(source_dir, case_name, 'rgb_*.png')))
    m_src_paths = sorted(glob(os.path.join(source_dir, case_name, 'depth_*.png')))
    case_dir = os.path.join(target_dir, 'facerecon_%s'%case_name)
    i_tgt_dir = os.path.join(case_dir, 'image')
    m_tgt_dir = os.path.join(case_dir, 'mask')
    os.makedirs(i_tgt_dir, exist_ok=True)
    os.makedirs(m_tgt_dir, exist_ok=True)

    assert(len(i_src_paths) == len(m_src_paths))
    N = len(i_src_paths)
    cnt = 0

    for i in range(1, N+1):
        if i % interval != 0: continue
    
        i_path = i_src_paths[i]
        m_path  = m_src_paths[i]
        image = cv2.imread(i_path)
        mask  = cv2.imread(m_path)
        # print(i_path, m_path)
        
        mask[mask < min_threshold] = 0
        mask[mask > max_threshold] = 0
        mask[mask != 0] = 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), dtype=np.uint8))
        
        cv2.imwrite(os.path.join(i_tgt_dir, '%03d.png'%cnt), image)
        cv2.imwrite(os.path.join(m_tgt_dir, '%03d.png'%cnt), mask)
        
        cnt += 1
        
    return case_dir, cnt


if __name__ == '__main__':
    # case_id = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    case_dict = {0: 'gyh',
                 1: 'kzy',
                 2: 'lsl',
                 3: 'zbf',
                 4: 'ztf'}

    for case_id in range(5):
        case_name = case_dict[case_id]
        
        case_dir, view_pts = gen_image_and_mask_dir(case_name)
        print('%d view points' % view_pts)

        data = xinzhu_gen_poses(case_dir)
        data = xinzhu_get_normalization(case_dir, data)
        np.savez(os.path.join(case_dir, 'cameras_sphere.npz'), **data)
        
        print('='*100)
        print('Data ready in the directory: %s' % case_dir)
        print('='*100)