from load_ply import *
from plywrite import *
import numpy as np
import os
import open3d as o3d
import math
from tqdm import tqdm
import subprocess 
import pickle
import shutil

os.makedirs('experiments/default/test_obj', exist_ok=True)
os.makedirs('default/test_point_clouds', exist_ok=True)

def read_from_pkl(pkl_file):
    import pickle
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    return data

def write_to_pkl(pkl_file, data):
    import pickle
    with open(pkl_file, 'wb') as f:
        pickle.dump(data, f)

def complex_extraction():
    # extraction
    process = subprocess.Popen(['python', 'PostProcess/complex_extraction.py', '--folder', './experiments/default/test_obj'])
    process.wait()

def geometric_refine():
    process = subprocess.Popen(['python', 'scripts/geometric_refine.py'])
    process.wait()
    os.system("rm *.obj")
    os.system("rm *.ply")
    os.system("rm *.xyz")

def visualization(ids):
    for id in ids:
        process = subprocess.Popen(['python', 'vis/gen_vis_result.py', '-i', f'experiments/default/test_obj/{id}_geom_refine.json'])
        process.wait()

def official_normalize_model(points_with_normal, to_unit_sphere=False):
    assert(len(points_with_normal.shape) == 2 and points_with_normal.shape[1] == 3)
    points = points_with_normal[:,:3]
    #normalize to unit bounding box
    max_coord = points.max(axis=0)
    min_coord = points.min(axis=0)
    center = (max_coord + min_coord) / 2.0
    scale = (max_coord - min_coord).max()
    normalized_points = points - center
    if(to_unit_sphere):
        scale = math.sqrt(np.square(normalized_points).sum(-1).max())*2
    # normalized_points *= 0.95/scale
    normalized_points *= 1.0/scale
    return normalized_points

def generate_all(ids):
    pkl_points = read_from_pkl('data/train_small/packed/packed_000000.pkl')

    data = []
    with open("nvd_test.pkl", 'wb') as f:
        for id in ids:
            ply_file = os.path.join('data/default/test_point_clouds', id + '_10000.ply')
            sample_id = ply_file.split('/')[-1].split('_')[0]
            print("Generate data for sample %s" % sample_id)
            # prepare pkl data from ply file    
            pc = o3d.io.read_point_cloud(ply_file)
            points = np.asarray(pc.points)
            points = official_normalize_model(points)
            normals = np.asarray(pc.normals)
            pwn = np.concatenate((points, normals), axis=1)
            item = pkl_points.copy()
            item["surface_points"] = pwn
            item['filename'] = sample_id + '_fix.pkl'
            data.append(item)
            pickle.dump(item, f)

    shutil.copy2("nvd_test.pkl", 'data/default/test/packed/packed_000000.pkl')

    os.system('python Minkowski_backbone.py --experiment_name default --enable_automatic_restore --no_pe --hn_scale --input_normal_signals --patch_grid --ourresnet --eval --no_output --parsenet --patch_close --patch_emd --patch_uv --gpu 0')

    return


if __name__ == '__main__':
    ids = [file.strip() for file in open(r"test_ids.txt").readlines()]
    generate_all(ids)
    complex_extraction()
    geometric_refine()
    # visualization(ids)