from load_ply import *
from plywrite import *
import numpy as np
import os
import open3d as o3d
import math
from tqdm import tqdm
import subprocess 

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


def get_test_point_cloud_name(ply_file):
    sample_id = ply_file.split('/')[-1].split('_')[0]
    return '%s_10000.ply' % sample_id

def generate_one(ply_file):
    sample_id = ply_file.split('/')[-1].split('_')[0]
    print("Generate data for sample %s" % sample_id)
    # prepare pkl data from ply file    
    points = np.asarray(o3d.io.read_point_cloud(ply_file).points)
    normals = np.asarray(o3d.io.read_point_cloud(ply_file).normals)
    pwn = np.concatenate((points, normals), axis=1)
    pkl_points = read_from_pkl('data/train_small/packed/packed_000000.pkl')
    pkl_points['surface_points'] = pwn
    pkl_points['filename'] = sample_id + '_fix.pkl'
    write_to_pkl('data/default/test/packed/packed_000000.pkl', pkl_points)
    # run network
    os.system('python Minkowski_backbone.py --experiment_name default --enable_automatic_restore --no_pe --hn_scale --input_normal_signals --patch_grid --ourresnet --eval --no_output --parsenet --patch_close --patch_emd --patch_uv --gpu 0')
    

def complex_extraction():
    # extraction
    process = subprocess.Popen(['python', 'PostProcess/complex_extraction.py', '--folder', './experiments/default/test_obj'])
    process.wait()

def geometric_refine():
    process = subprocess.Popen(['python', 'scripts/geometric_refine.py'])
    process.wait()

def visualization(ids):
    for id in ids:
        process = subprocess.Popen(['python', 'vis/gen_vis_result.py', '-i', f'experiments/default/test_obj/{id}_extraction.json'])
        process.wait()
        process = subprocess.Popen(['python', 'vis/gen_vis_result.py', '-i', f'experiments/default/test_obj/{id}_geom_refine.json'])
        process.wait()

def generate_complexgen_result(ids):
    # clean test_obj
    for id in tqdm(ids):
        generate_one(os.path.join('data/default/test_point_clouds', id + '_10000.ply'))

if __name__ == '__main__':
    ids = [file.strip() for file in open(r"test_ids.txt").readlines()]
    generate_complexgen_result(ids)
    complex_extraction()
    # geometric_refine()
    # visualization(ids)