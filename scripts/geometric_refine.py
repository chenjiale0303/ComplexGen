import os
from multiprocessing import Pool


exe_path = './GeometricRefine/Bin/GeometricRefine'
pc_ply_path = './data/default/test_point_clouds'
complex_path = './experiments/default/test_obj'

suffix = '_extraction.complex'

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--noise', action = 'store_true', help = 'noise/partial data')
args = parser.parse_args()

if args.noise:
	pc_ply_path = complex_path

def process_one(f):
	print(f)
	fn = f.replace(suffix,'')
	if not args.noise:
		os.system('{} -i {} -p {} --use_all_nb --last_iter 5'.format(exe_path, os.path.join(complex_path, f), os.path.join(pc_ply_path, '{}_10000.ply'.format(fn))))
	else:
		os.system('{} -i {} -p {} --use_all_nb --last_iter 5'.format(exe_path, os.path.join(complex_path, f), os.path.join(pc_ply_path, '{}_0_input.ply'.format(fn))))


def main_batch():
	allfs = os.listdir(complex_path)
	allcomplex = []
	flag_parallel = True
	for f in allfs:
		if f.endswith(suffix) and not os.path.exists(os.path.join(complex_path, f.replace(suffix, "_geom_refine.json"))):
			allcomplex.append(f)
	

	if not flag_parallel:
		for f in allcomplex:
			process_one(f)		
		return
	with Pool(12) as p:
		p.map(process_one, allcomplex)

if __name__ == "__main__":
	main_batch()
