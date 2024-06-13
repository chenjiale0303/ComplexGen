import copy
import json
import math
import multiprocessing
import os
import pymeshlab
import numpy as np
import open3d as o3d
from tqdm import tqdm
from multiprocessing import Pool
from plyfile import PlyData, PlyElement
import trimesh

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gt_dir', type=str, default='/data/GSP_test/test_data_whole3', help='path to gt data')
parser.add_argument('--output_dir', type=str, default='/data/GSP_test/baselines/ComplexGen_prediction_0415',
                    help='path to output sample and scale results')
parser.add_argument('--complexgen_output_dir', type=str, default='/root/code/ComplexGen/experiments/default/test_obj',
                    help='path to complexgen output')
parser.add_argument('--test_ids_dir', type=str, default='/root/code/ComplexGen/test_ids.txt', help='path to read test id list')
parser.add_argument('--topology', action='store_true', help='whether to output topology')
parser.add_argument('--viz_output_dir', type=str, default='', help='path to output viz results')
parser.add_argument('--prefix', type=str, default='', help='just output one model')
args = parser.parse_args()

gt_dir = args.gt_dir
output_dir = args.output_dir
viz_output_dir = args.viz_output_dir
complexgen_output_dir = args.complexgen_output_dir
test_ids_dir = args.test_ids_dir

os.makedirs(output_dir, exist_ok=True)

if viz_output_dir != '':
    os.makedirs(viz_output_dir, exist_ok=True)

os.makedirs(r"{}/sample_on_geom_refine/vertices".format(output_dir), exist_ok=True)
os.makedirs(r"{}/sample_on_geom_refine/curves".format(output_dir), exist_ok=True)
os.makedirs(r"{}/sample_on_geom_refine/surfaces".format(output_dir), exist_ok=True)
# ================================== Trim version =========================================
# os.makedirs(r"{}/sample_on_cut_grouped".format(output_dir), exist_ok=True)
# os.makedirs(r"{}/cut_grouped".format(output_dir), exist_ok=True)
# ================================== Trim version =========================================
os.makedirs(r"{}/geom_refine".format(output_dir), exist_ok=True)
os.makedirs(r"{}/topo".format(output_dir), exist_ok=True)

def complexgen_normalize_model(points_with_normal, to_unit_sphere=False):
    assert (len(points_with_normal.shape) == 2 and points_with_normal.shape[1] == 3)
    points = points_with_normal[:, :3]
    # normalize to unit bounding box
    max_coord = points.max(axis=0)
    min_coord = points.min(axis=0)
    center = (max_coord + min_coord) / 2.0
    scale = (max_coord - min_coord).max()
    # normalized_points = points - center
    # normalized_points *= 1.0/scale
    return center, scale


def process_id(id):
    import os

    json_file = r"{}/{}_geom_refine.json".format(complexgen_output_dir, id)
    if not os.path.exists(json_file):
        print(r"{} not exists".format(json_file))
        return

    # read json file
    with open(json_file, 'r') as f:
        data = json.load(f)

    gt_points = o3d.io.read_point_cloud(r'{}/poisson/{}.ply'.format(gt_dir, id))
    gt_points = np.asarray(gt_points.points)
    center, scale = complexgen_normalize_model(gt_points, to_unit_sphere=False)

    if args.topology:
        fe_adj_table = []
        if "patch2curve" in data and data["patch2curve"] is not None:
            for x, line in enumerate(data["patch2curve"]):
                neighbors = []
                neighbors.append(x)
                for y, is_adj in enumerate(line):
                    if is_adj == 1:
                        neighbors.append(y)
                fe_adj_table.append(neighbors)

        ev_adj_table = []
        if "curve2corner" in data and data["curve2corner"] is not None:
            for x, line in enumerate(data["curve2corner"]):
                neighbors = []
                neighbors.append(x)
                for y, is_adj in enumerate(line):
                    if is_adj == 1:
                        neighbors.append(y)
                ev_adj_table.append(neighbors)

        with open(r"{}/topo/{}.txt".format(output_dir, id), 'w') as f:
            f.write("FE\n")
            for neighbors in fe_adj_table:
                f.write(" ".join([str(x) for x in neighbors]))
                f.write("\n")
            f.write("EV\n")
            for neighbors in ev_adj_table:
                f.write(" ".join([str(x) for x in neighbors]))
                f.write("\n")

    # sampling on vertices
    if "corners" in data and data["corners"] is not None:
        out = o3d.geometry.PointCloud()
        for idx, corner in enumerate(data["corners"]):
            json_points = np.asarray(corner['pts']).reshape(-1, 3) * scale + center
            out.points = o3d.utility.Vector3dVector(
                np.concatenate([np.asarray(out.points), json_points], axis=0))
            if args.viz_output_dir != '':
                o_viz_pcd = o3d.geometry.PointCloud()
                o_viz_pcd.points = o3d.utility.Vector3dVector(json_points)
                os.makedirs(os.path.join(args.viz_output_dir, "{}".format(id)), exist_ok=True)
                o3d.io.write_point_cloud(os.path.join(args.viz_output_dir, "{}/{}_vertex.ply".format(id, idx)), o_viz_pcd)
        o3d.io.write_point_cloud(
            r"{}/sample_on_geom_refine/vertices/{}.ply".format(output_dir, id), out)

    # sampling on curves
    if "curves" in data and data["curves"] is not None:
        out_interpolate = o3d.geometry.PointCloud()
        out = o3d.geometry.PointCloud()
        for (idx, curve) in enumerate(data["curves"]):
            pts = np.asarray(curve['pts']).reshape(-1, 3) * scale + center
            interpolated_pts = []
            for i in range(len(pts) - 1):
                curve_length = np.linalg.norm(pts[i + 1] - pts[i])
                # Generate curve_length * 1000 interpolated points between each pair of points
                for t in np.linspace(0, 1, math.ceil(curve_length * 1000)):
                    interpolated_pt = (1 - t) * pts[i] + t * pts[i + 1]
                    interpolated_pts.append(interpolated_pt)
            interpolated_pts = np.asarray(interpolated_pts)
            out.normals = o3d.utility.Vector3dVector(
                np.concatenate([np.asarray(out.normals), np.ones_like(pts, dtype=np.int32) * idx], axis=0))
            out_interpolate.normals = o3d.utility.Vector3dVector(
                np.concatenate(
                    [np.asarray(out_interpolate.normals), np.ones_like(interpolated_pts, dtype=np.int32) * idx],
                    axis=0))
            out.points = o3d.utility.Vector3dVector(
                np.concatenate([np.asarray(out.points), pts], axis=0))
            out_interpolate.points = o3d.utility.Vector3dVector(
                np.concatenate([np.asarray(out_interpolate.points), interpolated_pts], axis=0))

            if args.viz_output_dir != '':
                o_viz_pcd = o3d.geometry.PointCloud()
                if curve['type'] == 'Line':
                    o_viz_pcd.points = o3d.utility.Vector3dVector(np.stack([pts[0], pts[-1]], axis=0))
                else:
                    o_viz_pcd.points = o3d.utility.Vector3dVector(interpolated_pts)
                os.makedirs(os.path.join(args.viz_output_dir, "{}".format(id)), exist_ok=True)
                o3d.io.write_point_cloud(os.path.join(args.viz_output_dir, "{}/{}_curve.ply".format(id, idx)), o_viz_pcd)

        o3d.io.write_point_cloud(
            r"{}/sample_on_geom_refine/curves/{}.ply".format(output_dir, id),
            out_interpolate)

    # sampling on patches
    if args.viz_output_dir == '' and "patches" in data and data["patches"] is not None and not os.path.exists(r"{}/sample_on_geom_refine/surfaces/{}.ply".format(output_dir, id)):
        out = o3d.geometry.PointCloud()
        for (idx, patch) in enumerate(data["patches"]):
            # (1200, 1) => (400, 3)
            grid = np.array(patch['grid']).reshape(-1, 3) * scale + center

            # create triangles
            triangles = [[i * 20 + j, i * 20 + j + 1, (i + 1) * 20 + j] for i in range(19) for j in range(19)] + [
                [i * 20 + j + 1, (i + 1) * 20 + j + 1, (i + 1) * 20 + j] for i in range(19) for j in range(19)]
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(grid)
            mesh.triangles = o3d.utility.Vector3iVector(triangles)

            primitive_area = mesh.get_surface_area()

            if primitive_area == 0:
                continue
            # sampling on the triangle grid
            pcl = mesh.sample_points_poisson_disk(number_of_points=math.ceil(primitive_area * 10000))
            out.normals = o3d.utility.Vector3dVector(
                np.concatenate([np.asarray(out.normals), np.ones_like(pcl.points, dtype=np.int32) * idx], axis=0))
            out.points = o3d.utility.Vector3dVector(
                np.concatenate([np.asarray(out.points), np.asarray(pcl.points)], axis=0))

        o3d.io.write_point_cloud(
            r"{}/sample_on_geom_refine/surfaces/{}.ply".format(output_dir, id), out)


def my_obj_loader(file_path):
    points = [] 
    faces = []
    groups = []
    current_group = -1

    try:
        with open(file_path, 'r') as file:
            for line in file:
                tokens = line.split()
                if not tokens:
                    continue

                if tokens[0] == 'v':
                    # Parse vertex coordinates and add to the points list
                    points.append([float(coord) for coord in tokens[1:4]])
                elif tokens[0] == 'f':
                    # Parse face indices (adjust for 0-based index) and add to the current group
                    face = [int(idx.split('/')[0]) - 1 for idx in tokens[1:]]
                    if current_group == -1:
                        groups.append([face])
                    else:
                        groups[current_group].append(face)
                elif tokens[0] == 'g':
                    # Start a new group
                    groups.append([])
                    current_group += 1

        # Convert lists to trimesh objects
        trimeshes = []
        for group_faces in groups:
            # Create a mesh from vertices and faces
            mesh = trimesh.Trimesh(vertices=np.array(points), faces=np.array(group_faces))
            trimeshes.append(mesh)

        return trimeshes

    except Exception as e:
        print(f"Error loading OBJ file: {e}")
        return []

def scale_geom_refine_and_cut_grouped(id):
    if not os.path.exists(r"{}/{}_geom_refine.obj".format(complexgen_output_dir, id)):
        return
    gt_points = o3d.io.read_point_cloud(r"{}/poisson/{}.ply".format(gt_dir, id))
    gt_points = np.asarray(gt_points.points)
    center, scale = complexgen_normalize_model(gt_points, to_unit_sphere=False)

    ms = pymeshlab.MeshSet()
    # Uncomment the following codes if you have trim version
    # ================================= Trim Version ===================================================
    # ms.load_new_mesh(r"{}/{}_extraction_cut_grouped.obj".format(complexgen_output_dir, id))
    # ms.set_current_mesh(0)
    # m = ms.current_mesh()
    # vertices = m.vertex_matrix()
    # faces = m.face_matrix()

    # # mesh=o3d.io.read(r"cut_grouped_bak/{}_extraction_cut_grouped.obj".format(prefix))
    # # vertices = np.asarray(mesh.vertices)

    # vertices = vertices * scale + center
    # m = pymeshlab.Mesh(vertices, faces)
    # ms = pymeshlab.MeshSet()
    # ms.add_mesh(m, "cube_mesh")
    # ms.save_current_mesh(r"{}/cut_grouped/{}_cut_grouped.ply".format(output_dir, id))

    # ms = my_obj_loader(r"{}/{}_extraction_cut_grouped.obj".format(complexgen_output_dir, id))

    # # plydata_vertices is (n, 4) array (x, y, z, primitive_index)
    # plydata_vertices = None
    # for idx, mesh in enumerate(ms):
    #     triangle_mesh = o3d.geometry.TriangleMesh()
    #     triangle_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    #     triangle_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    #     pcl = triangle_mesh.sample_points_poisson_disk(number_of_points=math.ceil(triangle_mesh.get_surface_area() * 10000))
    #     points = np.asarray(pcl.points) * scale + center
    #     primitive_index = np.ones((points.shape[0], 1), dtype=np.int32) * idx
    #     points = np.concatenate((points, primitive_index), axis=1)
    #     if plydata_vertices is None:
    #         plydata_vertices = points
    #     else:
    #         plydata_vertices = np.concatenate((plydata_vertices, points), axis=0)

    # plydata_vertices = np.array([tuple(v) for v in plydata_vertices], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('primitive_index', 'i4')])
    # PlyData([PlyElement.describe(plydata_vertices, 'vertex')], text=True).write(r"{}/sample_on_cut_grouped/{}.ply".format(output_dir, id))
    # ================================= Trim Version ===================================================
    
    
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(r"{}/{}_geom_refine.obj".format(complexgen_output_dir, id))
    ms.set_current_mesh(0)
    m = ms.current_mesh()
    vertices = m.vertex_matrix()
    faces = m.face_matrix()

    vertices = vertices * scale + center
    m = pymeshlab.Mesh(vertices, faces)
    ms = pymeshlab.MeshSet()
    ms.add_mesh(m, "cube_mesh")
    ms.save_current_mesh(r"{}/geom_refine/{}_geom_refine.ply".format(output_dir, id))

    # mesh.vertices = o3d.utility.Vector3dVector(vertices)
    # o3d.io.write_triangle_mesh(r"geom_refine/{}_geom_refine.ply".format(prefix), mesh)


if __name__ == "__main__":
    ids = [line.strip() for line in open(test_ids_dir).readlines()]
    ids = sorted(ids)
    if args.prefix != '':
        process_id(args.prefix)

    with Pool(multiprocessing.cpu_count()) as executor:
        executor.map(process_id, ids)

    # for id in ids:
    #     process_id(id)

    with Pool(multiprocessing.cpu_count()) as executor:
        executor.map(scale_geom_refine_and_cut_grouped, ids)


