import os
import ocnn
import numpy as np
import glob
import torch
from tqdm import tqdm
import yaml
import time
from easydict import EasyDict as edict
from ocnn.dataset import CollateBatch
from chamferdist import ChamferDistance
import open3d as o3d
from datasets.completion import Transform, ReadFile


class Completion:  # (torch.nn.Module):
    def __init__(self, flags):
        # super().__init__()
        self.flags = flags
        self.model = ocnn.models.OUNet(
            flags.MODEL.channel, flags.MODEL.nout, flags.MODEL.depth, 
            flags.MODEL.full_depth, feature=flags.MODEL.feature
        ).eval().cuda()
        self.model.load_state_dict(
            torch.load("logs/completion/shapenet/checkpoints/00300.model.pth", map_location="cpu", weights_only=True)
        )
        self.read_file = ReadFile(has_normal=True)
        self.transform = Transform(flags.DATA.test)
        self.collate_batch = CollateBatch(merge_points=False)
        self.chamfer_dist = ChamferDistance().cuda().eval()
        self.cluster_threshold = 0.9

    def __call__(self, plyname):
        plydata = self.read_file(plyname)
        sample = self.transform(plydata, idx=None, aug=False)
        scale = sample["points_scale"]
        radius = sample["radius"]
        center = sample["center"]
        sample = self.collate_batch([sample])
        with torch.no_grad():
            octree_out = self.model(sample["octree"].cuda())["octree_out"]
        points_out = octree_out.to_points().points.cpu().numpy()
        # points_out /= scale
        # points_out /= radius
        # points_out += center
        return points_out

    def completion2mesh(self, pcd):
        points = np.asarray(pcd.points).astype(np.float32)
        normals = np.asarray(pcd.normals).astype(np.float32)
        colors = np.asarray(pcd.colors).astype(np.float32)
        plydata = {
            "points": points,
            "normals": normals,
            "colors": colors
        }
        # plydata = self.read_file(plyname)
        sample = self.transform(plydata, idx=None, aug=False)
        scale = sample["points_scale"]
        radius = sample["radius"]
        center = torch.from_numpy(sample["center"]).cuda()
        sample = self.collate_batch([sample])
        res = 0.0312 / scale / radius
        with torch.no_grad():
            octree_out = self.model(sample["octree"].cuda())["octree_out"]
            points_out = octree_out.to_points().points
            points_out /= scale
            points_out /= radius
            points_out += center

            input_points = torch.from_numpy(points).cuda()
            chamfer_dis_mat = self.chamfer_dist(points_out[None], input_points[None], reduction=None)[0]

        mask = chamfer_dis_mat.cpu().numpy() > res ** 2
        unseen_idxs = np.where(mask)[0]

        pred_pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(points_out.cpu().numpy()))
        unseen_pcd = pred_pcd.select_by_index(unseen_idxs)
        unseen_pcd.estimate_normals()
        unseen_pcd.orient_normals_towards_camera_location()
        unseen_pcd.normals = o3d.utility.Vector3dVector(-np.asarray(unseen_pcd.normals))

        merge_pcd = pcd + unseen_pcd
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(merge_pcd)
        (
            triangle_clusters,
            cluster_n_triangles,
            cluster_area,
        ) = mesh.cluster_connected_triangles()
        triangle_clusters = np.asarray(triangle_clusters)
        cluster_n_triangles = np.asarray(cluster_n_triangles)
        triangles_to_remove = (
            cluster_n_triangles[triangle_clusters]
            < cluster_n_triangles.max() * self.cluster_threshold
        )
        mesh.remove_triangles_by_mask(triangles_to_remove)
        mesh.remove_unreferenced_vertices()
        mesh.compute_vertex_normals()

        return mesh


def main():
    with open("logs/completion/shapenet/all_configs.yaml") as f:
        flags = edict(yaml.load(f, Loader=yaml.FullLoader))

    completion_module = Completion(flags)
    root = "data/ocnn_completion/shape.ply/*/"
    # root = "data/ocnn_completion/test4completion"
    root = "data/ocnn_completion/partial_pcd"
    root = "/data/hewenhao/code/learn_to_pick/tools/ocnn/2024-08-22_15-59-54/"
    files = glob.glob(f"{root}/*_partial.ply")
    for fname in files[:: 100]:
        input_pcd = o3d.io.read_point_cloud(fname)
        t0 = time.time()
        pred_points = completion_module(fname)
        t1 = time.time()
        mesh = completion_module.completion2mesh(input_pcd)
        t2 = time.time()
        print(t1 - t0, t2 - t1)
        continue


        input_pcd = o3d.io.read_point_cloud(fname.replace("partial.ply", "complete.ply"))
        pcd = o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pred_points))
        pcd.paint_uniform_color((1, 0, 0))
        pcd.estimate_normals()
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        partial_center = input_pcd.get_center()
        invert_flag = ((points - partial_center) * normals).sum(1) < 0
        normals[invert_flag] *= -1
        # pcd.normals = o3d.utility.Vector3dVector(normals)
        o3d.io.write_point_cloud(f"{os.path.basename(fname).replace('.ply', '')}_input.ply", o3d.io.read_point_cloud(fname))
        o3d.io.write_point_cloud(f"{os.path.basename(fname).replace('.ply', '')}_gt.ply", input_pcd)
        o3d.io.write_point_cloud(f"{os.path.basename(fname).replace('.ply', '')}_pd.ply", pcd)


if __name__ == '__main__':
  main()
