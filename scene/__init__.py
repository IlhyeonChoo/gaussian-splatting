#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import random
import json
import numpy as np
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.colmap_loader import read_extrinsics_binary, read_extrinsics_text
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

def load_colmap_camera_quality(source_path):
    """Return quality score per image_name based on valid 3D correspondences."""
    bin_path = os.path.join(source_path, "sparse/0/images.bin")
    txt_path = os.path.join(source_path, "sparse/0/images.txt")
    try:
        if os.path.exists(bin_path):
            images = read_extrinsics_binary(bin_path)
        elif os.path.exists(txt_path):
            images = read_extrinsics_text(txt_path)
        else:
            return {}
    except Exception:
        return {}

    quality = {}
    for img in images.values():
        quality[img.name] = int(np.sum(img.point3D_ids != -1))
    return quality

def subsample_cameras_quality_random(cameras, target_count, quality_scores, quality_ratio, seed):
    if target_count <= 0 or len(cameras) <= target_count:
        return cameras

    ratio = max(0.0, min(1.0, float(quality_ratio)))
    quality_pick = min(target_count, int(round(target_count * ratio)))

    ranked = sorted(
        cameras,
        key=lambda c: (quality_scores.get(c.image_name, 0), c.image_name),
        reverse=True,
    )

    selected = ranked[:quality_pick]
    selected_names = {c.image_name for c in selected}
    remaining = [c for c in cameras if c.image_name not in selected_names]

    random_pick = target_count - len(selected)
    if random_pick > 0 and remaining:
        rng = random.Random(seed)
        if random_pick >= len(remaining):
            selected.extend(remaining)
        else:
            selected.extend(rng.sample(remaining, random_pick))

    if len(selected) < target_count:
        fallback = [c for c in ranked if c.image_name not in {x.image_name for x in selected}]
        selected.extend(fallback[:target_count - len(selected)])

    return sorted(selected[:target_count], key=lambda c: c.image_name)

class Scene:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path,
                args.images,
                args.depths,
                args.eval,
                args.train_test_exp,
                args,
            )
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path,
                args.white_background,
                args.depths,
                args.eval,
                args,
            )
        else:
            assert False, "Could not recognize scene type!"

        train_cam_infos = scene_info.train_cameras
        test_cam_infos = scene_info.test_cameras

        max_train_cameras = getattr(args, "max_train_cameras", 0)
        if max_train_cameras > 0 and len(train_cam_infos) > max_train_cameras:
            before = len(train_cam_infos)
            quality_scores = load_colmap_camera_quality(args.source_path)
            train_cam_infos = subsample_cameras_quality_random(
                train_cam_infos,
                max_train_cameras,
                quality_scores,
                getattr(args, "camera_quality_ratio", 0.7),
                getattr(args, "camera_selection_seed", 42),
            )
            print(
                "Limiting training cameras: "
                f"{before} -> {len(train_cam_infos)} "
                f"(max_train_cameras={max_train_cameras}, "
                f"camera_quality_ratio={getattr(args, 'camera_quality_ratio', 0.7)}, "
                f"seed={getattr(args, 'camera_selection_seed', 42)})"
            )

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if test_cam_infos:
                camlist.extend(test_cam_infos)
            if train_cam_infos:
                camlist.extend(train_cam_infos)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(train_cam_infos)  # Multi-res consistent random shuffling
            random.shuffle(test_cam_infos)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(train_cam_infos, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(test_cam_infos, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"), args.train_test_exp)
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, train_cam_infos, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
