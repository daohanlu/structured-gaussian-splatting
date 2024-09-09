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
import glob
import os
import sys
from scipy.spatial.transform import Rotation

from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=='OPENCV':
            # we're ignoring the 4 distortion
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            if idx == 0:
                print('  OPENCV intrinsic params', intr.params)
        else:
            assert False, f"Colmap camera model {intr.model} not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos


def readCamerasFromZero123(path, white_background, extension=".png", train_split=True):
    cam_infos = []
    image_files = sorted(list(glob.glob(os.path.join(path, '*' + extension))))
    if train_split:
        image_files = image_files[:10]
    else:
        image_files = image_files[10:]

    for idx, frame_path in enumerate(image_files):
        npy_file = frame_path.replace(extension, ".npy")
        # Zero123 uses the "3x4 RT matrix from Blender". I think it's [R|T]
        # https://github.com/cvlab-columbia/zero123/blob/main/objaverse-rendering/scripts/blender_script.py
        blender_RT = np.load(npy_file)
        # blender_RT = np.concatenate((blender_RT, [[0, 0, 0, 1]]), axis=0)
        cam_name = str(frame_path)

        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = blender_RT
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        # c2w[:3, 1:3] *= -1
        # c2w[:3, :3] = Rotation.from_euler('xyz', [90, 0, 0], degrees=True).as_matrix().T @ c2w[:3,:3]
        c2w[1:3, :3] *= -1
        # c2w[:3, :3] = Rotation.from_euler('xyz', [0, 45, 90], degrees=True).as_matrix() @ c2w[:3, :3]

        # get the world-to-camera transform and set R, T
        # w2c = np.linalg.inv(c2w)
        w2c = c2w
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = -w2c[:3, 3]

        image_path = os.path.join(path, cam_name)
        image_name = Path(cam_name).stem
        image = Image.open(image_path)

        im_data = np.array(image.convert("RGBA"))

        bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

        # blender camera intrinsics:
        # https://github.com/cvlab-columbia/zero123/blob/f426883b1a7353d91ddc34a551dd91b6223e4ce8/objaverse-rendering/scripts/blender_script.py#L62
        FovY = focal2fov(35, 32)
        FovX = focal2fov(35, 32)

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                    image_path=image_path, image_name=image_name, width=image.size[0],
                                    height=image.size[1]))
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def normalize_points(pts: np.array):
    bbox_max = np.max(pts, axis=0)
    bbox_min = np.min(pts, axis=0)
    print('bbox_max', bbox_max)
    print('bbox_min', bbox_min)
    # center
    offset = -(bbox_min + bbox_max) / 2
    pts += offset[np.newaxis, :]
    # scale
    scale = 1 / (bbox_max - bbox_min).max()
    pts *= scale
    print('offset', -(np.max(pts, axis=0) + np.min(pts, axis=0)) / 2)
    input()
    return pts


def readMeshSyntheticInfo(path, white_background, eval, obj_path=None, extension=".png", decimate_factor=1.0, mesh_max_faces=-1):
    if obj_path is not None:
        # Zero123 dataset
        print("Reading Training Transforms")
        train_cam_infos = readCamerasFromZero123(path, white_background, extension, train_split=True)
        print("Reading Test Transforms")
        test_cam_infos = readCamerasFromZero123(path, white_background, extension, train_split=False)
    else:
        # NeRF Synthetic Datset with obj
        raise NotImplementedError()
        print("Reading Training Transforms")
        train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
        print("Reading Test Transforms")
        test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)


    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    import open3d as o3d
    if obj_path != "":
        mesh_path = obj_path
    else:
        mesh_path = os.path.join(path, "mesh3d.ply")
    ply_path = os.path.join(path, "points3d.ply")
    # mesh = o3d.io.read_triangle_mesh(mesh_path, enable_post_processing=True)  # Read mesh
    tri_model = o3d.io.read_triangle_model(mesh_path)
    xyzs = []
    for mesh_info in tri_model.meshes:
        print('adding mesh with name', mesh_info.mesh_name)
        mesh = mesh_info.mesh
        assert decimate_factor == 1.0 or mesh_max_faces == -1, "Decimate factor and mesh_max_faces are mutually exclusive"
        if decimate_factor != 1.0:
            mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=int(len(mesh.triangles) / decimate_factor))
            # if len(mesh.triangles) > mesh_max_faces:
            #     mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=mesh_max_faces)
        vertices = np.asarray(mesh.vertices)
        triangles = np.asarray(mesh.triangles)
        xyz = np.zeros((len(triangles), 3), dtype=np.float32)
        batch_size = 1024
        for i in range(0, len(triangles), batch_size):
            centroids = vertices[triangles[i: i + batch_size]]
            centroids = np.mean(centroids, axis=1)
            xyz[i:i + batch_size] = centroids
        xyzs.append(xyz)
    xyzs: np.array = np.concatenate(xyzs, axis=0)
    if mesh_max_faces != -1:
        if len(xyzs) > mesh_max_faces:
            quit()
    with open(obj_path.replace('.glb', '_normalization.json'), 'r') as f:
        normalization_dict: dict = json.load(f)
    xyzs *= float(normalization_dict["scale"])
    offset = np.array(normalization_dict["offset"])[np.newaxis, :]
    xyzs += offset
    xyzs = xyzs[:, [0, 2, 1]]
    xyzs[:, 1] *= -1

    # xyzs = normalize_points(xyzs)
    # print('Vertices:', np.asarray(mesh.vertices).shape)
    # print('Triangles:', np.asarray(mesh.triangles).shape)
    # print("Try to render a mesh with normals (exist: " +
    #       str(mesh.has_vertex_normals()) + ") and colors (exist: " +
    #       str(mesh.has_vertex_colors()) + ")")
    # mesh.compute_triangle_normals()
    # o3d.visualization.draw_geometries([mesh])

    # for an entire model
    # o3d.visualization.draw([tri_model])

    # print('init xyzs', xyzs.shape, xyzs)
    shs = np.float32(np.random.random((len(xyzs), 3)) / 255.0)
    storePly(ply_path, xyzs, SH2RGB(shs) * 255.0)
    pcd = BasicPointCloud(points=xyzs, colors=SH2RGB(shs), normals=np.zeros((len(xyzs), 3)))

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Mesh" : readMeshSyntheticInfo,
    "Zero123": readMeshSyntheticInfo,
}