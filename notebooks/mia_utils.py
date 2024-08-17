from scipy.spatial.transform import Rotation as R, Slerp
import torch
import os
from os.path import join as opj
import numpy as np
from tqdm import tqdm
import pandas as pd

from concurrent.futures import ThreadPoolExecutor, as_completed


def axis_angle_to_quaternion(axis_angle):
    """Convert axis-angle to quaternion."""
    rotation = R.from_rotvec(axis_angle)
    return rotation.as_quat()


def quaternion_to_axis_angle(quaternion):
    """Convert quaternion to axis-angle."""
    rotation = R.from_quat(quaternion)
    return rotation.as_rotvec()


def slerp(t, q0, q1):
    """Spherical linear interpolation (slerp) of quaternions."""
    rotations = R.from_quat([q0, q1])
    slerp = Slerp([0, 1], rotations)
    return slerp(t).as_quat()


def interpolate_poses(pose1, pose2, n_interpolations=1):
    """Interpolate between two sets of pose parameters (axis-angles)."""
    interpolated_poses = []
    for i in range(1, n_interpolations + 1):
        t = i / (n_interpolations + 1)
        interpolated_pose = []
        for j in range(len(pose1) // 3):
            aa1 = pose1[3 * j : 3 * j + 3]
            aa2 = pose2[3 * j : 3 * j + 3]
            q1 = axis_angle_to_quaternion(aa1)
            q2 = axis_angle_to_quaternion(aa2)
            qi = slerp(t, q1, q2)
            interpolated_pose.extend(quaternion_to_axis_angle(qi))
        interpolated_poses.append(interpolated_pose)
    return interpolated_poses


def interpolate_batch(batch, interpolation_func, n_interpolations=1):
    """Interpolate each pair of consecutive samples in a batch."""
    interpolated_batch = []
    for i in range(len(batch) - 1):
        interpolated_batch.append(batch[i])
        interpolated_intermediates = interpolation_func(batch[i], batch[i + 1], n_interpolations)
        interpolated_batch.extend(interpolated_intermediates)
    interpolated_batch.append(batch[-1])
    return np.array(interpolated_batch)


def interpolate_linear(param1, param2, n_interpolations=1):
    """Linear interpolation of parameters."""
    interpolated_params = []
    for i in range(1, n_interpolations + 1):
        t = i / (n_interpolations + 1)
        interpolated_param = (1 - t) * param1 + t * param2
        interpolated_params.append(interpolated_param)
    return interpolated_params


def convert_pare_to_full_img_cam(pare_cam, bbox_width, bbox_height, bbox_center, img_w, img_h, focal_length):
    # From https://github.com/mchiquier/musclesinaction/tree/main
    # Converts weak perspective camera estimated by PARE in
    # bbox coords to perspective camera in full image coordinates
    # from https://arxiv.org/pdf/2009.06549.pdf
    s, tx, ty = pare_cam[:, 0], pare_cam[:, 1], pare_cam[:, 2]
    res = 224
    tz = 2 * focal_length / (res * s)
    # pdb.set_trace()
    cx = 2 * (bbox_center[:, 0] - (img_w / 2.0)) / (s * bbox_width)
    cy = 2 * (bbox_center[:, 1] - (img_h / 2.0)) / (s * bbox_height)

    cam_t = np.stack([tx + cx, ty + cy, tz], axis=-1)

    return cam_t


def get_leaf_directories(root_dir):
    leaf_directories = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if not dirnames:
            leaf_directories.append(dirpath)
    return leaf_directories


def find_files(base_dir, pattern="emgvalues.npy"):
    matches = []
    for root, _, filenames in tqdm(os.walk(base_dir)):
        for filename in filenames:
            if filename == pattern:
                matches.append(os.path.join(root, filename))
    return matches


def index_mia_dataset(data_dir):
    # Search and list all .npy files in the specified directory and its subdirectories.
    file_list = find_files(data_dir)

    print(file_list[0])
    print(len(file_list))

    extract_metadata = lambda path: (os.path.split(path)[0], *tuple(path.split(os.path.sep)[-5:-1]))

    metadata = list(map(extract_metadata, file_list))

    columns = ["path", "split", "subject", "activity", "repetition"]

    # Convert the list of tuples into a DataFrame
    mia_metadata = pd.DataFrame(metadata, columns=columns)

    return mia_metadata


def load_emg_data(file_list):

    def process_file(file):
        data = np.load(file)
        if np.isnan(data).any():
            print(file)
            return None
        return torch.tensor(data)

    data_list = []  # Create an empty list to hold data arrays from each file.

    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_file, file): file for file in file_list}
        for future in tqdm(as_completed(futures), total=len(futures)):
            result = future.result()
            if result is not None:
                data_list.append(result)

    data_list = torch.stack(data_list, axis=0)

    return data_list


def mia_to_smpl_body(pose_dir, bm):
    pose_np = np.load(opj(pose_dir, "pose.npy"))
    betas_np = np.load(opj(pose_dir, "betas.npy"))
    predcam_np = np.load(opj(pose_dir, "predcam.npy"))
    bboxes_np = np.load(opj(pose_dir, "bboxes.npy"))

    bbox_center = np.stack((bboxes_np[:, 0] + bboxes_np[:, 2] / 2, bboxes_np[:, 1] + bboxes_np[:, 3] / 2), axis=-1)
    bbox_width = bboxes_np[:, 2]
    bbox_height = bboxes_np[:, 3]

    # These parameters result from the MIA settings, extracted from https://github.com/mchiquier/musclesinaction/tree/main
    transl_np = convert_pare_to_full_img_cam(
        pare_cam=predcam_np,
        bbox_width=bbox_width,
        bbox_height=bbox_height,
        bbox_center=bbox_center,
        img_w=1920,
        img_h=1080,
        focal_length=5000,
    )

    transl_np = transl_np - transl_np[0]

    # Vibe depth estimation is not really good, but in Muscles in Action most actions (roughly) happen in a plane
    # So we can set the z translation to 0
    transl_np[:, 2] /= 20

    # Interpolate in bnetween to get 59 pose samples (20 fps instead of 10 fps)
    pose_np_inter = interpolate_batch(pose_np, interpolate_poses, n_interpolations=1)
    betas_np_inter = interpolate_batch(betas_np, interpolate_linear, n_interpolations=1)
    transl_np_inter = interpolate_batch(transl_np, interpolate_linear, n_interpolations=1)

    # Ensure the shapes are correct for the SMPL model
    assert pose_np.shape[1] == 72, "Each pose should have 72 parameters (24 joints * 3 rotations)."
    assert betas_np.shape[1] == 10, "Each betas should have 10 parameters (shape coefficients)."

    pose_tensor = torch.tensor(pose_np_inter, dtype=torch.float32).cuda()
    betas_tensor = torch.tensor(betas_np_inter, dtype=torch.float32).cuda()
    transl_tensor = torch.tensor(transl_np_inter, dtype=torch.float32).cuda()

    # We rotate the model into the same orientation as the AMASS samples
    # Since HumanML3D is made for AMASS samples.

    # Assuming pose_tensor is already defined with shape (30, 3)
    orig_rotation = pose_tensor[:, :3].cpu().numpy()  # Shape (30,3), axis angle representation

    # Rotation to be applied
    rotation_matrix = np.array([[-1.0, 0.0, 0.0], [0.0, -1, 0], [0.0, 0, 1]])

    # Convert axis-angle to rotation matrices
    orig_rot_matrices = R.from_rotvec(orig_rotation).as_matrix()  # Shape (30, 3, 3)

    # Apply the new rotation matrix
    new_rot_matrices = np.einsum("ij,kjl->kil", rotation_matrix, orig_rot_matrices)  # Shape (30, 3, 3)

    # Convert back to axis-angle representation if needed
    new_orientation = R.from_matrix(new_rot_matrices).as_rotvec()  # Shape (30, 3)

    new_orientation = torch.tensor(new_orientation).float().cuda()

    # Create SMPLH body model
    # Assume zero rotation for hands
    # 15 joints per hand * 3 rotations = 45
    left_hand_pose = torch.zeros((pose_tensor.shape[0], 45)).cuda()
    right_hand_pose = torch.zeros((pose_tensor.shape[0], 45)).cuda()

    body = bm(
        betas=betas_tensor,
        body_pose=pose_tensor[:, 3:66],
        left_hand_pose=left_hand_pose,
        right_hand_pose=right_hand_pose,
        global_orient=new_orientation,
        transl=transl_tensor,
    )

    return body


def load_and_concat_mia_dat(preds_path, metadata, preds_root, pad=-1):
    gts = []
    preds = []

    result = metadata.loc[preds_path]

    # Ensure result is a DataFrame, even for a single row
    if not isinstance(result, pd.DataFrame):
        result = result.to_frame().T

    gt_paths = result["gt_name"].iloc[:2].to_list()
    pred_paths = result["pred_name"].iloc[:2].to_list()
    start_times = result["time_start"].iloc[:2].to_list()

    last_start_frame = -1

    for gt_path, pred_path, start_time in zip(gt_paths, pred_paths, start_times):
        try:
            gt = np.load(opj(preds_root, gt_path))[0]
            pred = np.load(opj(preds_root, pred_path))[0]
        except:
            return None, None, None


        while (start_time * 20 - last_start_frame > 28):
            if last_start_frame == -1 and start_time * 20 > 0:  # Special case: missing first block
                gts.append(np.zeros((28, gt.shape[-1])))
                preds.append(np.zeros((28, pred.shape[-1])))

                last_start_frame = 0
            else:
                gts.append(np.zeros((28, gt.shape[-1])))
                preds.append(np.zeros((28, pred.shape[-1])))

                last_start_frame += 28

        gts.append(gt)
        preds.append(pred)

        last_start_frame = start_time * 20

    gts = np.concatenate(gts, axis=0)
    T = gts.shape[0]
    if T < pad:
        gts = np.pad(gts, ((0, pad - T), (0, 0)), mode="constant", constant_values=0)

    preds = np.concatenate(preds, axis=0)
    T = preds.shape[0]

    if T < pad:
        preds = np.pad(preds, ((0, pad - T), (0, 0)), mode="constant", constant_values=0)

    return gts, preds, int(start_times[0] * 20), int(start_times[-1] * 20 + 28)
