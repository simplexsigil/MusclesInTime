# This code is based on the notebooks for extracting and processing motion data from the HumanML3D project.
# The code was adapted to our use case.
# Source: https://github.com/EricGuo5513/HumanML3D
# The original code is licensed under the MIT License.
# Copyright (c) 2022 Chuan Guo


from os.path import join as pjoin

import numpy as np
import os
from musint.utils.hml3d_utils.paramUtil import *
from musint.utils.hml3d_utils.common.quaternion import *
from musint.utils.hml3d_utils.common.skeleton import Skeleton

import torch
from tqdm import tqdm
import os
import os.path as osp


tgt_offsets = None
n_raw_offsets = None
kinematic_chain = None
# Lower legs
l_idx1, l_idx2 = 5, 8
# Right/Left foot
fid_r, fid_l = [8, 11], [7, 10]
# Face direction, r_hip, l_hip, sdr_r, sdr_l
face_joint_indx = [2, 1, 17, 16]
# l_hip, r_hip
r_hip, l_hip = 2, 1
joints_num = 22
# ds_num = 8

def swap_left_right(data):
    """Swaps the left and right joints in the data for data augmentation.

    Args:
        data (ndarray): The input data array with shape (N, M, 3), where N is the number of samples,
            M is the number of joints, and 3 represents the x, y, and z coordinates of each joint.

    Returns:
        ndarray: The modified data array with the left and right joints swapped.

    Raises:
        AssertionError: If the input data array does not have the expected shape.

    References:
        This function is from the implementation in the HumanML3D repository:
        https://github.com/EricGuo5513/HumanML3D
    """
    assert len(data.shape) == 3 and data.shape[-1] == 3
    data = data.copy()
    data[..., 0] *= -1
    right_chain = [2, 5, 8, 11, 14, 17, 19, 21]
    left_chain = [1, 4, 7, 10, 13, 16, 18, 20]
    left_hand_chain = [22, 23, 24, 34, 35, 36, 25, 26, 27, 31, 32, 33, 28, 29, 30]
    right_hand_chain = [43, 44, 45, 46, 47, 48, 40, 41, 42, 37, 38, 39, 49, 50, 51]
    tmp = data[:, right_chain]
    data[:, right_chain] = data[:, left_chain]
    data[:, left_chain] = tmp
    if data.shape[1] > 24:
        tmp = data[:, right_hand_chain]
        data[:, right_hand_chain] = data[:, left_hand_chain]
        data[:, left_hand_chain] = tmp
    return data


# Recover global angle and positions for rotation data
# root_rot_velocity (B, seq_len, 1)
# root_linear_velocity (B, seq_len, 2)
# root_y (B, seq_len, 1)
# ric_data (B, seq_len, (joint_num - 1)*3)
# rot_data (B, seq_len, (joint_num - 1)*6)
# local_velocity (B, seq_len, joint_num*3)
# foot contact (B, seq_len, 4)
def recover_root_rot_pos(data):
    rot_vel = data[..., 0]
    r_rot_ang = torch.zeros_like(rot_vel).to(data.device)
    '''Get Y-axis rotation from rotation velocity'''
    r_rot_ang[..., 1:] = rot_vel[..., :-1]
    r_rot_ang = torch.cumsum(r_rot_ang, dim=-1)

    r_rot_quat = torch.zeros(data.shape[:-1] + (4,)).to(data.device)
    r_rot_quat[..., 0] = torch.cos(r_rot_ang)
    r_rot_quat[..., 2] = torch.sin(r_rot_ang)

    r_pos = torch.zeros(data.shape[:-1] + (3,)).to(data.device)
    r_pos[..., 1:, [0, 2]] = data[..., :-1, 1:3]
    '''Add Y-axis rotation to root position'''
    r_pos = qrot(qinv(r_rot_quat), r_pos)

    r_pos = torch.cumsum(r_pos, dim=-2)

    r_pos[..., 1] = data[..., 3]
    return r_rot_quat, r_pos


def recover_from_rot(data, joints_num, skeleton):
    r_rot_quat, r_pos = recover_root_rot_pos(data)

    r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)

    start_indx = 1 + 2 + 1 + (joints_num - 1) * 3
    end_indx = start_indx + (joints_num - 1) * 6
    cont6d_params = data[..., start_indx:end_indx]
    #     print(r_rot_cont6d.shape, cont6d_params.shape, r_pos.shape)
    cont6d_params = torch.cat([r_rot_cont6d, cont6d_params], dim=-1)
    cont6d_params = cont6d_params.view(-1, joints_num, 6)

    positions = skeleton.forward_kinematics_cont6d(cont6d_params, r_pos)

    return positions


def recover_from_ric(data, joints_num):
    r_rot_quat, r_pos = recover_root_rot_pos(data)
    positions = data[..., 4:(joints_num - 1) * 3 + 4]
    positions = positions.view(positions.shape[:-1] + (-1, 3))

    '''Add Y-axis rotation to local joints'''
    positions = qrot(qinv(r_rot_quat[..., None, :]).expand(positions.shape[:-1] + (4,)), positions)

    '''Add root XZ to joints'''
    positions[..., 0] += r_pos[..., 0:1]
    positions[..., 2] += r_pos[..., 2:3]

    '''Concate root and joints'''
    positions = torch.cat([r_pos.unsqueeze(-2), positions], dim=-2)

    return positions


def process_file(positions, feet_thre):
    # (seq_len, joints_num, 3)
    #     '''Down Sample'''
    #     positions = positions[::ds_num]

    '''Uniform Skeleton'''
    positions = uniform_skeleton(positions, tgt_offsets)

    '''Put on Floor'''
    floor_height = positions.min(axis=0).min(axis=0)[1]
    positions[:, :, 1] -= floor_height
    #     print(floor_height)

    #     plot_3d_motion("./positions_1.mp4", kinematic_chain, positions, 'title', fps=20)

    '''XZ at origin'''
    root_pos_init = positions[0]
    root_pose_init_xz = root_pos_init[0] * np.array([1, 0, 1])
    positions = positions - root_pose_init_xz

    # '''Move the first pose to origin '''
    # root_pos_init = positions[0]
    # positions = positions - root_pos_init[0]

    '''All initially face Z+'''
    r_hip, l_hip, sdr_r, sdr_l = face_joint_indx
    across1 = root_pos_init[r_hip] - root_pos_init[l_hip]
    across2 = root_pos_init[sdr_r] - root_pos_init[sdr_l]
    across = across1 + across2
    across = across / np.sqrt((across ** 2).sum(axis=-1))[..., np.newaxis]

    # forward (3,), rotate around y-axis
    forward_init = np.cross(np.array([[0, 1, 0]]), across, axis=-1)
    # forward (3,)
    forward_init = forward_init / np.sqrt((forward_init ** 2).sum(axis=-1))[..., np.newaxis]

    #     print(forward_init)

    target = np.array([[0, 0, 1]])
    root_quat_init = qbetween_np(forward_init, target)
    root_quat_init = np.ones(positions.shape[:-1] + (4,)) * root_quat_init

    positions_b = positions.copy()

    positions = qrot_np(root_quat_init, positions)

    #     plot_3d_motion("./positions_2.mp4", kinematic_chain, positions, 'title', fps=20)

    '''New ground truth positions'''
    global_positions = positions.copy()

    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(positions[:, 0, 0], positions[:, 0, 2], marker='o', color='r')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    """ Get Foot Contacts """

    def foot_detect(positions, thres):
        velfactor, heightfactor = np.array([thres, thres]), np.array([3.0, 2.0])

        feet_l_x = (positions[1:, fid_l, 0] - positions[:-1, fid_l, 0]) ** 2
        feet_l_y = (positions[1:, fid_l, 1] - positions[:-1, fid_l, 1]) ** 2
        feet_l_z = (positions[1:, fid_l, 2] - positions[:-1, fid_l, 2]) ** 2
        #     feet_l_h = positions[:-1,fid_l,1]
        #     feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor) & (feet_l_h < heightfactor)).astype(np.float)
        feet_l = ((feet_l_x + feet_l_y + feet_l_z) < velfactor).astype(np.float32)

        feet_r_x = (positions[1:, fid_r, 0] - positions[:-1, fid_r, 0]) ** 2
        feet_r_y = (positions[1:, fid_r, 1] - positions[:-1, fid_r, 1]) ** 2
        feet_r_z = (positions[1:, fid_r, 2] - positions[:-1, fid_r, 2]) ** 2
        #     feet_r_h = positions[:-1,fid_r,1]
        #     feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor) & (feet_r_h < heightfactor)).astype(np.float)
        feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float32)
        return feet_l, feet_r
    #
    feet_l, feet_r = foot_detect(positions, feet_thre)
    # feet_l, feet_r = foot_detect(positions, 0.002)

    '''Quaternion and Cartesian representation'''
    r_rot = None

    def get_rifke(positions):
        '''Local pose'''
        positions[..., 0] -= positions[:, 0:1, 0]
        positions[..., 2] -= positions[:, 0:1, 2]
        '''All pose face Z+'''
        positions = qrot_np(np.repeat(r_rot[:, None], positions.shape[1], axis=1), positions)
        return positions

    def get_quaternion(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=False)

        '''Fix Quaternion Discontinuity'''
        quat_params = qfix(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        quat_params[1:, 0] = r_velocity
        # (seq_len, joints_num, 4)
        return quat_params, r_velocity, velocity, r_rot

    def get_cont6d_params(positions):
        skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
        # (seq_len, joints_num, 4)
        quat_params = skel.inverse_kinematics_np(positions, face_joint_indx, smooth_forward=True)

        '''Quaternion to continuous 6D'''
        cont_6d_params = quaternion_to_cont6d_np(quat_params)
        # (seq_len, 4)
        r_rot = quat_params[:, 0].copy()
        #     print(r_rot[0])
        '''Root Linear Velocity'''
        # (seq_len - 1, 3)
        velocity = (positions[1:, 0] - positions[:-1, 0]).copy()
        #     print(r_rot.shape, velocity.shape)
        velocity = qrot_np(r_rot[1:], velocity)
        '''Root Angular Velocity'''
        # (seq_len - 1, 4)
        r_velocity = qmul_np(r_rot[1:], qinv_np(r_rot[:-1]))
        # (seq_len, joints_num, 4)
        return cont_6d_params, r_velocity, velocity, r_rot

    cont_6d_params, r_velocity, velocity, r_rot = get_cont6d_params(positions)
    positions = get_rifke(positions)

    #     trejec = np.cumsum(np.concatenate([np.array([[0, 0, 0]]), velocity], axis=0), axis=0)
    #     r_rotations, r_pos = recover_ric_glo_np(r_velocity, velocity[:, [0, 2]])

    # plt.plot(positions_b[:, 0, 0], positions_b[:, 0, 2], marker='*')
    # plt.plot(ground_positions[:, 0, 0], ground_positions[:, 0, 2], marker='o', color='r')
    # plt.plot(trejec[:, 0], trejec[:, 2], marker='^', color='g')
    # plt.plot(r_pos[:, 0], r_pos[:, 2], marker='s', color='y')
    # plt.xlabel('x')
    # plt.ylabel('z')
    # plt.axis('equal')
    # plt.show()

    '''Root height'''
    root_y = positions[:, 0, 1:2]

    '''Root rotation and linear velocity'''
    # (seq_len-1, 1) rotation velocity along y-axis
    # (seq_len-1, 2) linear velovity on xz plane
    r_velocity = np.arcsin(r_velocity[:, 2:3])
    l_velocity = velocity[:, [0, 2]]
    #     print(r_velocity.shape, l_velocity.shape, root_y.shape)
    root_data = np.concatenate([r_velocity, l_velocity, root_y[:-1]], axis=-1)

    '''Get Joint Rotation Representation'''
    # (seq_len, (joints_num-1) *6) quaternion for skeleton joints
    rot_data = cont_6d_params[:, 1:].reshape(len(cont_6d_params), -1)

    '''Get Joint Rotation Invariant Position Represention'''
    # (seq_len, (joints_num-1)*3) local joint position
    ric_data = positions[:, 1:].reshape(len(positions), -1)

    '''Get Joint Velocity Representation'''
    # (seq_len-1, joints_num*3)
    local_vel = qrot_np(np.repeat(r_rot[:-1, None], global_positions.shape[1], axis=1),
                        global_positions[1:] - global_positions[:-1])
    local_vel = local_vel.reshape(len(local_vel), -1)

    data = root_data
    data = np.concatenate([data, ric_data[:-1]], axis=-1)
    data = np.concatenate([data, rot_data[:-1]], axis=-1)
    #     print(data.shape, local_vel.shape)
    data = np.concatenate([data, local_vel], axis=-1)
    data = np.concatenate([data, feet_l, feet_r], axis=-1)

    return data, global_positions, positions, l_velocity


def uniform_skeleton(positions, target_offset):
    src_skel = Skeleton(n_raw_offsets, kinematic_chain, 'cpu')
    src_offset = src_skel.get_offsets_joints(torch.from_numpy(positions[0]))
    src_offset = src_offset.numpy()
    tgt_offset = target_offset.numpy()
    # print(src_offset)
    # print(tgt_offset)
    '''Calculate Scale Ratio as the ratio of legs'''
    src_leg_len = np.abs(src_offset[l_idx1]).max() + np.abs(src_offset[l_idx2]).max()
    tgt_leg_len = np.abs(tgt_offset[l_idx1]).max() + np.abs(tgt_offset[l_idx2]).max()

    scale_rt = tgt_leg_len / src_leg_len
    # print(scale_rt)
    src_root_pos = positions[:, 0]
    tgt_root_pos = src_root_pos * scale_rt

    '''Inverse Kinematics'''
    quat_params = src_skel.inverse_kinematics_np(positions, face_joint_indx)
    # print(quat_params.shape)

    '''Forward Kinematics'''
    src_skel.set_offset(target_offset)
    new_joints = src_skel.forward_kinematics_np(quat_params, tgt_root_pos)
    return new_joints


def generate_motion_representation(data_dir: str, dataset: str):
    
    # if __name__ == "__main__":
    if dataset == "train":
        example_id = "KIT/KIT/4/WalkInCounterClockwiseCircle06_poses_0"
    elif dataset == "val":
        example_id = "KIT/KIT/4/WalkInCounterClockwiseCircle10_poses_0"
    else:
        example_id = "KIT/KIT/11/WalkInCounterClockwiseCircle10_poses_0"
    # Lower legs
    l_idx1, l_idx2 = 5, 8
    # Right/Left foot
    fid_r, fid_l = [8, 11], [7, 10]
    # Face direction, r_hip, l_hip, sdr_r, sdr_l
    face_joint_indx = [2, 1, 17, 16]
    # l_hip, r_hip
    r_hip, l_hip = 2, 1
    joints_num = 22
    # ds_num = 8
    # data_dir = "/lsdf/data/activity/MuscleSim/mint_chat/joints/val/segmented_joints"

    save_dir1 = data_dir.replace("segmented_joints", "new_joints_comp")
    save_dir2 = data_dir.replace("segmented_joints", "new_joint_comp_vecs")


    os.makedirs(save_dir1, exist_ok=True)
    os.makedirs(save_dir2, exist_ok=True)

    global n_raw_offsets
    n_raw_offsets = torch.from_numpy(t2m_raw_offsets)
    global kinematic_chain
    kinematic_chain = t2m_kinematic_chain


    # Get offsets of target skeleton
    try:
        example_data = np.load(os.path.join(data_dir, example_id + ".npy"))
    except FileNotFoundError:
        print("Example data not found.")
        return
    example_data = example_data.reshape(len(example_data), -1, 3)
    example_data = torch.from_numpy(example_data)
    tgt_skel = Skeleton(n_raw_offsets, kinematic_chain, "cpu")
    # (joints_num, 3)
    global tgt_offsets
    tgt_offsets = tgt_skel.get_offsets_joints(example_data[0])
    print(tgt_offsets)


    import glob

    source_list = glob.glob(os.path.join(data_dir, "**/*.npy"), recursive=True)
    glob.glob(os.path.join(data_dir, "**/*.npy"), recursive=True)
    print(source_list[0])
    print(len(source_list))


    frame_num = 0
    for source_file in tqdm(source_list):
        source_data = np.load(os.path.join(source_file))[:, :joints_num]
        target_file = os.path.normpath(source_file).split(os.sep)

        target_file = target_file[-5:]

        target_file[0] = save_dir1
        dirs_1 = pjoin(*target_file[:-1])

        target_file2 = target_file[:]

        target_file2[0] = save_dir2
        dirs_2 = pjoin(*target_file2[:-1])

        os.makedirs(dirs_1, exist_ok=True)
        os.makedirs(dirs_2, exist_ok=True)

        try:
            data, ground_positions, positions, l_velocity = process_file(source_data, 0.002)
            rec_ric_data = recover_from_ric(torch.from_numpy(data).unsqueeze(0).float(), joints_num)
            np.save(pjoin(*target_file), rec_ric_data.squeeze().numpy())
            np.save(pjoin(*target_file2), data)
            frame_num += data.shape[0]
        except Exception as e:
            print(source_file)
            print(e)
    #         print(source_file)
    #         break

    print("Total clips: %d, Frames: %d, Duration: %fm" % (len(source_list), frame_num, frame_num / 20 / 60))


def segment_motions(data_root: str, save_dir: str, csv_file: str, data, dataset="val"):

    save_dir = osp.join(data_root, dataset, save_dir)
    os.makedirs(save_dir, exist_ok=True)
    csv_file = osp.join(data_root, dataset, csv_file)

    header = ["path", "start_frame", "end_frame"]
    # delete the csv file if it already exists
    if os.path.exists(csv_file):
        os.remove(csv_file)
    # create the csv file
    with open(csv_file, "w") as f:
        f.write(",".join(header) + "\n")

    for mint_data in tqdm(
        data, desc=f"Generating segments for {dataset} dataset"
    ):
        try:
            # get the source path of the HumanML3D sample
            source_path = mint_data.get_humanml3d_source_path(data_root)

            # get gaps and valid indices
            gaps = mint_data.get_gaps(as_frame=True, target_fps=20)
            valid_indices = mint_data.get_valid_indices(
                target_fps=20, as_time=False
            )

            data = np.load(source_path)

            segments_list = []
            segments_list_m = []

            # take only the valid indices from the data and save the begin and end of the sequence
            if not mint_data.has_gap:
                segments_list = segment_data(
                    data, valid_indices, segments_list, segments_list_m
                )

            else:
                start_frame = valid_indices[0]
                for i, gap in enumerate(gaps):
                    segments_list = segment_data(
                        data,
                        valid_indices[start_frame : gap[0]],
                        segments_list,
                        segments_list_m,
                    )

                    start_frame = gap[1]

                    # if last gap
                    if i == len(gaps) - 1:
                        end_frame = valid_indices[-1]
                        segments_list = segment_data(
                            data,
                            valid_indices[start_frame:end_frame],
                            segments_list,
                            segments_list_m,
                        )
                        break

            sequence_save_dir = osp.join(
                save_dir, mint_data.subdataset_path, mint_data.subject
            )
            os.makedirs(sequence_save_dir, exist_ok=True)

            # dump each pkl in the segment list into an own file and name it accordingly
            for i, segment, segment_m in zip(
                range(len(segments_list)), segments_list, segments_list_m
            ):
                npy_path = osp.join(
                    sequence_save_dir, mint_data.sequence + f"_{i}" + ".npy"
                )

                np.save(npy_path, segment["data"])

                npy_path_m = osp.join(
                    sequence_save_dir,
                    "M_" + mint_data.sequence + f"_{i}" + ".npy",
                )

                np.save(npy_path_m, segment_m["data"])

                rel_pkl_path = osp.relpath(npy_path, save_dir)
                rel_pkl_path_m = osp.relpath(npy_path_m, save_dir)

                # write to csv file
                with open(csv_file, "a") as f:
                    f.write(
                        f"{rel_pkl_path},{segment['start_frame']},{segment['end_frame']}\n"
                    )
                    f.write(
                        f"{rel_pkl_path_m},{segment_m['start_frame']},{segment_m['end_frame']}\n"
                    )

        except Exception as e:
            print(f"Error processing {mint_data.data_path}: {e}")
        
    generate_motion_representation(save_dir, dataset)



def segment_data(
    data,
    valid_indices,
    segments_list,
    segments_list_m,
    segment_length=41,
    overlap=31,
):
    """
    Segment the data into segments of segment_length.

    Args:
        data (numpy.ndarray): The input data.
        valid_indices (list): The list of valid indices.
        segments_list (list): The list to store the segmented data.
        segment_length (int, optional): The length of each segment. Defaults to 41 (~2 seconds at 20 fps).
        overlap (int, optional): The overlap between segments. Defaults to 20 (~1 second at 20 fps).

    Returns:
        list: The list of segmented data.

    """
    # Divide the indices into overlapping segments
    segments = [
        valid_indices[i : i + segment_length]
        for i in range(
            0, len(valid_indices) - segment_length + 1, segment_length - overlap
        )
    ]

    if segments == []:
        return []

    # Discard the last segment if its length is less than segment_length
    if len(segments[-1]) < segment_length:
        segments = segments[:-1]

    # Append the segments to the list
    for segment in segments:
        seg_data = data[segment]
        seg_data[..., 0] *= -1
        seg_data_m = swap_left_right(seg_data)

        start_frame = segment[0]
        end_frame = segment[-1]

        seg_dict = {
            "data": seg_data,
            "start_frame": start_frame,
            "end_frame": end_frame,
        }
        seg_dict_m = {
            "data": seg_data_m,
            "start_frame": start_frame,
            "end_frame": end_frame,
        }

        segments_list.append(seg_dict)
        segments_list_m.append(seg_dict_m)

    return segments_list