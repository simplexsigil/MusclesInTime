import numpy as np
import torch


def amass_to_smpl(amass_sample_path, bm_m, bm_f, num_betas=16):
    ex_fps = 20

    bdata = np.load(amass_sample_path, allow_pickle=True)
    fps = 0
    fps = bdata["mocap_framerate"]
    frame_number = bdata["trans"].shape[0]

    fId = 0
    pose_seq = []
    if bdata["gender"] == "male":
        bm = bm_m
    else:
        bm = bm_f

    down_sample = int(fps / ex_fps)

    bodies = []

    bdata_poses = bdata["poses"][::down_sample, ...]
    bdata_trans = bdata["trans"][::down_sample, ...]
    body_parms = {
        "root_orient": torch.Tensor(bdata_poses[:, :3]).cuda(),
        "pose_body": torch.Tensor(bdata_poses[:, 3:66]).cuda(),
        "pose_hand": torch.Tensor(bdata_poses[:, 66:]).cuda(),
        "trans": torch.Tensor(bdata_trans).cuda(),
        "betas": torch.Tensor(
            np.repeat(bdata["betas"][:num_betas][np.newaxis], repeats=len(bdata_trans), axis=0)
        ).cuda(),
    }

    with torch.no_grad():
        body = bm(**body_parms)

    return body
