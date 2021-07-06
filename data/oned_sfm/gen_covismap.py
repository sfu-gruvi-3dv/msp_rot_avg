import data.oned_sfm.sfminit as sfminit
import numpy as np
import copy, sys, argparse, os, glob, pickle

def gen_covis_map(data_dir, iccv_dir):
    input = {'EGs': os.path.join(data_dir, 'EGs.txt'),
             'ccs': os.path.join(data_dir, 'cc.txt'),
             'tracks': os.path.join(data_dir, 'tracks.txt'),
             'coords': os.path.join(data_dir, 'coords.txt'),
             'gt_soln': os.path.join(data_dir, 'gt_bundle.out')}

    iccv_bd_file = glob.glob(os.path.join(iccv_dir, '*.out'))
    if len(iccv_bd_file) != 1:
        print(iccv_bd_file)
        raise Exception('%s ICCV file not found' % data_dir)
    iccv_bd_file = iccv_bd_file[0]

    # tracks = sfminit.Tracks.from_file(input['tracks'])
    # coords = sfminit.Coords.from_file(input['coords'])
    models = sfminit.ModelList.from_EG_file(input['EGs'])
    cc = np.loadtxt(input['ccs'])
    n_CC = cc.shape[0]
    max_cam_idx_ = 0
    cc_reorder_map = dict()
    for i in range(n_CC):
        camera_id = int(cc[i])
        if camera_id > max_cam_idx_:
            max_cam_idx_ = camera_id
        cc_reorder_map[camera_id] = i
    gt_bundle = sfminit.Bundle.from_file(iccv_bd_file)

    # filtering the invalid cameras
    invalid_cam_flag = dict()
    for i, cam in enumerate(gt_bundle.cameras):
        if cam.f == 0:
            invalid_cam_flag[i] = True

    valid_cam2idx = dict()
    valid_cams = list()
    max_idx_ = 0
    n_valid_Cams = 0
    pts = gt_bundle.points
    for pt in pts:
        obs = pt.observations
        obs_cams = [o[0] for o in obs]
        max_idx = np.max(np.asarray(obs_cams))
        if max_idx > max_idx_:
            max_idx_ = max_idx

    if max_idx_ == max_cam_idx_:
        for pt in pts:
            obs = pt.observations
            obs_cams = [o[0] for o in obs]
            for o in obs_cams:
                if o in cc_reorder_map and o not in invalid_cam_flag and o not in valid_cam2idx:
                    valid_cam2idx[o] = n_valid_Cams
                    valid_cams.append(o)
                    n_valid_Cams += 1
    else:
        for pt in pts:
            obs = pt.observations
            obs_cams = [o[0] for o in obs]
            for o in obs_cams:
                ori_o = cc[o]
                if ori_o not in invalid_cam_flag and ori_o not in valid_cam2idx:
                    valid_cam2idx[ori_o] = n_valid_Cams
                    valid_cams.append(ori_o)
                    n_valid_Cams += 1

    print('%s [Valid: %d/%d], max_id: %d/%d' % (data_dir.split('/')[-1], n_valid_Cams, len(gt_bundle.cameras), max_idx_, max_cam_idx_))
    covis_map = np.zeros((n_valid_Cams, n_valid_Cams), dtype=np.int)
    for pt in pts:
        obs = pt.observations
        if max_idx_ == max_cam_idx_:
            obs_cams = [o[0] for o in obs]
        else:
            obs_cams = [cc[o[0]] for o in obs]

        for i in range(len(obs_cams)):
            i_cam_id = obs_cams[i]
            if i_cam_id not in valid_cam2idx:
                continue
            i_idx = valid_cam2idx[i_cam_id]

            for j in range(len(obs_cams)):
                j_cam_id = obs_cams[j]

                if i == j or j_cam_id not in valid_cam2idx:
                    continue
                j_idx = valid_cam2idx[j_cam_id]
                covis_map[i_idx, j_idx] += 1
                covis_map[j_idx, i_idx] += 1

    return covis_map, (valid_cams, valid_cam2idx)


if __name__ == '__main__':
    data_basedir = '/mnt/Exp_2/1dsfm/data/datasets'
    iccv_dir = '/mnt/Tango/pg/ICCV15/'
    datasets = glob.glob(os.path.join(data_basedir, '*'))
    for dataset in datasets:
        if os.path.isdir(dataset):

            dataset_name = dataset.split('/')[-1].strip()
            if dataset_name == 'Trafalgar':
                continue

            iccv_dataset = os.path.join(iccv_dir, dataset_name)

            covis_map, valid_cam_remap = gen_covis_map(dataset, iccv_dataset)
            np.save(os.path.join(dataset, 'cos_vis_map'), covis_map)
            with open(os.path.join(dataset, 'cam_remap'), 'wb') as f:
                pickle.dump(valid_cam_remap, f)
            # np.save(os.path.join(dataset, 'valid_cam_remap'), valid_cam_remap)


