import argparse
import os
import numpy as np
import SimpleITK as sitk
import pickle
from skimage.transform import resize


def get_bbox_from_mask_b(nonzero_mask, outside_value):
    mask_voxel_coords = np.where(nonzero_mask != outside_value)
    minzidx = int(np.min(mask_voxel_coords[0]))
    maxzidx = int(np.max(mask_voxel_coords[0])) + 1
    minxidx = int(np.min(mask_voxel_coords[1]))
    maxxidx = int(np.max(mask_voxel_coords[1])) + 1
    minyidx = int(np.min(mask_voxel_coords[2]))
    maxyidx = int(np.max(mask_voxel_coords[2])) + 1
    bbox = [[minzidx, maxzidx], [minxidx, maxxidx], [minyidx, maxyidx]]

    return bbox


def get_all_file_paths(folder_path, P='.npz'):
    file_paths = []
    # 遍历文件夹中的所有文件和文件夹
    for root, dirs, files in os.walk(folder_path):
        # 遍历当前文件夹中的所有文件
        for file in files:
            if file.endswith(P):
                file_path = os.path.join(root, file)
                file_paths.append(file_path)
    return sorted(file_paths, key=lambda x: x.split(os.sep)[-1].split('.')[0])


def load_pickle(path):
    return pickle.load(open(path, 'rb+'))


def save_pickle(obj, path):
    pickle.dump(obj, open(path, 'wb+'))


def resample_data(ori_array, ori_spacing,
                  target_spacing=None, only_z=False):
    # shape c w h d
    # spacing_nnunet = [1.8532123022052305, 1.512973664256994, 1.512973664256994]
    if target_spacing is None:
        target_spacing = [2.2838839, 1.8709113, 1.8709113]
    if only_z:
        target_spacing = [target_spacing[0], ori_spacing[0], ori_spacing[1]]
    ori_shape = ori_array.shape[1:]
    target_shape = [ori_spacing[i] * ori_shape[i] / target_spacing[i] // 1 for i in
                    range(len(ori_shape))]
    reshaped_data = []
    reshaped_data.append(resize(ori_array[0], target_shape, order=3)[None])
    reshaped_data.append(resize(ori_array[1], target_shape, order=0, preserve_range=True,
                                anti_aliasing=False)[None])
    reshaped_data.append(resize(ori_array[-1], target_shape, order=0, preserve_range=True,
                                anti_aliasing=False)[None])
    return np.vstack(reshaped_data), target_spacing


def convert(pseudo_paths, nnunet_npy_paths, CT_paths, label_paths):
    '''

    :param pseudo_paths: the path of pseudo labels, all the file must be ending with '*.nii.gz'.
    :param nnunet_npy_paths: the path of nnunet's basepath/nnUNet_preprocessed/Task098_FLARE2023/nnUNetData_plans_v2.1_stage1, it depends on which of the nnunet data you are using
    :return:
    '''
    # 1. input data convert to npy
    #       converted by nnunet already
    # 2. pseudo label convert into npy
    pseudo_path_list = get_all_file_paths(pseudo_paths, '.nii.gz')
    CT_path_list = get_all_file_paths(CT_paths, '.nii.gz')
    label_path_list = get_all_file_paths(label_paths, '.nii.gz')
    npy_path_list = get_all_file_paths(nnunet_npy_paths)
    assert len(pseudo_path_list) == len(npy_path_list) == len(npy_path_list) == len(
        label_path_list), 'length for pseudo labels must be same as nii files'
    for p_path, npy_path, CT_path, label_path in zip(pseudo_path_list, npy_path_list, CT_path_list, label_path_list):
        # load npy
        if not os.path.exists(npy_path.replace('.npz', '.npy')):
            images = np.load(npy_path, allow_pickle=True)
        else:
            images = np.load(npy_path.replace('.npz', '.npy'), allow_pickle=True)

        # load nii
        pseudo_array = sitk.GetArrayFromImage(sitk.ReadImage(p_path)).astype(np.float32)
        CT_array = sitk.GetArrayFromImage(sitk.ReadImage(CT_path)).astype(np.float32)
        label_array = sitk.GetArrayFromImage(sitk.ReadImage(label_path)).astype(np.float32)

        # load info
        properties = load_pickle(npy_path.replace('.npz', '.pkl'))

        # update the crop bbox
        crop_bbox = get_bbox_from_mask_b((pseudo_array + label_array) > 0, outside_value=0)
        sli = slice(crop_bbox[0][0], crop_bbox[0][1]), slice(crop_bbox[1][0], crop_bbox[1][1]), slice(crop_bbox[2][0],
                                                                                                      crop_bbox[2][1])
        # crop three arrays
        properties['crop_bbox'] = crop_bbox
        pseudo_array = pseudo_array[sli]
        CT_array = CT_array[sli]
        label_array = label_array[sli]
        properties['size_after_cropping'] = np.array(pseudo_array.shape)

        # concate
        cropped_data = np.stack((CT_array, label_array, pseudo_array), axis=0)

        # resample array
        resampled_data, current_spacing = resample_data(cropped_data, properties['original_spacing'],
                                                        properties['spacing_after_resampling'])

        # norm one
        ct_array = resampled_data[0].copy()
        if np.max(ct_array) < 1:
            percentile_95 = np.percentile(ct_array, 95)
            percentile_5 = np.percentile(ct_array, 5)
            std = np.std(ct_array)
            mn = np.mean(ct_array)
            ct_array = np.clip(ct_array, a_min=percentile_5, a_max=percentile_95).astype(np.float32)
            ct_array = (ct_array - mn) / std
        else:
            ct_array = np.clip(ct_array, a_min=-160., a_max=240.).astype(np.float32)
            ct_array = (ct_array + 160.) / 400.
        resampled_data[0] = ct_array

        properties['size_after_resampling'] = np.array(resampled_data[0].shape)

        # save to npy
        np.save(npy_path.replace('.npz', '.npy'), resampled_data)
        print('finish combine')

        # 3. location info add into pkl
        num_samples = 10000
        min_percent_coverage = 0.01  # at least 1% of the class voxels need to be selected, otherwise it may be too sparse
        rndst = np.random.RandomState(1234)
        class_locs = {}
        all_classes = range(15)
        for c in all_classes:
            if c == 0:
                continue
            all_locs = np.argwhere(resampled_data[-2 if c in resampled_data[-1] else -1] == c)
            if len(all_locs) == 0:
                class_locs[c] = []
                continue
            target_num_samples = min(num_samples, len(all_locs))
            target_num_samples = max(target_num_samples, int(np.ceil(len(all_locs) * min_percent_coverage)))

            selected = all_locs[rndst.choice(len(all_locs), target_num_samples, replace=False)]
            class_locs[c] = selected
            print(c, target_num_samples)
            properties['class_locations'] = class_locs
        save_pickle(properties, npy_path.replace('.npz', '.pkl'))


if __name__ == '__main__':
    argps = argparse.ArgumentParser()
    # -PSEUDO_LABEL_PATH --NNUNET_npy_PATH
    argps.add_argument('pseudo_path', type=str, help='path for pseudo label, all file must end with .nii.gz!')
    argps.add_argument('nnunet_npy_path', type=str,
                       help='path for npy from nnunet, it must be convert by nnunet first!')
    argps.add_argument('image_tr_path', type=str, help='path for npy from FLARE imageTr')
    argps.add_argument('label_tr_path', type=str, help='path for npy from FLARE labelTr')

    arg_s = argps.parse_args()

    convert(pseudo_paths=arg_s.pseudo_path, nnunet_npy_paths=arg_s.nnunet_npy_path, CT_paths=arg_s.image_tr_path,
            label_paths=arg_s.label_tr_path)
