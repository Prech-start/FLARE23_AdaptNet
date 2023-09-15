import argparse
import os
import numpy as np
import SimpleITK as sitk
import pickle
from skimage.transform import resize


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


def convert(pseudo_paths, nnunet_npy_paths):
    '''

    :param pseudo_paths: the path of pseudo labels, all the file must be ending with '*.nii.gz'.
    :param nnunet_npy_paths: the path of nnunet's basepath/nnUNet_preprocessed/Task098_FLARE2023/nnUNetData_plans_v2.1_stage1, it depends on which of the nnunet data you are using
    :return:
    '''
    # 1. input data convert to npy
    #       converted by nnunet already
    # 2. pseudo label convert into npy
    pseudo_path_list = get_all_file_paths(pseudo_paths)
    npy_path_list = get_all_file_paths(nnunet_npy_paths)
    assert len(pseudo_path_list) == len(npy_path_list), 'length for pseudo labels must be same as nii files'
    pseudo_path_list.sort()
    npy_path_list.sort()
    for p_path, npy_path in zip(pseudo_paths, nnunet_npy_paths):
        # load npy
        if not os.path.exists(npy_path.replace('.npz', '.npy')):
            images = np.load(npy_path, allow_pickle=True)
        else:
            images = np.load(npy_path.replace('.npz', '.npy'), allow_pickle=True)
        # load nii
        pseudo_array = sitk.GetArrayFromImage(sitk.ReadImage(p_path)).astype(np.float32)
        # load info
        properties = load_pickle(npy_path.replace('.npz', '.pkl'))
        # crop array
        crop_bbox = properties['crop_bbox']
        pseudo_array = pseudo_array[slice(crop_bbox)]
        # resample pseudo_array
        size_after_resampling = properties['size_after_resampling']
        pseudo_array = resize(pseudo_array, size_after_resampling, order=3, preserve_range=True,
                              anti_aliasing=False)
        # concate
        all_data = np.concatenate((images[0], images[1], pseudo_array), axis=0)
        # save to npy
        np.save(npy_path.replace('.npz', '.npy'), images)
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
            all_locs = np.argwhere(all_data[-1] == c or all_data[-2] == c)
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
    argps.add_argument('PSEUDO_LABEL_PATH')
    argps.add_argument('NNUNET_npy_PATH')

    arg_s = argps.parse_args()

    convert(pseudo_paths=arg_s['PSEUDO_LABEL_PATH'], nnunet_npy_paths=arg_s['NNUNET_npy_PATH'])
