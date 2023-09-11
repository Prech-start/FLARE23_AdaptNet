import torch.nn as nn
from network_architecture.generic_UNet import Generic_UNet
import pickle
import torch
from collections import OrderedDict


def load_pickle(file: str, mode: str = 'rb+'):
    with open(file, mode) as f:
        a = pickle.load(f)
        f.close()
    return a


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.Conv2d) or isinstance(module,
                                                                                        nn.ConvTranspose2d) or isinstance(
            module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


class my_Trainner():
    def __init__(self):
        self.base_num_features = self.num_classes = self.net_num_pool_op_kernel_sizes = self.conv_per_stage \
            = self.net_conv_kernel_sizes = None
        self.stage = 1
        self.network = None
        self.initial_par()

    def initial_par(self):
        plans_file, fold, output_folder, dataset_directory, batch_dice, stage, unpack_data, deterministic, fp16 = \
        load_pickle(
            '/media/ljc/data/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task098_FLARE2023/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/model_best.model.pkl')[
            'init']
        plans = load_pickle(plans_file)
        stage_plans = plans['plans_per_stage'][stage]
        self.net_pool_per_axis = stage_plans['num_pool_per_axis']
        self.base_num_features = plans['base_num_features']
        self.num_classes = plans['num_classes'] + 1  # background is no longer in num_classes
        if 'pool_op_kernel_sizes' not in stage_plans.keys():
            assert 'num_pool_per_axis' in stage_plans.keys()
            print(
                "WARNING! old plans file with missing pool_op_kernel_sizes. Attempting to fix it...")
            self.net_num_pool_op_kernel_sizes = []
            for i in range(max(self.net_pool_per_axis)):
                curr = []
                for j in self.net_pool_per_axis:
                    if (max(self.net_pool_per_axis) - j) <= i:
                        curr.append(2)
                    else:
                        curr.append(1)
                self.net_num_pool_op_kernel_sizes.append(curr)
        else:
            self.net_num_pool_op_kernel_sizes = stage_plans['pool_op_kernel_sizes']

        if "conv_per_stage" in plans.keys():  # this ha sbeen added to the plans only recently
            self.conv_per_stage = plans['conv_per_stage']
        else:
            self.conv_per_stage = 2

        if 'conv_kernel_sizes' not in stage_plans.keys():
            print(
                "WARNING! old plans file with missing conv_kernel_sizes. Attempting to fix it...")
            self.net_conv_kernel_sizes = [[3] * len(self.net_pool_per_axis)] * (max(self.net_pool_per_axis) + 1)
        else:
            self.net_conv_kernel_sizes = stage_plans['conv_kernel_sizes']

    def network_initial(self):
        conv_op = nn.Conv3d
        dropout_op = nn.Dropout3d
        norm_op = nn.InstanceNorm3d

        norm_op_kwargs = {'eps': 1e-5, 'affine': True}
        dropout_op_kwargs = {'p': 0, 'inplace': True}
        net_nonlin = nn.LeakyReLU
        net_nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        self.network = Generic_UNet(1, self.base_num_features, self.num_classes,
                                    len(self.net_num_pool_op_kernel_sizes),
                                    self.conv_per_stage, 2, conv_op, norm_op, norm_op_kwargs, dropout_op,
                                    dropout_op_kwargs,
                                    net_nonlin, net_nonlin_kwargs, True, False, lambda x: x, InitWeights_He(1e-2),
                                    self.net_num_pool_op_kernel_sizes, self.net_conv_kernel_sizes, False, True, True)

    def load_best_checkpoint(self, fname):
        checkpoint = torch.load(fname, map_location=torch.device('cpu'))
        curr_state_dict_keys = list(self.network.state_dict().keys())
        new_state_dict = OrderedDict()
        for k, value in checkpoint['state_dict'].items():
            key = k
            if key not in curr_state_dict_keys and key.startswith('module.'):
                key = key[7:]
            new_state_dict[key] = value
        self.network.load_state_dict(new_state_dict)  # todo jichao params not match

    def get_network(self):
        self.network_initial()
        self.load_best_checkpoint(
            fname='/media/ljc/data/DATASET/nnUNet_trained_models/nnUNet/3d_fullres/Task098_FLARE2023/nnUNetTrainerV2__nnUNetPlansv2.1/fold_0/model_best.model')
        return self.network

if __name__ == '__main__':
    trainner = my_Trainner()
    model = trainner.get_network()
    model.eval()
