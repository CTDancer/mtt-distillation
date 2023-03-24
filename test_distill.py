import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
import tqdm
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug, TensorDataset, epoch
import wandb
import copy
import random
import time
from reparam_module import ReparamModule

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):

    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)

    args.im_size = im_size
    # args.lr_net = [1e-4, 0.005532, 0.008318, 0.009999, 0.01087, 0.01172]

    for i in range(0, 5001, 1000):

        wandb.init(sync_tensorboard=False,
                    project='TestDistill',
                    entity="tongchen",
                    name='CRC1-'+args.pix_init+'-ipc_{}-max_start_epoch_{}-syn_steps_{}-expert_epochs{}-lr_teacher_{}-lr_lr_{}-lr_img_{}'.format(args.ipc, args.max_start_epoch, args.syn_steps, args.expert_epochs, args.lr_teacher, args.lr_lr, args.lr_img),
                    # name='test',
                    config=args,
                )

        net_eval = get_network(args.model, channel, num_classes, im_size).to(args.device) # get a random model

        save_dir = os.path.join(".", "logged_files", args.dataset, wandb.run.project)
        img = torch.load(os.path.join(save_dir, "images_{}.pt".format(i)))
        labels = torch.load(os.path.join(save_dir, "labels_{}.pt".format(i)))
        lr = torch.load(os.path.join(save_dir, "lr_{}.pt".format(i)))

        net = net_eval.to(args.device)
        images_train = img.to(args.device)
        labels_train = labels.to(args.device)
        # lr = float(args.lr_net[i//1000])
        Epoch = int(args.epoch_eval_train)
        lr_schedule = [Epoch//2+1]
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

        criterion = nn.CrossEntropyLoss().to(args.device)

        dst_train = TensorDataset(images_train, labels_train)
        trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)

        start = time.time()
        acc_train_list = []
        loss_train_list = []

        for ep in tqdm.tqdm(range(Epoch+1)):
            loss_train, acc_train = epoch('eval_train', trainloader, net, optimizer, criterion, args, aug=False, texture=False)
            acc_train_list.append(acc_train)
            loss_train_list.append(loss_train)
            wandb.log({'Train_Acc': acc_train}, step=ep)
            wandb.log({'Train_Loss': loss_train}, step=ep)
            wandb.log({'Lr': lr}, step=ep)

            if ep % (Epoch//10) == 0:
                with torch.no_grad():
                    loss_test, acc_test, auc_test = epoch('test', testloader, net, optimizer, criterion, args, aug=False)
                wandb.log({'Test_Acc': acc_test}, step=ep)
                wandb.log({'Test_Auc': auc_test}, step=ep)
                wandb.log({'Test_Loss': loss_test}, step=ep)

            # if ep == Epoch:
            #     with torch.no_grad():
            #         loss_test, acc_test, auc_test = epoch('test', testloader, net, optimizer, criterion, args, aug=False)

            if ep in lr_schedule:
                lr *= 0.1
                optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)


        time_train = time.time() - start

        print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f, test auc = %.4f' % (get_time(), i, Epoch, int(time_train), loss_train, acc_train, acc_test, auc_test))

        wandb.finish()
    

    pass






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')

    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')

    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')

    parser.add_argument('--model', type=str, default='ConvNet', help='model')

    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')

    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode, check utils.py for more info')

    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')

    parser.add_argument('--eval_it', type=int, default=100, help='how often to evaluate')

    parser.add_argument('--epoch_eval_train', type=int, default=1000, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=5000, help='how many distillation steps to perform')

    parser.add_argument('--lr_img', type=float, default=1000, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-05, help='learning rate for updating... learning rate')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='initialization for synthetic learning rate')

    parser.add_argument('--lr_init', type=float, default=0.01, help='how to init lr (alpha)')

    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=None, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')

    parser.add_argument('--pix_init', type=str, default='real', choices=["noise", "real"],
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')

    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')

    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')

    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')

    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')

    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")

    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")

    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')


    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')

    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')

    args = parser.parse_args()

    main(args)