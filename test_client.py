from utils import test_client, get_eval_pool, get_network
import argparse
import torch

def main(args):
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    syn_lr = torch.tensor(args.lr_teacher).to(args.device)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    args.lr_net = syn_lr.item()

    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    channel = 3
    im_size = (224, 224)
    num_classes = 2

    for model_eval in model_eval_pool:
        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device)
        test_client(args.ipc, 5, net_eval, '/shared/dqwang/scratch/tongchen/CRC-clients', args)

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

    parser.add_argument('--pix_init', type=str, default='real', choices=["noise", "real", "mean"],
                        help='noise/real/mean: initialize synthetic images from random noise or randomly sampled real images or from the mean and std of CRC.')

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
    
    parser.add_argument('--data_aug', action='store_true', help='this will use data augmentation after initializing synthetic images')

    parser.add_argument('--img_mom', type=float, default=0.5, help='the momentum of optimizer_img')
    parser.add_argument('--lr_mom', type=float, default=0.5, help='the momentum of optimizer_lr')
    parser.add_argument('--img_wd', type=float, default=0, help='the weight decay of optimizer_img')
    parser.add_argument('--lr_wd', type=float, default=0, help='the weight decay of optimize_lr')

    parser.add_argument('--lr_img_decay', type=float, default=1, help='the lr decay of lr_img')
    parser.add_argument('--lr_lr_decay', type=float, default=1, help='the lr decay of lr_lr')

    parser.add_argument('--seed', type=int, default=-1, help='random seed for torch.randn()')

    parser.add_argument('--client', type=int, default=-1, help='the index of client. default is the whole dataset')

    args = parser.parse_args()

    main(args)