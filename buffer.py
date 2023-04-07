import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import get_dataset, get_network, get_daparam,\
    TensorDataset, epoch, ParamDiffAug, epoch_mimic
import copy
import time
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main(args):

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()

    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)

    # print('\n================== Exp %d ==================\n '%exp)
    print('Hyper-parameters: \n', args.__dict__)

    save_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet":
        save_dir = os.path.join(save_dir, args.subset, str(args.res))
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        save_dir += "_NO_ZCA"
    save_dir = os.path.join(save_dir, args.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    ''' organize the real dataset '''
    # images_all = []
    # labels_all = []
    # indices_class = [[] for c in range(num_classes)]
    # print("BUILDING DATASET")
    # cnt = 0
    # for i in tqdm(range(len(dst_train))):
    #     cnt += 1
    #     if cnt >= 1000:
    #         break
    #     sample = dst_train[i]
    #     images_all.append(torch.unsqueeze(sample[0], dim=0))
    #     labels_all.append(class_map[torch.tensor(sample[1]).item()])

    # for i, lab in tqdm(enumerate(labels_all)):
    #     indices_class[lab].append(i)
    # images_all = torch.cat(images_all, dim=0).to("cpu")
    # labels_all = torch.tensor(labels_all, dtype=torch.long, device="cpu")

    # for c in range(num_classes):
    #     print('class c = %d: %d real images'%(c, len(indices_class[c])))

    # for ch in range(channel):
    #     print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

    criterion = nn.CrossEntropyLoss().to(args.device)

    trajectories = []

    # dst_train = TensorDataset(copy.deepcopy(images_all.detach()), copy.deepcopy(labels_all.detach()))
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=2)

    ''' set augmentation for whole-dataset training '''
    args.dc_aug_param = get_daparam(args.dataset, args.model, args.model, None)
    args.dc_aug_param['strategy'] = 'crop_scale_rotate'  # for whole-dataset training
    print('DC augmentation parameters: \n', args.dc_aug_param)

    for it in range(0, args.num_experts):

        wandb.init(sync_tensorboard=False,
                entity='tongchen',
                project="mtt-buffer-{}-{}-lr={}-l2={}-mom={}-ep{}".format(args.dataset, args.model, args.lr_teacher, args.l2, args.mom, args.train_epochs),
                name='r50-{}-{}'.format(args.dataset, it)
            #    name='test'
               )

        ''' Train synthetic data '''
        teacher_net = get_network(args.model, channel, num_classes, im_size, no_frz=args.no_frz).to(args.device) # get a random model
        # if torch.cuda.device_count() > 1: #如果有多个GPU，将模型并行化，用DataParallel来操作。这个过程会将key值加一个"module. ***"
        #     teacher_net = nn.DataParallel(teacher_net)
        # for name, module in teacher_net.named_modules():
        #     print(name, module)
        teacher_net.train()
        lr = args.lr_teacher
        teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)  # optimizer_img for synthetic data
        # teacher_optim = torch.optim.Adam(teacher_net.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=1e-4)
        scheduler = CosineAnnealingLR(teacher_optim, T_max=len(dst_train), eta_min=0)
        teacher_optim.zero_grad()

        timestamps = []

        timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

        lr_schedule = [args.train_epochs // 2 + 1]

        for e in range(args.train_epochs):
            wandb.log({"Progress": e}, step=e)

            if args.dataset.startswith('MIMIC'):
                start = time.time()
                train_loss, train_bag_class_auc, train_mean = epoch_mimic("train", dataset=dst_train, dataloader=trainloader, net=teacher_net, optimizer=teacher_optim,
                                        scheduler=scheduler, criterion=criterion, args=args, aug=False)
                epoch_time = time.time() - start
                print("Itr: {}\tEpoch: {}\tTime: {}\tTrain mean_bag_class_auc: {}".format(it, e, epoch_time, train_mean))

                start = time.time()
                test_loss, test_bag_class_auc, test_mean = epoch_mimic("test", dataset=dst_test, dataloader=testloader, net=teacher_net, optimizer=teacher_optim,
                                        scheduler=scheduler, criterion=criterion, args=args, aug=False)
                epoch_time = time.time() - start
                print("Itr: {}\tEpoch: {}\tTime: {}\tTest mean_bag_class_auc: {}".format(it, e, epoch_time, test_mean))
                
                print(train_bag_class_auc)
                print(test_bag_class_auc)

                wandb.log({'Train Loss': train_loss}, step=e)
                wandb.log({'Train mean_bag_auc': train_mean}, step=e)
                # for i in range(len(train_bag_class_auc)):
                #     wandb.log({'bag_auc_{}'.format(i): train_bag_class_auc[i]}, step=e)
                
                wandb.log({'Test Loss': test_loss}, step=e)
                wandb.log({'Test mean_bag_auc': test_mean}, step=e)
                for i in range(len(test_bag_class_auc)):
                    wandb.log({'bag_auc_{}'.format(i): test_bag_class_auc[i]}, step=e)
                
                wandb.log({'lr': lr}, step=e)
            
            else:
                train_loss, train_acc, train_auc = epoch("train", dataloader=trainloader, net=teacher_net, optimizer=teacher_optim,
                                            criterion=criterion, args=args, aug=False)

                test_loss, test_acc, test_auc = epoch("test", dataloader=testloader, net=teacher_net, optimizer=None,
                                            criterion=criterion, args=args, aug=False)

                print("Itr: {}\tEpoch: {}\tTrain Acc: {}\tTrain Auc: {}\tTest Acc: {}\tTest Auc: {}".format(it, e, train_acc, train_auc, test_acc, test_auc))

                wandb.log({'Train Loss': train_loss}, step=e)
                wandb.log({'Train Acc': train_acc}, step=e)
                wandb.log({'Train Auc': train_auc}, step=e)
                # wandb.log({'Test Loss': test_loss}, step=e)
                # wandb.log({'Test Acc': test_acc}, step=e)
                # wandb.log({'Test Auc': test_auc}, step=e)
                wandb.log({'lr': lr}, step=e)

            timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])

            if e in lr_schedule and args.decay:
                lr *= 0.1
                teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)
                teacher_optim.zero_grad()
            
            # if e == 49:
            #     dict_dir = os.path.join('./ckpt', wandb.run.project)
            #     if not os.path.exists(dict_dir):
            #         os.makedirs(dict_dir)
            #     torch.save(teacher_net.state_dict(), os.path.join(dict_dir, 'dict_{}.pth'.format(it)))

        trajectories.append(timestamps)

        if len(trajectories) == args.save_interval:
            n = 0
            while os.path.exists(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))):
                n += 1
            print("Saving {}".format(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))))
            torch.save(trajectories, os.path.join(save_dir, "replay_buffer_{}.pt".format(n)))
            trajectories = []
        
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--subset', type=str, default='imagenette', help='subset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--num_experts', type=int, default=100, help='training iterations')
    parser.add_argument('--lr_teacher', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real loader')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--train_epochs', type=int, default=50)
    parser.add_argument('--zca', action='store_true')
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--mom', type=float, default=0, help='momentum')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization')
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--no_frz', action='store_true')

    args = parser.parse_args()
    main(args)

