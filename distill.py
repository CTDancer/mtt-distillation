import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
from tqdm import tqdm
from utils import get_dataset, get_network, get_eval_pool, evaluate_synset, get_time, DiffAugment, ParamDiffAug
import wandb
import copy
import random
from torchvision import datasets, transforms
from PIL import Image
from reparam_module import ReparamModule
from torch.optim.lr_scheduler import CosineAnnealingLR
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

torch.cuda.empty_cache()

def main(args):
    
    if args.seed > 0:
        torch.manual_seed(args.seed)

    if args.zca and args.texture:
        raise AssertionError("Cannot use zca and texture together")

    if args.texture and args.pix_init == "real":
        print("WARNING: Using texture with real initialization will take a very long time to smooth out the boundaries between images.")

    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files

    print("CUDNN STATUS: {}".format(torch.backends.cudnn.enabled))

    args.dsa = True if args.dsa == 'True' else False
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()[1:]
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv = get_dataset(args.dataset, args.data_path, args.batch_real, args.subset, args=args)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    im_res = im_size[0]

    args.im_size = im_size

    accs_all_exps = dict() # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    if args.dsa:
        # args.epoch_eval_train = 1000
        args.dc_aug_param = None

    args.dsa_param = ParamDiffAug()

    dsa_params = args.dsa_param
    if args.zca:
        zca_trans = args.zca_trans
    else:
        zca_trans = None

    if args.dataset.startswith('CRC'):
        wandb.init(sync_tensorboard=False,
                project="DatasetDistillation-CRC",
                entity="tongchen",
                name=args.pix_init+'-client{}-ipc_{}-syn_steps_{}-expert_epochs_{}-data_aug_{}-lr_img_{}-seed_{}'.format(args.client, args.ipc, args.syn_steps, args.expert_epochs, args.data_aug, args.lr_img, args.seed),
                    # name='test',
                config=args,
                )
    else:
        wandb.init(sync_tensorboard=False,
                project="DatasetDistillation-MIMIC",
                entity="tongchen",
                name=args.pix_init+'-ipc_{}-syn_steps_{}-data_aug_{}-lr_teacher_{}-lr_lr_{}-lr_img_{}-seed_{}'.format(args.ipc, args.syn_steps, args.data_aug, args.lr_teacher, args.lr_lr, args.lr_img, args.seed),
                    # name='test',
                config=args,
                )

    args = type('', (), {})()

    for key in wandb.config._items:
        setattr(args, key, wandb.config._items[key])

    args.dsa_param = dsa_params
    args.zca_trans = zca_trans

    if args.batch_syn is None:
        args.batch_syn = num_classes * args.ipc

    args.distributed = torch.cuda.device_count() > 1


    print('Hyper-parameters: \n', args.__dict__)
    print('Evaluation model pool: ', model_eval_pool)

    ''' organize the real dataset '''
    def get_images_CRC(c, n):  # get random n images from class c
        train_pth = '/shared/dqwang/datasets/CRC/CRC_DX_train'
        subfolder_pth = os.path.join(train_pth, os.listdir(train_pth)[c])
        file_list = os.listdir(subfolder_pth) # get the list of file names in the folder
        selected_files = random.sample(file_list, n)
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if args.zca:
            transform = transforms.Compose([transforms.ToTensor()])
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        
        images = []
        for image_name in selected_files:
            image_path = os.path.join(subfolder_pth, image_name)
            img = Image.open(image_path)
            image = transform(img)
            images.append(torch.unsqueeze(image, dim=0))
        
        images = torch.cat(images, dim=0).to("cpu")
        
        return images

    def get_images_MIMIC(c, n, dataset):
        # pdb.set_trace()
        candidate_indices = []
        images = []
        data_infos = dataset.data_infos
        for i, data_info in enumerate(data_infos):
            if data_info['gt_label'][c] != 1:
                continue
            candidate_indices.append(i)
        # pdb.set_trace()
        if not candidate_indices:
            return None
        else:
            random.shuffle(candidate_indices)
            for i in candidate_indices[:n]:
                img = Image.open(data_infos[i]['image_path']).convert('RGB')
                if dataset.transform is not None:
                    img = dataset.transform(img)
                images.append(img)
            images = torch.cat(images, dim=0).to("cpu")
            return images



    ''' initialize the synthetic data '''
    if args.dataset.startswith('MIMIC'):
        label_syn = torch.vstack([torch.eye(num_classes, requires_grad=False, device=args.device)[i].repeat(args.ipc, 1) for i in range(num_classes)])
    else:
        label_syn = torch.tensor([np.ones(args.ipc,dtype=np.int_)*i for i in range(num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]

    if args.pix_init == 'noise':
        if args.texture:
            image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0]*args.canvas_size, im_size[1]*args.canvas_size), dtype=torch.float)
        else:
            image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)

    elif args.pix_init == 'mean':
        # Generate synthetic images with initial mean and std
        if args.texture:
            image_syn = [torch.randn(channel, im_size[0]*args.canvas_size, im_size[1]*args.canvas_size) * torch.tensor([0.229, 0.224, 0.225]).view(channel, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(channel, 1, 1) for _ in range(num_classes*args.ipc)]
        else:
            image_syn = [torch.randn(channel, im_size[0], im_size[1]) * torch.tensor([0.229, 0.224, 0.225]).view(channel, 1, 1) + torch.tensor([0.485, 0.456, 0.406]).view(channel, 1, 1) for _ in range(num_classes*args.ipc)]
        image_syn = torch.stack(image_syn)

    syn_lr = torch.tensor(args.lr_teacher).to(args.device)

    if args.pix_init == 'real':
        if args.texture:
            image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0]*args.canvas_size, im_size[1]*args.canvas_size), dtype=torch.float)
        else:
            image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float)
        print('initialize synthetic data from random real images')
        if args.texture:
            for c in range(num_classes):
                for i in range(args.canvas_size):
                    for j in range(args.canvas_size):
                        image_syn.data[c * args.ipc:(c + 1) * args.ipc, :, i * im_size[0]:(i + 1) * im_size[0],
                        j * im_size[1]:(j + 1) * im_size[1]] = torch.cat(
                            [get_images(c, 1).detach().data for s in range(args.ipc)])
        for c in range(num_classes):
            if args.dataset.startswith('CRC'):
                image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images_CRC(c, args.ipc).detach().data
            elif args.dataset.startswith('MIMIC'):
                image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images_MIMIC(c, args.ipc, dst_train).detach().data
    else:
        print('initialize synthetic data from random noise')

    # data augmentation
    if args.data_aug:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        ])
        
        augmented_images = []
        for i in range(image_syn.size(0)): 
            augmented_images.append(transform(image_syn[i]))
            
        image_syn = torch.stack(augmented_images)

    ''' training '''
    image_syn = image_syn.detach().to(args.device).requires_grad_(True)
    syn_lr = syn_lr.detach().to(args.device).requires_grad_(True)
    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_img, momentum=args.img_mom, weight_decay=args.img_wd)
    optimizer_lr = torch.optim.SGD([syn_lr], lr=args.lr_lr, momentum=args.lr_mom, weight_decay=args.lr_wd)
    if args.dataset.startswith('MIMIC'):
        scheduler_img = CosineAnnealingLR(optimizer_img, T_max=args.Iteration, eta_min=0)
        scheduler_lr = CosineAnnealingLR(optimizer_lr, T_max=args.Iteration, eta_min=1e-7)
    else:
        scheduler_img = CosineAnnealingLR(optimizer_img, T_max=args.Iteration, eta_min=0)
        scheduler_lr = CosineAnnealingLR(optimizer_lr, T_max=args.Iteration, eta_min=1e-12)
    optimizer_img.zero_grad()

    criterion = nn.CrossEntropyLoss().to(args.device)
    print('%s training begins'%get_time())

    expert_dir = os.path.join(args.buffer_path, args.dataset)
    if args.dataset == "ImageNet":
        expert_dir = os.path.join(expert_dir, args.subset, str(args.res))
    if args.dataset in ["CIFAR10", "CIFAR100"] and not args.zca:
        expert_dir += "_NO_ZCA"
    expert_dir = os.path.join(expert_dir, args.model)
    print("Expert Dir: {}".format(expert_dir))

    if args.load_all:
        buffer = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))

    else:
        expert_files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        file_idx = 0
        expert_idx = 0
        random.shuffle(expert_files)
        if args.max_files is not None:
            expert_files = expert_files[:args.max_files]
        print("loading file {}".format(expert_files[file_idx]))
        buffer = torch.load(expert_files[file_idx])
        if args.max_experts is not None:
            buffer = buffer[:args.max_experts]
        random.shuffle(buffer)

    best_acc = {m: 0 for m in model_eval_pool}

    best_std = {m: 0 for m in model_eval_pool}

    best_auc = {m: 0 for m in model_eval_pool}


    for it in range(0, args.Iteration+1):
        save_this_it = False

        # writer.add_scalar('Progress', it, it)
        wandb.log({"Progress": it}, step=it)
        ''' Evaluate synthetic data '''
        if it in eval_it_pool:
            for model_eval in model_eval_pool:
                print('-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                if args.dsa:
                    print('DSA augmentation strategy: \n', args.dsa_strategy)
                    print('DSA augmentation parameters: \n', args.dsa_param.__dict__)
                else:
                    print('DC augmentation parameters: \n', args.dc_aug_param)

                if args.dataset == "MIMIC":
                    auc_tests = []
                    auc_trains = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model

                        eval_labs = label_syn
                        with torch.no_grad():
                            image_save = image_syn
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_save.detach()), copy.deepcopy(eval_labs.detach()) # avoid any unaware modification

                        args.lr_net = syn_lr.item()

                        _, auc_test = evaluate_synset(it, it_eval, net_eval, image_syn_eval, label_syn_eval, dst_test, testloader, args, texture=args.texture)
                        auc_tests.append(auc_test)
                        
                    auc_tests = np.array(auc_tests)
                    auc_test_mean = np.mean(auc_tests)
                    auc_test_std = np.std(auc_tests)

                    if auc_test_mean > best_auc[model_eval]:
                        best_auc[model_eval] = auc_test_mean
                        best_std[model_eval] = auc_test_std
                        save_this_it = True
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(auc_tests), model_eval, auc_test_mean, auc_test_std))
                    wandb.log({'AUC/{}'.format(model_eval): auc_test_mean}, step=it)
                    wandb.log({'Std_AUC/{}'.format(model_eval): auc_test_std}, step=it)
                    wandb.log({'Max_AUC/{}'.format(model_eval): best_auc[model_eval]}, step=it)    

                else:
                    accs_test = []
                    accs_train = []
                    aucs_test = []
                    for it_eval in range(args.num_eval):
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(args.device) # get a random model

                        eval_labs = label_syn
                        with torch.no_grad():
                            image_save = image_syn
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_save.detach()), copy.deepcopy(eval_labs.detach()) # avoid any unaware modification

                        args.lr_net = syn_lr.item()
                        _, acc_train, acc_test, auc_test = evaluate_synset(it, it_eval, net_eval, image_syn_eval, label_syn_eval, dst_test, testloader, args, texture=args.texture)
                        accs_test.append(acc_test)
                        accs_train.append(acc_train)
                        aucs_test.append(auc_test)
                    accs_test = np.array(accs_test)
                    accs_train = np.array(accs_train)
                    aucs_test = np.array(aucs_test)

                    acc_test_mean = np.mean(accs_test)
                    acc_test_std = np.std(accs_test)
                    auc_test_mean = np.mean(aucs_test)
                    auc_test_std = np.std(aucs_test)
                    if auc_test_mean > best_acc[model_eval]:
                        best_acc[model_eval] = auc_test_mean
                        best_std[model_eval] = auc_test_std
                        save_this_it = True
                    print('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------'%(len(accs_test), model_eval, auc_test_mean, auc_test_std))
                    wandb.log({'Accuracy/{}'.format(model_eval): acc_test_mean}, step=it)
                    # wandb.log({'Max_Accuracy/{}'.format(model_eval): best_acc[model_eval]}, step=it)
                    wandb.log({'Std_Acc/{}'.format(model_eval): acc_test_std}, step=it)
                    # wandb.log({'Max_Std/{}'.format(model_eval): best_std[model_eval]}, step=it)
                    wandb.log({'Auc/{}'.format(model_eval): auc_test_mean}, step=it)
                    wandb.log({'Max_Auc/{}'.format(model_eval): best_acc[model_eval]}, step=it)
                    wandb.log({'Std_Auc/{}'.format(model_eval): auc_test_std}, step=it)
                    wandb.log({'Max_Std_Auc/{}'.format(model_eval): best_std[model_eval]}, step=it)


        # if it in eval_it_pool and (save_this_it or it % 1000 == 0):
        if (save_this_it or it % 1000 == 0):
            with torch.no_grad():
                image_save = image_syn.cuda()

                save_dir = os.path.join(".", "logged_files", args.dataset, wandb.run.name)

                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)

                torch.save(image_save.cpu(), os.path.join(save_dir, "images_{}.pt".format(it)))
                torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_{}.pt".format(it)))

                if save_this_it:
                    torch.save(image_save.cpu(), os.path.join(save_dir, "images_best.pt".format(it)))
                    torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_best.pt".format(it)))
                    torch.save(syn_lr.item(), os.path.join(save_dir, "lr_best.pt".format(it)))

                wandb.log({"Pixels": wandb.Histogram(torch.nan_to_num(image_syn.detach().cpu()))}, step=it)

                if args.ipc < 50 or args.force_save:
                    upsampled = image_save
                    if args.dataset != "ImageNet":
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                        upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                    grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                    wandb.log({"Synthetic_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                    wandb.log({'Synthetic_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)

                    for clip_val in [2.5]:
                        std = torch.std(image_save)
                        mean = torch.mean(image_save)
                        upsampled = torch.clip(image_save, min=mean-clip_val*std, max=mean+clip_val*std)
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"Clipped_Synthetic_Images/std_{}".format(clip_val): wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)

                    if args.zca:
                        image_save = image_save.to(args.device)
                        image_save = args.zca_trans.inverse_transform(image_save)
                        image_save.cpu()

                        torch.save(image_save.cpu(), os.path.join(save_dir, "images_zca_{}.pt".format(it)))

                        upsampled = image_save
                        if args.dataset != "ImageNet":
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                            upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                        grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                        wandb.log({"Reconstructed_Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))}, step=it)
                        wandb.log({'Reconstructed_Pixels': wandb.Histogram(torch.nan_to_num(image_save.detach().cpu()))}, step=it)

                        for clip_val in [2.5]:
                            std = torch.std(image_save)
                            mean = torch.mean(image_save)
                            upsampled = torch.clip(image_save, min=mean - clip_val * std, max=mean + clip_val * std)
                            if args.dataset != "ImageNet":
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=2)
                                upsampled = torch.repeat_interleave(upsampled, repeats=4, dim=3)
                            grid = torchvision.utils.make_grid(upsampled, nrow=10, normalize=True, scale_each=True)
                            wandb.log({"Clipped_Reconstructed_Images/std_{}".format(clip_val): wandb.Image(
                                torch.nan_to_num(grid.detach().cpu()))}, step=it)

        wandb.log({"Synthetic_LR": syn_lr.detach().cpu()}, step=it)

        student_net = get_network(args.model, channel, num_classes, im_size, dist=False).to(args.device)  # get a random model

        student_net = ReparamModule(student_net)

        if args.distributed:
            student_net = torch.nn.DataParallel(student_net)

        # student_net.train()

        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])

        if args.load_all:
            expert_trajectory = buffer[np.random.randint(0, len(buffer))]
        else:
            expert_trajectory = buffer[expert_idx]
            expert_idx += 1
            if expert_idx == len(buffer):
                expert_idx = 0
                file_idx += 1
                if file_idx == len(expert_files):
                    file_idx = 0
                    random.shuffle(expert_files)
                print("loading file {}".format(expert_files[file_idx]))
                if args.max_files != 1:
                    del buffer
                    buffer = torch.load(expert_files[file_idx])
                if args.max_experts is not None:
                    buffer = buffer[:args.max_experts]
                random.shuffle(buffer)

        start_epoch = np.random.randint(0, args.max_start_epoch)
        starting_params = expert_trajectory[start_epoch]

        target_params = expert_trajectory[start_epoch+args.expert_epochs]
        target_params = torch.cat([p.data.to(args.device).reshape(-1) for p in target_params], 0)

        student_params = [torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0).requires_grad_(True)]

        starting_params = torch.cat([p.data.to(args.device).reshape(-1) for p in starting_params], 0)

        syn_images = image_syn

        y_hat = label_syn.to(args.device)

        param_loss_list = []
        param_dist_list = []
        indices_chunks = []

        for step in range(args.syn_steps):

            if not indices_chunks:
                indices = torch.randperm(len(syn_images))
                indices_chunks = list(torch.split(indices, args.batch_syn))

            these_indices = indices_chunks.pop()


            x = syn_images[these_indices]
            this_y = y_hat[these_indices]

            if args.texture:
                x = torch.cat([torch.stack([torch.roll(im, (torch.randint(im_size[0]*args.canvas_size, (1,)), torch.randint(im_size[1]*args.canvas_size, (1,))), (1,2))[:,:im_size[0],:im_size[1]] for im in x]) for _ in range(args.canvas_samples)])
                this_y = torch.cat([this_y for _ in range(args.canvas_samples)])

            if args.dsa and (not args.no_aug):
                x = DiffAugment(x, args.dsa_strategy, param=args.dsa_param)

            if args.distributed:
                forward_params = student_params[-1].unsqueeze(0).expand(torch.cuda.device_count(), -1)
            else:
                forward_params = student_params[-1]
            x = student_net(**dict(x=x, flat_param=forward_params, buffers=None))
            ce_loss = criterion(x, this_y)

            grad = torch.autograd.grad(ce_loss, student_params[-1], create_graph=True)[0]
            
            tmp = [student_params[-1][:-1026]]
            tmp.append(student_params[-1][-1026:] - syn_lr * grad[-1026:])
            tmp = torch.cat(tmp, 0)
            student_params.append(tmp)


        param_loss = torch.tensor(0.0).to(args.device)
        param_dist = torch.tensor(0.0).to(args.device)

        param_loss += torch.nn.functional.mse_loss(student_params[-1][-1026:], target_params[-1026:], reduction="sum")
        param_dist += torch.nn.functional.mse_loss(starting_params[-1026:], target_params[-1026:], reduction="sum")

        param_loss_list.append(param_loss)
        param_dist_list.append(param_dist)


        param_loss /= num_params
        param_dist /= num_params

        param_loss /= param_dist

        grand_loss = param_loss

        optimizer_img.zero_grad()
        optimizer_lr.zero_grad()

        grand_loss.backward()

        optimizer_img.step()
        scheduler_img.step()
        optimizer_lr.step()
        scheduler_lr.step()

        wandb.log({"Grand_Loss": grand_loss.detach().cpu(),
                   "Start_Epoch": start_epoch})

        for _ in student_params:
            del _

        if it%10 == 0:
            print('%s iter = %04d, loss = %.4f' % (get_time(), it, grand_loss.item()))

    wandb.finish()


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

