# adapted from
# https://github.com/VICO-UoE/DatasetCondensation

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import kornia as K
from tqdm import tqdm
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from scipy.ndimage.interpolation import rotate as scipyrotate
from networks import MLP, ConvNet, LeNet, AlexNet, VGG11BN, VGG11, ResNet18, ResNet18BN_AP, ResNet18_AP, \
    ResNet34, ResNet50

import numpy as np
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score, f1_score
from mmcv.cnn.resnet import ResNet
from collections import defaultdict
import os.path as osp
import random
import wandb
import copy
import pdb

class CRCK(Dataset):
    """USPS Dataset.
    Args:
        root (string): Root directory of dataset where dataset file exist.
        train (bool, optional): If True, use the training split.
        download (bool, optional): If true, downloads the dataset
            from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
        transform (callable, optional): A function/transform that takes in
            an PIL image and returns a transformed version.
            E.g, ``transforms.RandomCrop``
    """

    # url = "https://github.com/mingyuliutw/CoGAN/raw/master/cogan_pytorch/data/uspssample/usps_28x28.pkl"

    def __init__(self, root, ann_file, cls_ind, train=True, transform=None):
        """Init CRCK dataset."""

        self.train = train
        self.ann_file = ann_file
        self.transform = transform
        self.dataset_size = None
        self.ann_list = self.list_from_file(self.ann_file,root)
        self.cls_ind = cls_ind


    def __getitem__(self, index):
        """Get images and target for data loader.
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img_path = self.ann_list[index].split(' ')[0]
        label = self.ann_list[index].split(' ')[1].split(',')[self.cls_ind]
        img = Image.open(img_path).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        label = np.int64(label).item()
        bagname = self.ann_list[index].split(' ')[0][-27:-15]
        if self.train:
            # return img, label
            return img, label, bagname
        else:
            # return img, label
            return img, label, bagname

    def __len__(self):
        """Return size of dataset."""
        return len(self.ann_list)

    def list_from_file(self,ann,pre):
        """Load a text file and parse the content as a list of strings."""
        print(ann)
        item_list = []
        with open(ann,'r') as f:
            for line in f:
                item_list.append(pre+'/'+line.rstrip('\n\r'))
        return item_list

class MIMIC(Dataset):
    def __init__(self, root, ann_file, train=True, transform=None):
        """Init MIMIC dataset."""
        self.train = train
        self.data_prefix = root
        self.ann_file = ann_file
        self.transform = transform
        self.dataset_size = None
        self.subject_infos = {}
        self.data_infos = self.load_annotations()
        self.CLASSES = [str(i) for i in range(14)]

    def __getitem__(self, index):
        img = Image.open(self.data_infos[index]['image_path']).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        gt_label = self.data_infos[index]['gt_label']
        bagname = self.data_infos[index]['bag']

        if self.train:
            # return img, label
            # print(gt_label)
            return img, gt_label, bagname
        else:
            # return img, label
            return img, gt_label, bagname

    def __len__(self):
        """Return size of dataset."""
        return len(self.data_infos)

    def load_annotations(self):
        """
        Load a text file and parse the content as a list of strings.
        Each string is composed of : path + bagname + label.
        """
        data_infos = []
        with open(self.ann_file,'r') as f:
            for line in f:
                filename, class_name = line.strip().split(' ')
                subject_id, study_id, filename = filename.split('_')
                img_path = os.path.join(self.data_prefix, filename)
                bagname = f'{subject_id}_{study_id}'
                gt_label = [int(label) for label in class_name]  # class_to_idx
                info = {
                    'image_path': img_path,
                    'bag': bagname,
                    'gt_label': gt_label,
                    'subject_id': subject_id,
                    'study_id': study_id
                }
                data_infos.append(info)

        for i, info in enumerate(data_infos):
            self.subject_infos[info['bag']] = self.subject_infos.get(info['bag'], []) + [i] 
        return data_infos


class GlobalAveragePooling(nn.Module):
    """Global Average Pooling neck.

    Note that we use `view` to remove extra channel after pooling. We do not
    use `squeeze` as it will also remove the batch dimension when the tensor
    has a batch dimension of size 1, which can lead to unexpected errors.

    Args:
        dim (int): Dimensions of each sample channel, can be one of {1, 2, 3}.
            Default: 2
    """

    def __init__(self, dim=2):
        super(GlobalAveragePooling, self).__init__()
        assert dim in [1, 2, 3], 'GlobalAveragePooling dim only support ' \
            f'{1, 2, 3}, get {dim} instead.'
        if dim == 1:
            self.gap = nn.AdaptiveAvgPool1d(1)
        elif dim == 2:
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
        else:
            self.gap = nn.AdaptiveAvgPool3d((1, 1, 1))

    def init_weights(self):
        pass

    def forward(self, inputs):
        if isinstance(inputs, tuple):
            outs = tuple([self.gap(x) for x in inputs])
            outs = tuple(
                [out.view(x.size(0), -1) for out, x in zip(outs, inputs)])
        elif isinstance(inputs, torch.Tensor):
            outs = self.gap(inputs)
            outs = outs.view(inputs.size(0), -1)
        else:
            raise TypeError('neck inputs should be tuple or torch.tensor')
        return outs

class MyResNet18(nn.Module):
    def __init__(self, ck_path=None, num_classes=2, no_frz=False):
        super(MyResNet18, self).__init__()
        if no_frz:
            self.resnet = ResNet(depth=18, out_indices=(3, ))
        else:
            self.resnet = ResNet(depth=18, out_indices=(3, ), frozen_stages=3)
        if ck_path is not None:
            self.resnet.init_weights(ck_path)
        self.classifier = nn.Linear(512, num_classes, bias=True)
        self.pool = GlobalAveragePooling()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # 3*224*224
        out = self.resnet(x) # 64/128/256/512*7*7
        # out = F.avg_pool2d(out, kernel_size=7) # 512*1
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.classifier(out)
        out = F.softmax(out, dim=-1)
        return out

class MyResNet50(nn.Module):
    def __init__(self, ck_path=None, num_classes=2, no_frz=False):
        super(MyResNet50, self).__init__()
        if no_frz:
            self.resnet = ResNet(depth=50, out_indices=(3, ))
        else:
            self.resnet = ResNet(depth=50, out_indices=(3, ), frozen_stages=3)
        if ck_path is not None:
            self.resnet.init_weights(ck_path)
        self.classifier = nn.Linear(2048, num_classes, bias=True)
        self.pool = GlobalAveragePooling()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # 3*224*224
        out = self.resnet(x) # 64/128/256/512*7*7
        # out = F.avg_pool2d(out, kernel_size=7) # 512*1
        out = self.pool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.classifier(out)
        out = F.softmax(out, dim=-1)
        return out

class Config:
    imagenette = [0, 217, 482, 491, 497, 566, 569, 571, 574, 701]

    # ["australian_terrier", "border_terrier", "samoyed", "beagle", "shih-tzu", "english_foxhound", "rhodesian_ridgeback", "dingo", "golden_retriever", "english_sheepdog"]
    imagewoof = [193, 182, 258, 162, 155, 167, 159, 273, 207, 229]

    # ["tabby_cat", "bengal_cat", "persian_cat", "siamese_cat", "egyptian_cat", "lion", "tiger", "jaguar", "snow_leopard", "lynx"]
    imagemeow = [281, 282, 283, 284, 285, 291, 292, 290, 289, 287]

    # ["peacock", "flamingo", "macaw", "pelican", "king_penguin", "bald_eagle", "toucan", "ostrich", "black_swan", "cockatoo"]
    imagesquawk = [84, 130, 88, 144, 145, 22, 96, 9, 100, 89]

    # ["pineapple", "banana", "strawberry", "orange", "lemon", "pomegranate", "fig", "bell_pepper", "cucumber", "green_apple"]
    imagefruit = [953, 954, 949, 950, 951, 957, 952, 945, 943, 948]

    # ["bee", "ladys slipper", "banana", "lemon", "corn", "school_bus", "honeycomb", "lion", "garden_spider", "goldfinch"]
    imageyellow = [309, 986, 954, 951, 987, 779, 599, 291, 72, 11]

    dict = {
        "imagenette" : imagenette,
        "imagewoof" : imagewoof,
        "imagefruit": imagefruit,
        "imageyellow": imageyellow,
        "imagemeow": imagemeow,
        "imagesquawk": imagesquawk,
    }

config = Config()

def get_dataset(dataset, data_path, batch_size=1, subset="imagenette", args=None):

    class_map = None
    loader_train_dict = None
    class_map_inv = None

    if dataset == 'CIFAR10':
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        if args.zca:
            transform = transforms.Compose([transforms.ToTensor()])
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR10(data_path, train=True, download=True, transform=transform) # no augmentation
        dst_test = datasets.CIFAR10(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes
        class_map = {x:x for x in range(num_classes)}


    elif dataset == 'Tiny':
        channel = 3
        im_size = (64, 64)
        num_classes = 200
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if args.zca:
            transform = transforms.Compose([transforms.ToTensor()])
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.ImageFolder(os.path.join(data_path, "train"), transform=transform) # no augmentation
        dst_test = datasets.ImageFolder(os.path.join(data_path, "val", "images"), transform=transform)
        class_names = dst_train.classes
        class_map = {x:x for x in range(num_classes)}


    elif dataset == 'ImageNet':
        channel = 3
        im_size = (128, 128)
        num_classes = 10

        config.img_net_classes = config.dict[subset]

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if args.zca:
            transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Resize(im_size),
                                        transforms.CenterCrop(im_size)])
        else:
            transform = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std),
                                            transforms.Resize(im_size),
                                            transforms.CenterCrop(im_size)])

        dst_train = datasets.ImageNet(data_path, split="train", transform=transform) # no augmentation
        dst_train_dict = {c : torch.utils.data.Subset(dst_train, np.squeeze(np.argwhere(np.equal(dst_train.targets, config.img_net_classes[c])))) for c in range(len(config.img_net_classes))}
        dst_train = torch.utils.data.Subset(dst_train, np.squeeze(np.argwhere(np.isin(dst_train.targets, config.img_net_classes))))
        loader_train_dict = {c : torch.utils.data.DataLoader(dst_train_dict[c], batch_size=batch_size, shuffle=True, num_workers=16) for c in range(len(config.img_net_classes))}
        dst_test = datasets.ImageNet(data_path, split="val", transform=transform)
        dst_test = torch.utils.data.Subset(dst_test, np.squeeze(np.argwhere(np.isin(dst_test.targets, config.img_net_classes))))
        for c in range(len(config.img_net_classes)):
            dst_test.dataset.targets[dst_test.dataset.targets == config.img_net_classes[c]] = c
            dst_train.dataset.targets[dst_train.dataset.targets == config.img_net_classes[c]] = c
        print(dst_test.dataset)
        class_map = {x: i for i, x in enumerate(config.img_net_classes)}
        class_map_inv = {i: x for i, x in enumerate(config.img_net_classes)}
        class_names = None


    elif dataset.startswith('CIFAR100'):
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]

        if args.zca:
            transform = transforms.Compose([transforms.ToTensor()])
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        dst_train = datasets.CIFAR100(data_path, train=True, download=True, transform=transform)  # no augmentation
        dst_test = datasets.CIFAR100(data_path, train=False, download=True, transform=transform)
        class_names = dst_train.classes
        class_map = {x: x for x in range(num_classes)}
    
    elif dataset.startswith('CRC'):
        channel = 3
        im_size = (224, 224)
        num_classes = 2
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        if args.zca:
            transform = transforms.Compose([transforms.ToTensor()])
        else:
            transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        
        task = int(dataset[3])-1
        class_names = ['MSI', 'MSS']
        CRC_train = '/shared/dqwang/datasets/CRC/CRC_DX_train'
        CRC_test = '/shared/dqwang/datasets/CRC/CRC_DX_test'
        # train_ann_path = '/shared/dqwang/datasets/CRC/annotation/train_ann.txt'
        if args.client >= 0:
            train_ann_path = '/shared/dqwang/datasets/CRC/annotation/federate/split_5_1/train_' + str(args.client) + '.txt'
        else:
            train_ann_path = '/shared/dqwang/datasets/CRC/annotation/patch_split/msi/train_50.txt'
        test_ann_path = '/shared/dqwang/datasets/CRC/annotation/test_ann.txt'
        dst_train = CRCK(CRC_train, train_ann_path, cls_ind=task, train=True, transform=transform) # no augmentation
        dst_test = CRCK(CRC_test, test_ann_path, cls_ind=task, train=False, transform=transform)
        # class_names = dst_train.classes
        class_map = {x: x for x in range(num_classes)}

    elif dataset=='MIMIC':
        channel = 3
        im_size = (224, 224)
        num_classes = 14
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

        if args.zca:
            transform = transforms.Compose([transforms.ToTensor()])
        else:
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
            test_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])

        class_names = [str(i) for i in range(14)]
        MIMIC_train = '/shared/dqwang/scratch/tongchen/MIMIC/train'
        MIMIC_test = '/shared/dqwang/scratch/tongchen/MIMIC/test'
        train_ann_path = '/shared/dqwang/scratch/yunkunzhang/mimic_multi-label_ann/train.txt'
        test_ann_path = '/shared/dqwang/scratch/yunkunzhang/mimic_multi-label_ann/test.txt'
        dst_train = MIMIC(MIMIC_train, train_ann_path, train=True, transform=train_transform)
        dst_test = MIMIC(MIMIC_test, test_ann_path, train=False, transform=test_transform)
        # class_names = dst_train.classes
        class_map = {x: x for x in range(num_classes)}

    elif dataset=='MIMIC_small':
        channel = 3
        im_size = (224, 224)
        num_classes = 14
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]

        if args.zca:
            transform = transforms.Compose([transforms.ToTensor()])
        else:
            train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])
            test_transform = transforms.Compose([
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])

        class_names = [str(i) for i in range(14)]
        MIMIC_train = '/shared/dqwang/scratch/tongchen/MIMIC_small/train'
        MIMIC_test = '/shared/dqwang/scratch/tongchen/MIMIC_small/test'
        train_ann_path = '/shared/dqwang/scratch/tongchen/MIMIC_small/annotation/train_ann.txt'
        test_ann_path = '/shared/dqwang/scratch/tongchen/MIMIC_small/annotation/test_ann.txt'
        dst_train = MIMIC(MIMIC_train, train_ann_path, train=True, transform=train_transform)
        dst_test = MIMIC(MIMIC_test, test_ann_path, train=False, transform=test_transform)
        # class_names = dst_train.classes
        class_map = {x: x for x in range(num_classes)}
    else:
        exit('unknown dataset: %s'%dataset)

    if args.zca:
        if args.dataset.startswith('CRC') or args.dataset.startswith('MIMIC'):
            images = []
            labels = []
            bagnames = []
            print("Train ZCA")
            for i in tqdm(range(len(dst_train))):
                im, lab, bagname = dst_train[i]
                images.append(im)
                labels.append(lab)
                bagnames.append(bagname)
            images = torch.stack(images, dim=0).to(args.device)
            labels = torch.tensor(labels, dtype=torch.long, device="cpu")
            # pdb.set_trace()
            # bagnames = torch.tensor(bagnames)
            zca = K.enhance.ZCAWhitening(eps=0.1, compute_inv=True)
            zca.fit(images)
            zca_images = zca(images).to("cpu")
            dst_train = BagDataset(zca_images, labels, bagnames)

            images = []
            labels = []
            bagnames = []
            print("Test ZCA")
            for i in tqdm.tqdm(range(len(dst_test))):
                im, lab, bagname = dst_test[i]
                images.append(im)
                labels.append(lab)
                bagnames.append(bagname)
            images = torch.stack(images, dim=0).to(args.device)
            labels = torch.tensor(labels, dtype=torch.long, device="cpu")
            # bagnames = torch.tensor(bagnames)
            zca_images = zca(images).to("cpu")
            dst_test = BagDataset(zca_images, labels, bagnames)
        else:
            images = []
            labels = []
            print("Train ZCA")
            for i in tqdm(range(len(dst_train))):
                im, lab = dst_train[i]
                images.append(im)
                labels.append(lab)
            images = torch.stack(images, dim=0).to(args.device)
            labels = torch.tensor(labels, dtype=torch.long, device="cpu")
            zca = K.enhance.ZCAWhitening(eps=0.1, compute_inv=True)
            zca.fit(images)
            zca_images = zca(images).to("cpu")
            dst_train = TensorDataset(zca_images, labels)

            images = []
            labels = []
            print("Test ZCA")
            for i in tqdm.tqdm(range(len(dst_test))):
                im, lab = dst_test[i]
                images.append(im)
                labels.append(lab)
            images = torch.stack(images, dim=0).to(args.device)
            labels = torch.tensor(labels, dtype=torch.long, device="cpu")

            zca_images = zca(images).to("cpu")
            dst_test = TensorDataset(zca_images, labels)

        args.zca_trans = zca


    testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=0)


    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader, loader_train_dict, class_map, class_map_inv



class TensorDataset(Dataset):
    def __init__(self, images, labels): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.images.shape[0]


def test_client(ipc, num_clients, net, client_path, args):
    wandb.init(sync_tensorboard=False,
                project="MTT-TestClients",
                entity="tongchen",
                name='epoch_eval_train={}'.format(args.epoch_eval_train),
                    # name='test',
                config=args,
                )
    
    channel = 3
    im_size = (224, 224)
    num_classes = 2
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if args.zca:
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    task = int(args.dataset[3])-1
    class_names = ['MSI', 'MSS']
    CRC_test = '/shared/dqwang/datasets/CRC/CRC_DX_test'
    test_ann_path = '/shared/dqwang/datasets/CRC/annotation/test_ann.txt'
    dst_test = CRCK(CRC_test, test_ann_path, cls_ind=task, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=0)

    images_train = []
    labels_train = []
    lrs = []

    for i in range(num_clients):
        img_tensor = torch.load(client_path+'/images/client'+str(i)+'.pt')
        images_train.append(img_tensor)
        labels = []
        for j in range(2*ipc):
            if j < ipc :
                labels.append(0)
            else:
                labels.append(1)
        labels = torch.tensor(labels)
        labels_train.append(labels)
        lr = torch.load(client_path+'/lrs/client'+str(i)+'.pt')
        lrs.append(lr)

    # pdb.set_trace()
    images_train = torch.stack(images_train)
    labels_train = torch.stack(labels_train)

    it = 0
    it_eval = 0
    
    net = net.to(args.device)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch//2+1]
    criterion = nn.CrossEntropyLoss().to(args.device)
    loss_avg, acc_avg, num_exp = 0, 0, 0
    
    net.eval()
    optimizers = []
    for i in range(0, num_clients):
        optimizer = torch.optim.SGD(net.parameters(), lr=lrs[i], momentum=0.9, weight_decay=0.0005)
        optimizers.append(optimizer)

    for ep in tqdm(range(Epoch+1)):
        # train the network using distilled dataset
        for i in range(0, num_clients):
            img = images_train[i].to(args.device)
            lab = labels_train[i].to(args.device)
            optimizer = optimizers[i]

            n_b = lab.shape[0]

            output = net(img)
            loss = criterion(output, lab)
            acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

            loss_avg += loss.item()*n_b
            acc_avg += acc
            num_exp += n_b
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        loss_avg /= num_exp
        acc_avg /= num_exp
        
        wandb.log({"loss": loss_avg}, step=ep)
        wandb.log({"acc": acc_avg}, step=ep)
        
        if ep == Epoch:
            # this optimizer doesn't matter, won't be used in epoch
            with torch.no_grad():
                loss_test, acc_test, auc_test = epoch('test', testloader, net, optimizers[0], criterion, args, aug=False)
                wandb.log({"Test Loss": loss_test})
                wandb.log({"Test ACC": acc_test})
                wandb.log({"Test AUC": auc_test})

        if ep in lr_schedule:
            optimizer = torch.optim.SGD(net.parameters(), lr=lrs[i]*0.1, momentum=0.9, weight_decay=0.0005)
            optimizers.append(optimizer)
            
    wandb.finish()
    
def test_client_weighted(ipc, num_clients, net, client_path, args):
    cnt = [14646, 13004, 17960, 23209, 24589]
    weight = cnt / np.sum(cnt)
    wandb.init(sync_tensorboard=False,
                project="MTT-TestClients",
                entity="tongchen",
                name='weighted-epoch_eval_train={}'.format(args.epoch_eval_train),
                    # name='test',
                config=args,
                )
    
    channel = 3
    im_size = (224, 224)
    num_classes = 2
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if args.zca:
        transform = transforms.Compose([transforms.ToTensor()])
    else:
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    task = int(args.dataset[3])-1
    class_names = ['MSI', 'MSS']
    CRC_test = '/shared/dqwang/datasets/CRC/CRC_DX_test'
    test_ann_path = '/shared/dqwang/datasets/CRC/annotation/test_ann.txt'
    dst_test = CRCK(CRC_test, test_ann_path, cls_ind=task, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(dst_test, batch_size=256, shuffle=False, num_workers=0)

    images_train = []
    labels_train = []
    lrs = []

    for i in range(num_clients):
        img_tensor = torch.load(client_path+'/images/client'+str(i)+'.pt')
        images_train.append(img_tensor)
        labels = []
        for j in range(2*ipc):
            if j < ipc :
                labels.append(0)
            else:
                labels.append(1)
        labels = torch.tensor(labels)
        labels_train.append(labels)
        lr = torch.load(client_path+'/lrs/client'+str(i)+'.pt')
        lrs.append(lr)

    # pdb.set_trace()
    images_train = torch.stack(images_train)
    labels_train = torch.stack(labels_train)

    it = 0
    it_eval = 0
    
    net = net.to(args.device)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch//2+1]
    criterion = nn.CrossEntropyLoss().to(args.device)
    loss_avg, acc_avg, num_exp = 0, 0, 0
    
    net.eval()
    optimizers = []
    for i in range(0, num_clients):
        optimizer = torch.optim.SGD(net.parameters(), lr=lrs[i]*weight[i], momentum=0.9, weight_decay=0.0005)
        optimizers.append(optimizer)

    for ep in tqdm(range(Epoch+1)):
        # train the network using distilled dataset
        for i in range(0, num_clients):
            img = images_train[i].to(args.device)
            lab = labels_train[i].to(args.device)
            optimizer = optimizers[i]

            n_b = lab.shape[0]

            output = net(img)
            loss = criterion(output, lab)
            acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

            loss_avg += loss.item()*n_b
            acc_avg += acc
            num_exp += n_b
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        loss_avg /= num_exp
        acc_avg /= num_exp
        
        wandb.log({"loss": loss_avg}, step=ep)
        wandb.log({"acc": acc_avg}, step=ep)
        
        if ep == Epoch:
            # this optimizer doesn't matter, won't be used in epoch
            with torch.no_grad():
                loss_test, acc_test, auc_test = epoch('test', testloader, net, optimizers[0], criterion, args, aug=False)
                wandb.log({"Test Loss": loss_test})
                wandb.log({"Test ACC": acc_test})
                wandb.log({"Test AUC": auc_test})

        if ep in lr_schedule:
            optimizer = torch.optim.SGD(net.parameters(), lr=lrs[i]*weight[i]*0.1, momentum=0.9, weight_decay=0.0005)
            optimizers.append(optimizer)
            
    wandb.finish()

def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling



def get_network(model, channel, num_classes, im_size=(32, 32), dist=True, no_frz=False):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    if model == 'MLP':
        net = MLP(channel=channel, num_classes=num_classes)
    elif model == 'ConvNet':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'LeNet':
        net = LeNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNet':
        net = AlexNet(channel=channel, num_classes=num_classes)
    elif model == 'VGG11':
        net = VGG11( channel=channel, num_classes=num_classes)
    elif model == 'VGG11BN':
        net = VGG11BN(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18':
        net = MyResNet18(ck_path='/shared/dqwang/scratch/lfzhou/r18_imgpre.pth', num_classes=num_classes, no_frz=no_frz)
        # net = ResNet34(channel=channel, num_classes=num_classes)
    elif model == 'ResNet34':
        net = ResNet34(channel=channel, num_classes=num_classes)
    elif model == 'ResNet50':
        # net = ResNet50(channel=channel, num_classes=num_classes)
        net = MyResNet50(ck_path='/shared/dqwang/scratch/tongchen/r50_imgpre.pth', num_classes=num_classes, no_frz=no_frz)
    elif model == 'ResNet18BN_AP':
        net = ResNet18BN_AP(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18_AP':
        net = ResNet18_AP(channel=channel, num_classes=num_classes)

    elif model == 'ConvNetD1':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=1, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD2':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=2, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD3':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=3, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD4':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=4, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD5':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=5, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD6':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=6, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD7':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=7, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD8':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=8, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)


    elif model == 'ConvNetW32':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=32, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW64':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=64, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW128':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=128, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW256':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=256, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW512':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=512, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetW1024':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=1024, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling)

    elif model == "ConvNetKIP":
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=1024, net_depth=net_depth, net_act=net_act,
                      net_norm="none", net_pooling=net_pooling)

    elif model == 'ConvNetAS':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='sigmoid', net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetAR':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='relu', net_norm=net_norm, net_pooling=net_pooling)
    elif model == 'ConvNetAL':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='leakyrelu', net_norm=net_norm, net_pooling=net_pooling)

    elif model == 'ConvNetNN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='none', net_pooling=net_pooling)
    elif model == 'ConvNetBN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='batchnorm', net_pooling=net_pooling)
    elif model == 'ConvNetLN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='layernorm', net_pooling=net_pooling)
    elif model == 'ConvNetIN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='instancenorm', net_pooling=net_pooling)
    elif model == 'ConvNetGN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='groupnorm', net_pooling=net_pooling)

    elif model == 'ConvNetNP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='none')
    elif model == 'ConvNetMP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='maxpooling')
    elif model == 'ConvNetAP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='avgpooling')


    else:
        net = None
        exit('DC error: unknown model')

    if dist:
        gpu_num = torch.cuda.device_count()
        if gpu_num>0:
            device = 'cuda'
            if gpu_num>1:
                net = nn.DataParallel(net)
        else:
            device = 'cpu'
        net = net.to(device)

    return net



def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))

def epoch_mimic(mode, dataset, dataloader, net, optimizer, criterion, scheduler, args, aug, texture=False):
    """ epoch() for MIMIC dataset """
    loss_avg, num_exp = 0, 0
    net = net.to(args.device)
    
    # I added dropout for MyResNet
    if mode == 'train':
        net.train()
    else:
        net.eval()

    if mode == 'test' or mode == 'train':
        data_infos = dataset.data_infos
        subject_infos = dataset.subject_infos

        gt_labels = np.array([data['gt_label'] for data in data_infos])

    results = {}

    # if mode=='eval_train':
    #     pdb.set_trace()
    
    start = time.time()
    for i_batch, datum in tqdm(enumerate(dataloader), total=len(dataloader), desc="Loading data", position=0):
        # pdb.set_trace()
        img = datum[0].float().to(args.device)
        if mode == 'test' or mode == 'train':
            lab = torch.stack(datum[1]).transpose(0,1).float().to(args.device)
            bagnames = datum[2]
        else:
            lab = datum[1].float().to(args.device)

        # pdb.set_trace()

        if aug:
            if args.dsa:
                img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
            else:
                img = augment(img, args.dc_aug_param, device=args.device)

        n_b = lab.shape[0]

        if mode =='test':
            with torch.no_grad():
                output = net(img)
        else:
            output = net(img)
        
        # pdb.set_trace()
        if mode == 'test' or mode == 'train':
            for i in range(len(bagnames)):
                results[bagnames[i]] = results.get(bagnames[i], []) + [output[i].tolist()]

        loss = criterion(output, lab)
        loss_avg += loss.item()*n_b
        num_exp += n_b

        if mode != 'test':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if mode == 'train':
                scheduler.step()
    
    dataset_time = time.time() - start
    print("Time for enumerating the whole dataset: {}".format(dataset_time))
    
    loss_avg /= num_exp

    # pdb.set_trace()
    if mode == 'test' or mode == 'train':
        for bag in list(results.keys()):
            results[bag] = np.array(results[bag])
        
        # pdb.set_trace()
        start = time.time()
        # num_imgs = len(results)
        threshold = 0.5
        bags = list(subject_infos.keys())
        bag_gt_labels = np.array([gt_labels[subject_infos[b][0]] for b in bags])    # bag_gt_labels.shape = (999, 14)
        bag_results = np.array([np.mean(results[b], axis=0) for b in bags])
        bag_class_auc = []
        for i in range(len(dataset.CLASSES)):
            auc = roc_auc_score(bag_gt_labels[:, i], bag_results[:, i])
            bag_class_auc.append(auc)
        mean_bag_class_auc = roc_auc_score(bag_gt_labels, bag_results, average='micro')
        time_cal = time.time() - start
        print("Caculate bag_class_auc time: {}".format(time_cal))
        return loss_avg, bag_class_auc, mean_bag_class_auc
    else:
        return loss_avg

def epoch(mode, dataloader, net, optimizer, criterion, args, aug, texture=False):
        
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(args.device)

    if args.dataset == "ImageNet":
        class_map = {x: i for i, x in enumerate(config.img_net_classes)}

    # I added dropout for MyResNet
    if mode == 'train':
        net.train()
    else:
        net.eval()

    if mode == 'test' or mode == 'train':
        bag={}
        bag_label={}
        bag_score={}
        bag_pred_dict = defaultdict(list)
        bag_results=[]

    for i_batch, datum in tqdm(enumerate(dataloader), total=len(dataloader), desc="Loading data", position=0):
        img = datum[0].float().to(args.device)
        lab = datum[1].long().to(args.device)
        if mode == 'test' or mode == 'train':
            bagname = datum[2]

        if mode == "train" and texture:
            img = torch.cat([torch.stack([torch.roll(im, (torch.randint(args.im_size[0]*args.canvas_size, (1,)), torch.randint(args.im_size[0]*args.canvas_size, (1,))), (1,2))[:,:args.im_size[0],:args.im_size[1]] for im in img]) for _ in range(args.canvas_samples)])
            lab = torch.cat([lab for _ in range(args.canvas_samples)])

        if aug:
            if args.dsa:
                img = DiffAugment(img, args.dsa_strategy, param=args.dsa_param)
            else:
                img = augment(img, args.dc_aug_param, device=args.device)

        if args.dataset == "ImageNet" and mode != "train":
            lab = torch.tensor([class_map[x.item()] for x in lab]).to(args.device)

        n_b = lab.shape[0]

        if mode =='test':
            with torch.no_grad():
                output = net(img)
        else:
            output = net(img)
        loss = criterion(output, lab)
        
        # img = img.cpu()

        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

        if mode != 'test':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if mode == 'test' or mode == 'train':
            for j in range(len(bagname)):
                if bagname[j] not in bag:
                    bag[bagname[j]] = 0
                    bag_label[bagname[j]] = int(lab[j])
                    bag_score[bagname[j]] = 0
                bag_score[bagname[j]] += output[j][0]
                bag[bagname[j]] += 1

    loss_avg /= num_exp
    acc_avg /= num_exp

    if mode == 'test' or mode == 'train':
        for b in bag_score.keys():
            mean_0 = bag_score[b] / bag[b]
            bag_pred_dict[b]=[mean_0, 1 - mean_0]

        bag_results = [kv[1][1].cpu().detach().numpy() for kv in sorted(bag_pred_dict.items(), key=lambda x: x[0])]
        # print(bag_results)
        # bag_results = np.vstack(bag_results)
        bag_labels = [kv[1] for kv in sorted(bag_label.items(), key=lambda x: x[0])]
        # bag_labels = np.array(bag_labels)
        # print(bag_labels)
        aucs = roc_auc_score(bag_labels,bag_results)
        # print('test auc: {}'.format(aucs))

    if mode == 'test' or mode == 'train':
        return loss_avg, acc_avg, aucs
    return loss_avg, acc_avg



def evaluate_synset(it, it_eval, net, images_train, labels_train, test_dataset, testloader, args, return_loss=False, texture=False):
    net = net.to(args.device)
    images_train = images_train.to(args.device)
    labels_train = labels_train.to(args.device)
    lr = float(args.lr_net)
    Epoch = int(args.epoch_eval_train)
    lr_schedule = [Epoch//2+1]
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    criterion = nn.CrossEntropyLoss().to(args.device)

    dst_train = TensorDataset(images_train, labels_train)
    trainloader = torch.utils.data.DataLoader(dst_train, batch_size=args.batch_train, shuffle=True, num_workers=0)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(dst_train), eta_min=0)

    start = time.time()
    acc_train_list = []
    loss_train_list = []

    if args.dataset.startswith('MIMIC'):
        for ep in tqdm(range(Epoch+1)):
            loss_train = epoch_mimic('eval_train', dst_train, trainloader, net, optimizer, criterion, scheduler, args, aug=False, texture=texture)
            loss_train_list.append(loss_train)

            if ep == Epoch:
                with torch.no_grad():
                    loss_test, _, auc_test = epoch_mimic('test', test_dataset, testloader, net, optimizer, criterion, scheduler, args, aug=False)

            if ep in lr_schedule:
                lr *= 0.1
                optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)

    else:
        for ep in tqdm(range(Epoch+1)):
            loss_train, acc_train = epoch('eval_train', trainloader, net, optimizer, criterion, args, aug=False, texture=texture)
            acc_train_list.append(acc_train)
            loss_train_list.append(loss_train)

            # if ep % (Epoch//10) == 0:
            #     with torch.no_grad():
            #         loss_test, acc_test, auc_test = epoch('test', testloader, net, optimizer, criterion, args, aug=False)
            #     wandb.log({'Acc/Eval_{}_{}'.format(it, it_eval): acc_test}, step=ep)
            #     wandb.log({'Auc/Eval_{}_{}'.format(it, it_eval): auc_test}, step=ep)
            #     wandb.log({'Lr/Eval_{}_{}'.format(it, it_eval): lr}, step=ep)
            if ep == Epoch:
                with torch.no_grad():
                    loss_test, acc_test, auc_test = epoch('test', testloader, net, optimizer, criterion, args, aug=False)

            if ep in lr_schedule:
                lr *= 0.1
                optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=0.0005)


    time_train = time.time() - start
    if args.dataset.startswith('MIMIC'):
        print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f test auc = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, auc_test))
        if return_loss:
            return net, auc_test, loss_train_list, loss_test
        else:
            return net, auc_test
    else:
        print('%s Evaluate_%02d: epoch = %04d train time = %d s train loss = %.6f train acc = %.4f, test acc = %.4f, test auc = %.4f' % (get_time(), it_eval, Epoch, int(time_train), loss_train, acc_train, acc_test, auc_test))
        if return_loss:
            return net, acc_train_list, acc_test, loss_train_list, loss_test, auc_test
        else:
            return net, acc_train_list, acc_test, auc_test


def augment(images, dc_aug_param, device):
    # This can be sped up in the future.

    if dc_aug_param != None and dc_aug_param['strategy'] != 'none':
        scale = dc_aug_param['scale']
        crop = dc_aug_param['crop']
        rotate = dc_aug_param['rotate']
        noise = dc_aug_param['noise']
        strategy = dc_aug_param['strategy']

        shape = images.shape
        mean = []
        for c in range(shape[1]):
            mean.append(float(torch.mean(images[:,c])))

        def cropfun(i):
            im_ = torch.zeros(shape[1],shape[2]+crop*2,shape[3]+crop*2, dtype=torch.float, device=device)
            for c in range(shape[1]):
                im_[c] = mean[c]
            im_[:, crop:crop+shape[2], crop:crop+shape[3]] = images[i]
            r, c = np.random.permutation(crop*2)[0], np.random.permutation(crop*2)[0]
            images[i] = im_[:, r:r+shape[2], c:c+shape[3]]

        def scalefun(i):
            h = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            w = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            tmp = F.interpolate(images[i:i + 1], [h, w], )[0]
            mhw = max(h, w, shape[2], shape[3])
            im_ = torch.zeros(shape[1], mhw, mhw, dtype=torch.float, device=device)
            r = int((mhw - h) / 2)
            c = int((mhw - w) / 2)
            im_[:, r:r + h, c:c + w] = tmp
            r = int((mhw - shape[2]) / 2)
            c = int((mhw - shape[3]) / 2)
            images[i] = im_[:, r:r + shape[2], c:c + shape[3]]

        def rotatefun(i):
            im_ = scipyrotate(images[i].cpu().data.numpy(), angle=np.random.randint(-rotate, rotate), axes=(-2, -1), cval=np.mean(mean))
            r = int((im_.shape[-2] - shape[-2]) / 2)
            c = int((im_.shape[-1] - shape[-1]) / 2)
            images[i] = torch.tensor(im_[:, r:r + shape[-2], c:c + shape[-1]], dtype=torch.float, device=device)

        def noisefun(i):
            images[i] = images[i] + noise * torch.randn(shape[1:], dtype=torch.float, device=device)


        augs = strategy.split('_')

        for i in range(shape[0]):
            choice = np.random.permutation(augs)[0] # randomly implement one augmentation
            if choice == 'crop':
                cropfun(i)
            elif choice == 'scale':
                scalefun(i)
            elif choice == 'rotate':
                rotatefun(i)
            elif choice == 'noise':
                noisefun(i)

    return images



def get_daparam(dataset, model, model_eval, ipc):
    # We find that augmentation doesn't always benefit the performance.
    # So we do augmentation for some of the settings.

    dc_aug_param = dict()
    dc_aug_param['crop'] = 4
    dc_aug_param['scale'] = 0.2
    dc_aug_param['rotate'] = 45
    dc_aug_param['noise'] = 0.001
    dc_aug_param['strategy'] = 'none'

    if dataset == 'MNIST':
        dc_aug_param['strategy'] = 'crop_scale_rotate'

    if model_eval in ['ConvNetBN']:  # Data augmentation makes model training with Batch Norm layer easier.
        dc_aug_param['strategy'] = 'crop_noise'

    return dc_aug_param


def get_eval_pool(eval_mode, model, model_eval):
    if eval_mode == 'M': # multiple architectures
        # model_eval_pool = ['MLP', 'ConvNet', 'AlexNet', 'VGG11', 'ResNet18', 'LeNet']
        model_eval_pool = ['ConvNet', 'AlexNet', 'VGG11', 'ResNet18_AP', 'ResNet18']
        # model_eval_pool = ['MLP', 'ConvNet', 'AlexNet', 'VGG11', 'ResNet18']
    elif eval_mode == 'W': # ablation study on network width
        model_eval_pool = ['ConvNetW32', 'ConvNetW64', 'ConvNetW128', 'ConvNetW256']
    elif eval_mode == 'D': # ablation study on network depth
        model_eval_pool = ['ConvNetD1', 'ConvNetD2', 'ConvNetD3', 'ConvNetD4']
    elif eval_mode == 'A': # ablation study on network activation function
        model_eval_pool = ['ConvNetAS', 'ConvNetAR', 'ConvNetAL']
    elif eval_mode == 'P': # ablation study on network pooling layer
        model_eval_pool = ['ConvNetNP', 'ConvNetMP', 'ConvNetAP']
    elif eval_mode == 'N': # ablation study on network normalization layer
        model_eval_pool = ['ConvNetNN', 'ConvNetBN', 'ConvNetLN', 'ConvNetIN', 'ConvNetGN']
    elif eval_mode == 'S': # itself
        model_eval_pool = [model[:model.index('BN')]] if 'BN' in model else [model]
    elif eval_mode == 'C':
        model_eval_pool = [model, 'ConvNet']
    else:
        model_eval_pool = [model_eval]
    return model_eval_pool


class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S' #'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5 # the size would be 0.5x0.5
        self.ratio_noise = 0.05
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5


def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1


def DiffAugment(x, strategy='', seed = -1, param = None):
    if seed == -1:
        param.batchmode = False
    else:
        param.batchmode = True

    param.latestseed = seed

    if strategy == 'None' or strategy == 'none':
        return x

    if strategy:
        if param.aug_mode == 'M': # original
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == 'S':
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
        else:
            exit('Error ZH: unknown augmentation mode.')
        x = x.contiguous()
    return x


# We implement the following differentiable augmentation strategies based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
def rand_scale(x, param):
    # x>1, max scale
    # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    theta = [[[sx[i], 0,  0],
            [0,  sy[i], 0],] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.batchmode: # batch-wise:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    x = F.grid_sample(x, grid, align_corners=True)
    return x


def rand_rotate(x, param): # [-180, 180], 90: anticlockwise 90 degree
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
        [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.batchmode: # batch-wise:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape, align_corners=True).to(x.device)
    x = F.grid_sample(x, grid, align_corners=True)
    return x


def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.batchmode: # batch-wise:
        randf[:] = randf[0]
    return torch.where(randf < prob, x.flip(3), x)


def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        randb[:] = randb[0]
    x = x + (randb - 0.5)*ratio
    return x


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        rands[:] = rands[0]
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.batchmode:  # batch-wise:
        randc[:] = randc[0]
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    # The image is padded on its surrounding and then cropped.
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    if param.batchmode:  # batch-wise:
        translation_x[:] = translation_x[0]
        translation_y[:] = translation_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.batchmode:  # batch-wise:
        offset_x[:] = offset_x[0]
        offset_y[:] = offset_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}
