""""""
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np


def get_mnist_dataset():
    """returns trainset and testsets for MNIST dataset"""

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform_test)

    return trainset, testset


def get_fmnist_dataset():
    """returns trainset and testsets for Fashion MNIST dataset"""

    transform_train = transforms.Compose([
        transforms.ToTensor()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])

    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform_test)

    return trainset, testset


def get_cifar10_dataset():
    """returns trainset and testsets for Fashion CIFAR10 dataset"""

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    return trainset, testset


def get_dataset(args):

    dataset_name = args.dataset_name

    if dataset_name == 'mnist':
        trainset, testset = get_mnist_dataset()
    elif dataset_name == 'fmnist':
        trainset, testset = get_fmnist_dataset()
    elif dataset_name == 'cifar10':
        trainset, testset = get_cifar10_dataset()
    else:
        raise ValueError('dataset name can only be mnist, fmnist or cifar10')

    return trainset, testset

def get_indices(trainset, args):
    """returns the indices of sample for each worker in either iid, or non_iid manner provided in args"""

    if args.dataset_dist == 'iid':
        inds = get_iid_index(trainset, args)
    elif args.dataset_dist == 'non_iid':
        inds = get_non_iid_index(trainset, args)
    elif args.dataset_dist == 'non_iid2':
        inds = get_non_iid_index_2(trainset,args)
    else:
        raise ValueError('Dataset distribution can only be iid or non_iid')
    return inds


def get_non_iid_index(trainset, args):
    """Returns the indexes of samples for each user such that the distributions of data for each user
    have a non_iid distribution. Sorts the indexs that have a lablel 0 to label 10. Then equally splits
     the indexes for each user"""
    if args.dataset_name == 'mnist':
        num_samples = trainset.train_labels.shape[0]
        labels = trainset.train_labels.numpy()
    elif args.dataset_name == 'cifar10':
        labels = trainset.targets
        num_samples = len(labels)

    inds_sorted = np.argsort(labels) # sort indicies based on labels
    num_sample_perworker = int(num_samples / args.num_client)

    indx_sample = {n: [] for n in range(args.num_client)}

    for user in range(args.num_client):
        indx_sample[user] = inds_sorted[user * num_sample_perworker: (user + 1) * num_sample_perworker] # assign indices to each user
    print(np.shape(indx_sample[10]))
    return indx_sample

def get_non_iid_index_2(trainset, args):

    labels = np.asarray(trainset.targets)
    num_samples = len(labels)
    userVec = np.zeros((args.num_client))
    num_class = 10 ##cifar-10
    inds_sorted = np.argsort(labels)
    class_size = int(num_samples/ 10)
    blocks_per_class = int(args.num_client * args.numb_cls_usr / num_class)
    data = []
    img_per_block = int(class_size/blocks_per_class)
    for i in range(num_class):
        startp = i*class_size
        class_datas = []
        excess_data = class_size % blocks_per_class
        for y in range(blocks_per_class):
            endp = 0
            if excess_data !=0:
                endp = startp + img_per_block + 1
                excess_data -=1
            else:
                endp = startp + img_per_block
            class_datas.append(inds_sorted[startp:endp])
            startp = endp
        data.append(class_datas)

    indx_sample = {n: [] for n in range(args.num_client)}
    for i in range(num_class):
        randomly_select_worker = blocks_per_class
        if i >= num_class - args.numb_cls_usr and len(np.where(userVec == i + args.numb_cls_usr - num_class)[0]) > 0 and i != num_class - 1:
            special_workers = np.where(userVec == i + args.numb_cls_usr - num_class)[0]
            userVec[special_workers] += 1
            randomly_select_worker -= len(special_workers)
        eligble_workers = np.random.choice(np.where(userVec < args.numb_cls_usr)[0],randomly_select_worker,replace=False)
        userVec[eligble_workers] +=1
        for y, worker in enumerate(eligble_workers):
            indx_sample[worker] = np.concatenate((indx_sample[worker],data[i][y]),axis=0)
            indx_sample[worker] = indx_sample[worker].astype('int64')
    return indx_sample


def get_iid_index(trainset, args):
    """Returns the indexes of samples for each user such that the distributions of data for each user
    have a iid distribution. Then equally splits
     the indexes for each user"""

    if args.dataset_name == 'mnist':
        num_samples = trainset.train_labels.shape[0]
        labels = trainset.train_labels.numpy()
    elif args.dataset_name == 'cifar10':
        labels = trainset.targets
        num_samples = len(labels)
    num_sample_perworker = int(num_samples / args.num_client)
    inds = [*range(num_samples)]
    inds_split = np.random.choice(inds, [args.num_client, num_sample_perworker], replace=False)
    indx_sample = {n: [] for n in range(args.num_client)}
    for user in range(args.num_client):
        indx_sample[user] = list(inds_split[user])

    return indx_sample


class DatasetSplit(Dataset):
    def __init__(self, dataset, indxs):
        self.dataset = dataset
        self.indxs = indxs

    def __len__(self):
        return len(self.indxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.indxs[item]]
        return image, label

