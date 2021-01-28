import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu_id', type=int, default=1, help='cuda:No')

    # dataset related
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='mnist, fmnist, cifar10')
    parser.add_argument('--nn_name', type=str, default='resnet18', help='mnist, fmnist, simplecifar, resnet18')
    parser.add_argument('--dataset_dist', type=str, default='iid', help='distribution of dataset; iid or non_iid')
    # Federated params
    parser.add_argument('--mode', type=str, default='hybrid', help='hybrid or top-K')
    parser.add_argument('--gamma', type=float, default=0.9, help='freq value')
    parser.add_argument('--warmUp', type=int, default=20, help='No sparse for that much rounds')
    parser.add_argument('--num_client', type=int, default=10, help='number of clients')
    parser.add_argument('--comm_round', type=int, default=1200, help='number of epochs')
    parser.add_argument('--LocalIter', type=int, default=16, help='Local iterations')
    parser.add_argument('--bs', type=int, default=64, help='batchsize')
    parser.add_argument('--lr', type=float, default=0.1, help='learning_rate')
    parser.add_argument('--nesterov', type=bool, default=False, help='enable nesterov momentum')
    parser.add_argument('--SGDmomentum', type=float, default=0, help='momentum')
    parser.add_argument('--biasFair', type=bool, default=True, help='enable bias Fairness for sparsing ')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay Value')
    parser.add_argument('--sparsity', type=int, default=128, help='/ sparsing ')
    parser.add_argument('--C_rand', type=int, default=256, help='/ random sends ')
    parser.add_argument('--freq_K', type=int, default=128, help=' frequency vector top-k ')
    args = parser.parse_args()
    return args
