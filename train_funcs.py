import torch
from torch.utils.data import DataLoader
# custom modules
import data_loader as dl
from nn_classes import *
import server_functions as sf
import math
from parameters import *
import time
import numpy as np
from tqdm import tqdm


def evaluate_accuracy(model, testloader, device):
    """Calculates the accuracy of the model"""
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def evaluate_training_loss(model, trainloader,device):
    criterion = nn.CrossEntropyLoss()
    count = 0
    loss = 0
    with torch.no_grad():
        for data in trainloader:
            count+=1
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss += criterion(outputs, labels).item()
    return loss/count


def train(args, device):

    num_client = args.num_client
    trainset, testset = dl.get_dataset(args)
    sample_inds = dl.get_indices(trainset, args)
    # PS model
    net_ps = get_net(args).to(device)
    net_users = [get_net(args).to(device) for u in range(num_client)]
    optimizers = [torch.optim.SGD(net_users[cl].parameters(), lr=args.lr, weight_decay=1e-4,momentum=args.SGDmomentum) for cl in
                  range(num_client)]
    criterions = [nn.CrossEntropyLoss() for u in range(num_client)]
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=2)

    # synch all clients models models with PS
    [sf.pull_model(net_users[cl], net_ps) for cl in range(num_client)]

    net_sizes, net_nelements = sf.get_model_sizes(net_ps)
    ind_pairs = sf.get_indices(net_sizes, net_nelements)
    N_s = (50000 if args.dataset_name == 'cifar10' else 60000)
    accuracys = []
    acc = evaluate_accuracy(net_ps, testloader, device)
    accuracys.append(acc * 100)
    model_size = sf.count_parameters(net_ps)
    assert N_s/num_client > args.LocalIter * args.bs
    localIterCap = args.LocalIter
    bias_mask = sf.get_BN_mask(net_ps,device) if args.biasFair else torch.zeros(model_size).to(device)
    total_bias = torch.sum(bias_mask).item()
    freq_vec = torch.zeros(model_size,device=device)
    k = int(math.ceil(model_size / args.sparsity) - total_bias)
    C = int(math.ceil(model_size / args.C_rand) - total_bias)
    freq_k = math.ceil(model_size /args.freq_K)
    errors = [torch.zeros(model_size,device=device) for cl in range(num_client)]
    randMask = [None for h in range(args.LocalIter-1)]
    m_freq_inds = None

    print('bias percent',total_bias / model_size * 100, total_bias)
    for round in tqdm(range(args.comm_round)):
        avg_res = torch.zeros(model_size,device=device)
        ps_flat = sf.get_model_flattened(net_ps, device)
        for cl in range(num_client):
            localIter = 0
            prev_dif = torch.zeros(model_size,device=device)

            trainloader = DataLoader(dl.DatasetSplit(trainset, sample_inds[cl]), batch_size=args.bs,
                                     shuffle=True)
            worker_0 = sf.get_model_flattened(net_users[cl], device)
            for data in trainloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizers[cl].zero_grad()
                predicts = net_users[cl](inputs)
                loss = criterions[cl](predicts, labels)
                loss.backward()
                optimizers[cl].step()
                localIter +=1
                if round >args.warmUp:## after WarmUp
                    woker_flat = sf.get_model_flattened(net_users[cl], device)
                    dif = woker_flat.sub(worker_0, alpha=1).mul(1/localIter)
                    res_mask = torch.zeros(model_size,device=device)
                    res_ = dif.sub(prev_dif, alpha=1).add(errors[cl]) ## get Difference values then add errors
                    if localIter < localIterCap: ## tau < T
                        res_mask[torch.masked_select(m_freq_inds, randMask[localIter-1])] = 1 ## make mask from random inds
                    else:
                        freq_inds = torch.topk(res_.mul(1-bias_mask).abs(), k=k, dim=0)[1]
                        res_mask[freq_inds] = 1 ## make the mask from all inds
                    res_mask.add_(bias_mask,alpha=1)
                    prev_dif = dif
                    errors[cl] = res_.mul(1 - res_mask) ### update errors
                    avg_res.add_(res_.mul(res_mask), alpha=1) ## mask residual values and send to PS
                if num_client>1 and localIter == localIterCap:
                    break
        freq_vec.mul_(args.gamma)
        if round < args.warmUp: ## warmUp
            avg_dif = torch.zeros(model_size,device=device)
            for user in net_users:
                flat_model= sf.get_model_flattened(user,device)
                avg_dif.add_(flat_model.sub(ps_flat,alpha=1),alpha=1/num_client)
            sf.make_model_unflattened(net_ps, ps_flat.add(avg_dif, alpha=1), net_sizes, ind_pairs)
            freq_vec[torch.topk(avg_dif.abs(), k=k, dim=0)[1]] = +1
        else:
            avg_res.mul_(localIterCap / num_client)
            sf.make_model_unflattened(net_ps, ps_flat.add(avg_res, alpha=1), net_sizes, ind_pairs)
            freq_vec[torch.topk(avg_res.abs(),k=k,dim=0)[1]] = +1 ##update freq Vec
            for cl in range(localIterCap-1): ### make random boolean masks for next iter
                randMask[cl] = torch.randperm(k, device=device) < C
                '''
                generates randomly distributed values from 0 to k,
                pick the inds where values are less then C,
                Every worker uses same mask for the same local iteration
                Every local iteration has different random mask. 
                '''
            m_freq_inds = torch.topk(freq_vec.mul(1-bias_mask),k=k,dim=0)[1] #### get most frequent inds


        [sf.pull_model(user,net_ps) for user in net_users]
        acc = evaluate_accuracy(net_ps, testloader, device)
        accuracys.append(acc * 100)
        print('accuracy:{}'.format(acc*100))
    return accuracys

def train_topk(args, device):

    num_client = args.num_client
    trainset, testset = dl.get_dataset(args)
    sample_inds = dl.get_indices(trainset, args)
    # PS model
    net_ps = get_net(args).to(device)
    # prev_models = [get_net(args).to(device) for u in range(num_client)]
    # [sf.initialize_zero(prev_models[u]) for u in range(num_client)]



    net_users = [get_net(args).to(device) for u in range(num_client)]
    optimizers = [torch.optim.SGD(net_users[cl].parameters(), lr=args.lr, weight_decay=1e-4,momentum=args.SGDmomentum) for cl in
                  range(num_client)]
    criterions = [nn.CrossEntropyLoss() for u in range(num_client)]
    testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=2)

    # synch all clients models models with PS
    [sf.pull_model(net_users[cl], net_ps) for cl in range(num_client)]

    net_sizes, net_nelements = sf.get_model_sizes(net_ps)
    ind_pairs = sf.get_indices(net_sizes, net_nelements)
    N_s = (50000 if args.dataset_name == 'cifar10' else 60000)
    accuracys = []
    acc = evaluate_accuracy(net_ps, testloader, device)
    accuracys.append(acc * 100)
    model_size = sf.count_parameters(net_ps)
    assert N_s/num_client > args.LocalIter * args.bs
    localIterCap = args.LocalIter
    bias_mask = sf.get_BN_mask(net_ps, device) if args.biasFair else torch.zeros(model_size).to(device)
    total_bias = torch.sum(bias_mask).item()
    print('bias percent',total_bias / model_size * 100, total_bias)
    k = math.ceil(model_size / args.sparsity)
    errors = [torch.zeros(model_size,device=device) for cl in range(num_client)]
    for round in tqdm(range(args.comm_round)):
        ps_flat = sf.get_model_flattened(net_ps, device)
        for cl in range(num_client):
            localIter = 0
            trainloader = DataLoader(dl.DatasetSplit(trainset, sample_inds[cl]), batch_size=args.bs,
                                     shuffle=True)
            for data in trainloader:
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                optimizers[cl].zero_grad()
                predicts = net_users[cl](inputs)
                loss = criterions[cl](predicts, labels)
                loss.backward()
                optimizers[cl].step()
                localIter +=1
                if num_client>1 and localIter == localIterCap:
                    break
        avg_dif = torch.zeros(model_size, device=device)
        if round < args.warmUp:
            for user in net_users:
                flat_model= sf.get_model_flattened(user,device)
                avg_dif.add_(flat_model.sub(ps_flat,alpha=1),alpha=1/num_client)
        else:
            c = 0
            for user in net_users:
                topped_m = torch.zeros(model_size,device=device)
                flat_model= sf.get_model_flattened(user,device)
                dif = flat_model.sub(ps_flat,alpha=1).add(errors[c])
                vals, inds = torch.topk(dif.mul(1-bias_mask).abs(), k=k,dim=0)
                topped_m[inds] = 1
                topped_m.add_(bias_mask,alpha=1)
                avg_dif.add_(dif.mul(topped_m),alpha=1/num_client)
                errors[c] = dif.mul(1-topped_m)
                c+=1
        sf.make_model_unflattened(net_ps, ps_flat.add(avg_dif, alpha=1), net_sizes, ind_pairs)



        [sf.pull_model(user,net_ps) for user in net_users]
        acc = evaluate_accuracy(net_ps, testloader, device)
        accuracys.append(acc * 100)
        print('accuracy:{}'.format(acc*100))
    return accuracys



