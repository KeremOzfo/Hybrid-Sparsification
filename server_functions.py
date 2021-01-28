import torch
import math
import time
import numpy as np
import torch.nn as nn


def pull_model(model_user, model_server):
    for param_user, param_server in zip(model_user.parameters(), model_server.parameters()):
        param_user.data = param_server.data[:] + 0
    return None


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def zero_grad_ps(model):
    for param in model.parameters():
        param.grad = torch.zeros_like(param.data)

    return None


def push_grad(model_user, model_server, num_cl):
    for param_user, param_server in zip(model_user.parameters(), model_server.parameters()):
        param_server.grad.data += param_user.grad.data / num_cl
    return None


def push_model(model_user, model_server, num_cl):
    for param_user, param_server in zip(model_user.parameters(), model_server.parameters()):
        param_server.data += param_user.data / num_cl
    return None


def initialize_zero(model):
    for param in model.parameters():
        param.data.mul_(0)
    return None


def update_model(model, prev_model, lr, momentum, weight_decay):
    for param, prevIncrement in zip(model.parameters(), prev_model.parameters()):
        incrementVal = param.grad.data.add(weight_decay, param.data)
        incrementVal.add_(momentum, prevIncrement.data)
        incrementVal.mul_(lr)
        param.data.add_(-1, incrementVal)
        prevIncrement.data = incrementVal
    return None


def get_grad_flattened(model, device):
    grad_flattened = torch.empty(0).to(device)
    for p in model.parameters():
        if p.requires_grad:
            a = p.grad.data.flatten().to(device)
            grad_flattened = torch.cat((grad_flattened, a), 0)
    return grad_flattened


def get_model_flattened(model, device):
    model_flattened = torch.empty(0).to(device)
    for p in model.parameters():
        a = p.data.flatten().to(device)
        model_flattened = torch.cat((model_flattened, a), 0)
    return model_flattened


def get_model_sizes(model):
    # get the size of the layers and number of eleents in each layer.
    # only layers that are trainable
    net_sizes = []
    net_nelements = []
    for p in model.parameters():
        if p.requires_grad:
            net_sizes.append(p.data.size())
            net_nelements.append(p.nelement())
    return net_sizes, net_nelements


def unshuffle(shuffled_vec, seed):
    orj_vec = torch.empty(shuffled_vec.size())
    perm_inds = torch.tensor([i for i in range(shuffled_vec.nelement())])
    perm_inds_shuffled = shuffle_deterministic(perm_inds, seed)
    for i in range(shuffled_vec.nelement()):
        orj_vec[perm_inds_shuffled[i]] = shuffled_vec[i]
    return orj_vec


def shuffle_deterministic(grad_flat, seed):
    # Shuffle the list ls using the seed `seed`
    torch.manual_seed(seed)
    idx = torch.randperm(grad_flat.nelement())
    return grad_flat.view(-1)[idx].view(grad_flat.size())


def get_indices(net_sizes, net_nelements):
    # for reconstructing grad from flattened grad
    ind_pairs = []
    ind_start = 0
    ind_end = 0
    for i in range(len(net_sizes)):

        for j in range(i + 1):
            ind_end += net_nelements[j]
        # print(ind_start, ind_end)
        ind_pairs.append((ind_start, ind_end))
        ind_start = ind_end + 0
        ind_end = 0
    return ind_pairs


def make_grad_unflattened(model, grad_flattened, net_sizes, ind_pairs):
    # unflattens the grad_flattened into the model.grad
    i = 0
    for p in model.parameters():
        if p.requires_grad:
            temp = grad_flattened[ind_pairs[i][0]:ind_pairs[i][1]]
            p.grad.data = temp.reshape(net_sizes[i])
            i += 1
    return None


def make_model_unflattened(model, model_flattened, net_sizes, ind_pairs):
    # unflattens the grad_flattened into the model.grad
    i = 0
    for p in model.parameters():
        temp = model_flattened[ind_pairs[i][0]:ind_pairs[i][1]]
        p.data = temp.reshape(net_sizes[i])
        i += 1
    return None


def make_sparse_grad(grad_flat, sparsity_window, device):
    # sparsify using block model
    num_window = math.ceil(grad_flat.nelement() / sparsity_window)

    for i in range(num_window):
        ind_start = i * sparsity_window
        ind_end = min((i + 1) * sparsity_window, grad_flat.nelement())
        a = grad_flat[ind_start: ind_end]
        ind = torch.topk(a.abs(), k=1, dim=0)[1]  # return index of top not value
        val = a[ind]
        ind_true = ind_start + ind
        grad_flat[ind_start: ind_end] *= torch.zeros(a.nelement()).to(device)
        grad_flat[ind_true] = val

    return None


def adjust_learning_rate(optimizer, epoch, lr_change, lr):
    lr_change = np.asarray(lr_change)
    loc = np.where(lr_change == epoch)[0][0] + 1
    lr *= (0.1 ** loc)
    lr = round(lr, 3)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_LR(optimizer):
    lr = None
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
    return lr


def lr_warm_up(optimizers, num_workers, epoch, start_lr):
    for cl in range(num_workers):
        for param_group in optimizers[cl].param_groups:
            if epoch == 0:
                param_group['lr'] = 0.1
            else:
                lr_change = (start_lr - 0.1) / 4
                param_group['lr'] = (lr_change * epoch) + 0.1

def get_bias_mask(model,device):
    model_flattened = torch.empty(0).to(device)
    for name, p in zip(model.named_parameters(),model.parameters()):
        layer = name[0].split('.')
        a = p.data.flatten().to(device)
        if layer[len(layer)-1] == 'bias':
            temp = torch.ones_like(a).to(device)
            model_flattened = torch.cat((model_flattened, temp), 0)
        else:
            temp = torch.zeros_like(a).to(device)
            model_flattened = torch.cat((model_flattened, temp), 0)
    return model_flattened


def modify_freq_vec(freq_vec, grad, mask,bias_mask,add_percent,args):
    topk = math.ceil(add_percent * (grad.numel() - torch.sum(bias_mask).item()) / 100)
    vals, inds = torch.topk(grad.mul(1-mask).abs(), k=topk, dim=0)
    freq_vec.mul_(args.freq_momentum)
    freq_vec[inds] += 1
    return None

def add_to_mask(freq_vec,mask,bias_mask,add_percent):
    topk = math.ceil(add_percent * (freq_vec.numel() - torch.sum(bias_mask).item()) / 100)
    vals, inds = torch.topk(freq_vec, k=topk, dim=0)
    mask[inds] = 1
    return None

def remove_from_mask(model,mask,bias_mask,drop_val):
    model_size = model.numel()
    zeros = model_size - (torch.nonzero(model.mul(1-bias_mask), as_tuple=False)).numel()
    drop_k = math.ceil(drop_val * (model_size - torch.sum(bias_mask).item()) / 100)
    vals, inds = torch.topk((model.mul(1-bias_mask)).abs(),k=model_size,dim=0)
    inds = torch.flip(inds, dims=[0])
    inds = inds[zeros:zeros+drop_k]
    mask[inds] = 0
    return None


def sparse_special_mask(flat_grad, sparsity_window, layer_spar, ind_pairs, device):
    inds = torch.empty(0).to(device)
    for layer in ind_pairs:
        startPoint = (layer[0])
        endPoint = (layer[1])
        layer_len = endPoint - startPoint
        l_top_k = math.ceil(layer_len / layer_spar)
        l_vals, l_ind = torch.topk((flat_grad[startPoint:endPoint]).abs(), k=l_top_k, dim=0)
        l_ind.add_(startPoint)
        inds = torch.cat((inds.float(), l_ind.float()), 0)
    inds = inds.long()
    clone_grad = torch.clone(flat_grad).to(device)
    clone_grad[inds] = 0
    topk = math.ceil(len(flat_grad) / (sparsity_window)) - inds.numel()
    vals_, inds_ = torch.topk(clone_grad.abs(), k=topk, dim=0)
    inds = torch.cat((inds, inds_), 0)
    clone_grad *= 0
    clone_grad[inds] = 1
    return clone_grad


def groups(grad_flat, group_len, denominator, device):
    sparseCount = torch.sum(grad_flat != 0)
    sparseCount = sparseCount.__int__()
    vals, ind = torch.topk(grad_flat.abs(), k=sparseCount, dim=0)
    group_boundries = torch.zeros(group_len + 1).to(device)
    group_boundries[0] = vals[0].float()
    sign_mask = torch.sign(grad_flat[ind])
    for i in range(1, group_len):
        group_boundries[i] = group_boundries[i - 1] / denominator
    startPoint = 0
    newVals = torch.zeros_like(vals)
    startPointz = []
    for i in range(group_len):
        if vals[startPoint] > group_boundries[i + 1]:
            startPointz.append(startPoint)
            for index, val in enumerate(vals[startPoint:vals.numel()]):
                if val <= group_boundries[i + 1] and group_boundries[i + 1] != 0:
                    newVals[startPoint:startPoint + index] = torch.mean(vals[startPoint:startPoint + index])
                    startPoint += index
                    break
                elif group_boundries[i + 1] == 0:
                    newVals[startPoint:vals.numel()] = torch.mean(vals[startPoint:vals.numel()])
                    break
    newVals *= sign_mask
    grad_flat *= 0
    grad_flat[ind] = newVals

def get_momentum_flattened(opt,device):
    momentum_flattened = torch.empty(0).to(device)
    for groupAvg in (opt.param_groups):  # momentum
        for p_avg in groupAvg['params']:
            param_state_avg = opt.state[p_avg]
            if 'momentum_buffer' not in param_state_avg:
                buf_avg = param_state_avg['momentum_buffer'] = torch.zeros_like(p_avg.data)
            else:
                buf_avg = param_state_avg['momentum_buffer']
            momentum_flattened = torch.cat((momentum_flattened, buf_avg.flatten().to(device)), 0)
    return momentum_flattened

def make_momentum_unflattened(opt, momentum_flattened, net_sizes, ind_pairs):
    import copy
    i = 0
    for groupAvg in (opt.param_groups):  # momentum
        for p_avg in groupAvg['params']:
            temp = momentum_flattened[ind_pairs[i][0]:ind_pairs[i][1]]
            opt.state[p_avg]['momentum_buffer'] = temp.reshape(net_sizes[i])
            i+=1
    return None

def custom_SGD(model,flat_momentum,mask,net_sizes,ind_pairs,lr,device,args):
    flat_model = get_model_flattened(model,device)
    flat_grad = get_grad_flattened(model,device)
    flat_grad = flat_grad.add(flat_model,alpha=args.wd)
    flat_grad.mul_(mask)
    flat_momentum.mul_(args.SGDmomentum).add_(flat_grad, alpha=1)
    if args.nesterov:
        flat_grad = flat_grad.add(flat_momentum, alpha=args.SGDmomentum)
    else:
        flat_grad = flat_momentum
    flat_model = flat_model.add(flat_grad, alpha=-lr)
    make_model_unflattened(model,flat_model,net_sizes,ind_pairs)
    return None

def get_BN_mask(net,device):
    mask = torch.empty(0).to(device)
    for layer in net.modules():  # Prune only convolutional and linear layers
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer_weight = layer.weight
            len = layer_weight.numel()
            mask_ = torch.zeros(len,device=device)
            mask = torch.cat((mask, mask_), 0)
            if layer.bias is not None:
                bias = layer.bias.numel()
                mask_ = torch.ones(bias, device=device)
                mask = torch.cat((mask, mask_), 0)
        elif isinstance(layer, nn.BatchNorm2d):
            bn_params = 0
            for p in layer.parameters():
                bn_params += p.numel()
            mask_ = torch.ones(bn_params, device=device)
            mask = torch.cat((mask, mask_), 0)
    return mask




