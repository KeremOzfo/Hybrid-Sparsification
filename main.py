from train_funcs import *
import numpy as np
from parameters import *
import torch
import random
import datetime
import os

device = torch.device("cpu")
args = args_parser()

if __name__ == '__main__':
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    simulation_ID = int(random.uniform(1,999))
    print('device:',device)
    args = args_parser()
    for arg in vars(args):
       print(arg, ':', getattr(args, arg))
    x = datetime.datetime.now()
    date = x.strftime('%b') + '-' + str(x.day)
    if args.mode == 'top-K':
        simulation = 'TopK-LocalIter_{}-K_{}'.format(args.LocalIter,args.sparsity)
    else:
        simulation = 'Hyprid_sparse-LocalIter_{}-sparse_K_{}_C{}_freq-K_{}'.format(
            args.LocalIter,args.sparsity,args.C_rand,args.freq_K)
    newFile = simulation
    if not os.path.exists(os.getcwd() + '/Results'):
        os.mkdir(os.getcwd() + '/Results')
    n_path = os.path.join(os.getcwd(), 'Results', newFile)
    n_path_acc = n_path + '/acc'
    for i in range(5):
        if args.mode == 'top-K':
            accs = train_topk(args,device)
        else:
            accs = train(args, device)
        if i == 0:
            os.mkdir(n_path)
            os.mkdir(n_path_acc)
            f = open(n_path + '/simulation_Details.txt', 'w+')
            f.write('simID = ' + str(simulation_ID) + '\n')
            f.write('############## Args ###############' + '\n')
            for arg in vars(args):
                line = str(arg) + ' : ' + str(getattr(args, arg))
                f.write(line + '\n')
            f.write('############ Results ###############' + '\n')
            f.close()
        s_loc = 'hybrid_spars' + '--' + str(i)
        s_loc = os.path.join(n_path_acc,s_loc)
        np.save(s_loc,accs)
        f = open(n_path + '/simulation_Details.txt', 'a+')
        f.write('Trial ' + str(i) + ' results at ' + str(accs[len(accs)-1]) + '\n')
        f.close()