from math import exp
import torch
from network import Network
from metric import valid
from torch.utils.data import Dataset
import numpy as np
import argparse
from loss import Loss
from dataloader import load_data
import os
import time
from tqdm import tqdm, trange
import torch.nn.functional as F
import torch.nn as nn
import math

from sklearn.cluster import KMeans, MiniBatchKMeans

 
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

Dataname = 'DHA'
# Dataname = 'NUSWIDE'
# Dataname = 'Caltech_5V'
# Dataname = 'Caltech'  # 7class
# Dataname = 'Caltech20class'
# Dataname = 'MSRCv1'
# Dataname = 'scene'
# Dataname = 'Fashion'
 

                         

parser = argparse.ArgumentParser(description='train')
parser.add_argument('--dataset', default=Dataname)
parser.add_argument('--batch_size', default=128, type=int)     # 256
parser.add_argument("--temperature_f", default=1.0)            # 1.0
parser.add_argument("--bi_level_iteration", default=4)         # 4
parser.add_argument("--times_for_K", default=1)                # 0.5 1 2 4
parser.add_argument("--Lambda", default=1)                     # 0.001 0.01 0.1 1 10 100 1000
parser.add_argument("--learning_rate", default=0.0003)         # 0.0003
parser.add_argument("--weight_decay", default=0.)              # 0.
parser.add_argument("--workers", default=8)                    # 8
parser.add_argument("--mse_epochs", default=100)               # 100 scene10/100; 2/300,500
parser.add_argument("--con_epochs", default=100)               # 100
parser.add_argument("--feature_dim", default=512)              # 512
parser.add_argument("--high_feature_dim", default=128)         # 128
args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if args.dataset == "DHA":
    args.con_epochs = 50
    args.bi_level_iteration = 1
    args.high_feature_dim = 512

if args.dataset == "NUSWIDE":
    args.con_epochs = 20 
    args.bi_level_iteration = 5 
    args.high_feature_dim = 96

if args.dataset == "Caltech":
    args.con_epochs = 50 
    args.bi_level_iteration = 3


if args.dataset == "Caltech_5V":
    args.con_epochs = 50 #70
    args.bi_level_iteration = 4 #4
    # or
    # args.bi_level_iteration = 3

if args.dataset == "Caltech20class":
    args.con_epochs = 25 
    args.bi_level_iteration = 4
    # or
    # args.bi_level_iteration = 3

if args.dataset == "MSRCv1":
    args.con_epochs = 100
    args.bi_level_iteration = 10

if args.dataset == "Fashion":
    args.con_epochs = 50 
    args.bi_level_iteration = 10 
    # Fashion2024-04-05 17:26:09

if args.dataset == "scene":
    args.con_epochs = 300 
    args.bi_level_iteration = 4 
    args.mse_epochs = 2



Total_con_epochs = args.con_epochs * args.bi_level_iteration


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True


accs = []
nmis = []
aris = []
purs = []
accs_bsv = []
nmis_bsv = []
ACC_tmp = 0
result_record = {"ACC": [], "NMI": [], "PUR": []}
for Runs in range(1):   # 10
    print("ROUND:{}".format(Runs+1))

    t1 = time.time()
    # setup_seed(5)   # if we find that the initialization of networks is sensitive, we can set a seed for stable performance.
    dataset, dims, view, data_size, class_num = load_data(args.dataset)

    data_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            # drop_last=True,
            drop_last=False,
        )


    def Low_level_rec_train():
        tot_loss = 0.
        criterion = torch.nn.MSELoss()


        for batch_idx, (xs, _, _) in enumerate(data_loader):
            for v in range(view):
                xs[v] = xs[v].to(device)

            xnum = xs[0].shape[0]              
            optimizer.zero_grad()
            zs_low, _, xrs, hs_low, _ = model(xs) #xxx hs
                
        
            loss_list = []
            for v in range(view):
                loss_list.append(criterion(xs[v], xrs[v]))
            loss = sum(loss_list) 
            loss.backward()
            optimizer.step()
            tot_loss += loss.item()
        return hs_low, zs_low

    def High_level_contrastive_train(epoch, nmi_matrix, sil_matrix, Lambda=1.0):
        tot_loss = 0.
        mes = torch.nn.MSELoss()      

        record_loss_con = []
        record_totloss_con = []
        Vones_full = []
        Vones_not_full = []

        best_view = np.argmax(sil_matrix)

        for v in range(view):
            record_loss_con.append([])


        for batch_idx, (xs, _, _) in enumerate(data_loader):
            for v in range(view):
                # print("xs", v, np.array(xs[v]).shape)
                xs[v] = xs[v].to(device)
                # print(cluster_feature)
                  

            optimizer.zero_grad()
            zs, qs, xrs, hs, re_h = model(xs)
            loss_list = []

            xnum = xs[0].shape[0]

            Lcc = []

            for v in range(view):
                    tmp = criterion.forward_feature_InfoNCE(zs[v], zs[best_view], batch_size=xnum) 
                    loss_list.append(tmp * nmi_matrix[v][best_view] * exp(sil_matrix[v]) * exp(sil_matrix[best_view])) 
                    record_loss_con[v].append(tmp) 
       
            for v in range(view):      
                loss_list.append(Lambda * mes(xs[v], xrs[v])) 


            loss = sum(loss_list)
            record_totloss_con.append(loss)

            loss.backward()
            optimizer.step()
            tot_loss += loss.item()

        return Vones_full, Vones_not_full, record_loss_con, record_totloss_con

    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')

    model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device)
    # print(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = Loss(args.batch_size, class_num, args.temperature_f, device).to(device)
    
    print("Initialization......")
    epoch = 0
    while epoch < args.mse_epochs:
        epoch += 1
        if epoch == 1:
           _, _ = Low_level_rec_train()
        else:
             hs_low, zs_low = Low_level_rec_train()
    acc, nmi, ari, pur, nmi_matrix_1, _, sil_matrix_h, sil_matrix_z = valid(model, device, dataset, view, data_size, class_num,
                                                eval_h=True, eval_z=False, times_for_K=args.times_for_K, 
                                                test=False, hs_low=hs_low, zs_low=zs_low)
    accs_bsv.append(acc)
    nmis_bsv.append(nmi)
    

    state = model.state_dict()
    time_point = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    torch.save(state, './checkpoints/' + args.dataset + time_point  + '.pth')
    
    print("Self-Weighted Multi-view Contrastive Learning with Reconstruction Regularization...")
    Iteration = 1
    print("Iteration " + str(Iteration) + ":")
    epoch = 0
    record_loss_con = []
    record_cos = []
    t = time.time()

    while epoch < Total_con_epochs:
        epoch += 1
        if epoch == 1:
            mask_ones_full, mask_ones_not_full, record_loss_con_, record_cos_ = High_level_contrastive_train(epoch,
                                                                                             nmi_matrix_1,
                                                                                             sil_matrix_h)
        else:
            _, _, record_loss_con_, record_cos_ = High_level_contrastive_train(epoch,
                                                                  nmi_matrix_1,
                                                                  sil_matrix_h
                                                                  )

        record_loss_con.append(record_loss_con_)
        record_cos.append(record_cos_)
        if epoch % args.con_epochs == 0:
            print("Total time elapsed: {:.2f}s".format(time.time() - t))
            # break

            if epoch == args.mse_epochs + Total_con_epochs:
                break

            # print(nmi_matrix_1)

            acc, nmi, ari, pur, _, nmi_matrix_2, sil_matrix_h, sil_matrix_z = valid(model, device, dataset, view, data_size, class_num,
                                                        eval_h=False, eval_z=True, times_for_K=args.times_for_K,
                                                        test=False, hs_low=hs_low, zs_low=zs_low)
            nmi_matrix_1 = nmi_matrix_2
            sil_matrix_h = sil_matrix_z
            if epoch < Total_con_epochs:
                Iteration += 1
                print("Iteration " + str(Iteration) + ":")
            if acc > ACC_tmp:
                ACC_tmp = acc
                state = model.state_dict()
                time_point = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

                torch.save(state, './checkpoints/' + args.dataset + time_point  + '.pth')    

        pg = [p for p in model.parameters() if p.requires_grad]
        #  this code matters, to re-initialize the optimizers
        optimizer = torch.optim.Adam(pg, lr=args.learning_rate, weight_decay=args.weight_decay)

    accs.append(acc)
    nmis.append(nmi)
    aris.append(ari)
    purs.append(pur)


    t2 = time.time()
    print("Time cost: " + str(t2 - t1))
    print('End......')


print(accs, np.mean(accs)/0.01, np.std(accs)/0.01)
print(nmis, np.mean(nmis)/0.01, np.std(nmis)/0.01)
# print(aris, np.mean(aris)/0.01, np.std(aris)/0.01)
# print(purs, np.mean(purs)/0.01, np.std(purs)/0.01)
