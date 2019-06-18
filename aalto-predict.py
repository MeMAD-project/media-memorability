#! /usr/bin/env python3

import torch
import numpy as np
import scipy.stats
import re
import csv
import argparse
import sys

#import matplotlib
## matplotlib.use('TkAgg')
#import matplotlib.pyplot as plt

from picsom.code.label_index import picsom_label_index
from picsom.code.class_file  import picsom_class
from picsom.code.bin_data    import picsom_bin_data

device = None

torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def read_picsom_features(args):
    labels = picsom_label_index('picsom/meta/labels.txt')
    dev    = picsom_class('picsom/classes/dev')
    test   = picsom_class('picsom/classes/test')

    devi = sorted([ labels.index_by_label(i) for i in dev.objects() ])
    
    fx = []
    ff = args.picsom_features.split(',')
    for f in ff:
        feat = picsom_bin_data('picsom/features/'+f+'.bin')
        fdat = np.array(feat.get_float_list(devi))
        fx.append(fdat)

    return np.concatenate(fx, axis=1)

    
def read_data(args):
    vid    = []
    data_y = []
    with open('ground-truth_dev-set.csv', newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in rows:
            m = re.match('^(video.*webm).*$', row[0])
            if m:
                v = m.group(1)
                vid.append(v)
                data_y.append([ float(row[1]), float(row[3]) ])

    data_y = np.array(data_y)
    data_x = read_picsom_features(args)
    return (vid, data_x, data_y)


def train_one(args, i, t_x, t_y, v_x, v_y, nepochs, val_interval):
    D_in  = t_x.shape[1]
    H     = args.hidden_size
    D_out = t_y.shape[1]

    if i==0:
        print('t_x =', t_x.shape, 't_y =', t_y.shape)
        print('v_x =', v_x.shape, 'v_y =', v_y.shape)
        print('network structure', D_in, H, D_out)
    
    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Dropout(),
        torch.nn.Linear(H, D_out),
        torch.nn.Sigmoid(),
    )
    
    if str(device)!='cpu':
        model.cuda(device=device)

    loss_fn = torch.nn.MSELoss(reduction='mean')
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    res = []
    
    for t in range(nepochs+1):
        model.train()
        t_y_pred = model(t_x)
        loss = loss_fn(t_y_pred, t_y)
        # print('train', t, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t%val_interval==0:
            model.eval()
            v_y_pred = model(v_x)
            v_loss = loss_fn(v_y_pred, v_y)
            if False:
                print('train', t, loss.item())
                print('target ', t_y[0:10])
                print('predict', t_y_pred[0:10])
                print('VAL ', t, v_loss.item())
                print('target ', v_y[0:10])
                print('predict', v_y_pred[0:10])
            p0 = v_y_pred.detach().cpu()[:,0]
            g0 = v_y.cpu()[:,0]
            # print(p0[:10], g0[:10])
            r0 = scipy.stats.spearmanr(p0, g0).correlation
            r1 = 0

            if D_out==2:
                p1 = v_y_pred.detach().cpu()[:,1]
                g1 = v_y.cpu()[:,1]
                r1 = scipy.stats.spearmanr(p1, g1).correlation

            if False:
                print('{:7d} {:8.6f} {:8.6f} {:8.6f} {:8.6f}'.
                      format(t, loss.item(), v_loss.item(), r0, r1))

            if False:
                plt.scatter(p0, g0)
                plt.show()

            res.append([t, r0, r1])

    return res


def solve_max(r):
    r_max = [ 0,  0]
    e_max = [-1, -1]

    for t, r0, r1 in r:
        if r0>r_max[0]:
            r_max[0] = r0
            e_max[0] = t
                
        if r1>r_max[1]:
            r_max[1] = r1
            e_max[1] = t

    return r_max[0], e_max[0], r_max[1], e_max[1]
            

def average_results(rr):
    res = []
    for i in range(len(rr)):
        r = rr[i]
        for j in range(len(r)):
            t, r0, r1 = r[j]
            if i==0:
                res.append([t, 0, 0])
            res[j][1] += r0
            res[j][2] += r1
    for j in range(len(res)):
        res[j][1] /= len(rr)
        res[j][2] /= len(rr)

    return res

        
def main(args, vid, data_x, data_y):    
    global device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    
    target = args.target
    s = [target=='short' or target=='both', target=='long' or target=='both']
    print('data_x =', data_x.shape, 'data_y =', data_y.shape, 'target =', target)

    ntrain = args.ntrain
    nval   = data_x.shape[0]-ntrain
    assert ntrain>0 and nval>0, \
        'train-split {} {} does not make sense'.format(ntrain, nval)

    dtype = torch.float
    train_x = torch.tensor(data_x[:ntrain,:], device=device, dtype=dtype)
    train_y = torch.tensor(data_y[:ntrain,s], device=device, dtype=dtype)
    val_x   = torch.tensor(data_x[-nval:,:],  device=device, dtype=dtype)
    val_y   = torch.tensor(data_y[-nval:,s],  device=device, dtype=dtype)

    print('train_x =', train_x.shape, 'train_y =', train_y.shape)
    print('val_x =',   val_x.shape,   'val_y =',   val_y.shape)

    epochs = args.epochs
    val_interval = args.val_interval
    nfolds = args.folds
    folds = [ [ False ] *train_x.shape[0]  for i in range(nfolds) ]
    for j in range(train_x.shape[0]):
        i = j*nfolds // train_x.shape[0]
        folds[i][j] = True

    res = []
    for i in range(nfolds):
        s = folds[i]
        r = [ not j for j in s ]
        t_x = train_x[r,:]
        t_y = train_y[r,:]
        v_x = train_x[s,:]
        v_y = train_y[s,:]
        r = train_one(args, i, t_x, t_y, v_x, v_y, epochs, val_interval)
        res.append(r)
        r0, e0, r1, e1 = solve_max(r)
        if target=='both':
            print('{} max correlations short={:8.6f} (epoch {:d}) long={:8.6f} (epoch {:d})'.
                  format(i, r0, e0, r1, e1))
        else:
            print('{} max correlation {}={:8.6f} (epoch {:d})'.format(i, target, r0, e0))

    avg = average_results(res)
    r0, e0, r1, e1 = solve_max(avg)
    if target=='both':
        print('{} max correlations short={:8.6f} (epoch {:d}) long={:8.6f} (epoch {:d})'.
              format('AVG', r0, e0, r1, e1))
    else:
        print('{} max correlation {}={:8.6f} (epoch {:d})'.format('AVG', target, r0, e0))

    r = train_one(args, 0, train_x, train_y, val_x, val_y, e0, e0)
    r0, e0, r1, e1 = solve_max(r)
    if target=='both':
        print('{} max correlations short={:8.6f} (epoch {:d}) long={:8.6f} (epoch {:d})'.
              format('FIN', r0, e0, r1, e1))
    else:
        print('{} max correlation {}={:8.6f} (epoch {:d})'.format('FIN', target, r0, e0))
    
        
if __name__ == '__main__':
    # print(sys.version, torch.__version__)
    
    picsom_def_feat = 'c_in12_rn152_pool5o_d_a,sun-397-c_in12_rn152_pool5o_d_a'
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action="store_true",
                        help="Use CPU even when GPU is available")
    parser.add_argument('--ntrain', type=int, default=6000,
                        help="Number of training samples used")
    parser.add_argument('--hidden_size', type=int, default=430,
                        help="Hidden layer size")
    parser.add_argument('--target', type=str, default='both',
                        help="Predicted variable: short, long or both (default) ")
    parser.add_argument('--folds', type=int, default=10,
                        help="Number folds in cross-validation, default=10")
    parser.add_argument('--epochs', type=int, default=1000,
                        help="Number of epochs in training, default=1000")
    parser.add_argument('--val_interval', type=int, default=10,
                        help="Interval between vaolidations, default=10")
    parser.add_argument('--picsom_features', type=str,
                        default=picsom_def_feat, help="PicSOM features used")
    args = parser.parse_args()

    vid, data_x, data_y = read_data(args)
    main(args, vid, data_x, data_y)

    
