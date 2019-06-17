#! /usr/bin/env python3

import torch
import numpy as np
import scipy.stats
import re
import csv
import argparse

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


def main(args, vid, data_x, data_y):    
    global device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    
    ty = args.target
    s = [ty=='short' or ty=='both', ty=='long' or ty=='both']
    print('data_x =', data_x.shape, 'data_y =', data_y.shape, 'target =', ty)

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

    every_epoch = 1
    epochs = 5000

    D_in  = train_x.shape[1]
    H     = args.hidden_size
    D_out = train_y.shape[1]

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

    r_max = [0, 0]

    loss_fn = torch.nn.MSELoss(reduction='mean')
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for t in range(epochs+1):
        model.train()
        train_y_pred = model(train_x)
        loss = loss_fn(train_y_pred, train_y)
        # print('train', t, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if t%every_epoch==0:
            model.eval()
            val_y_pred = model(val_x)
            val_loss = loss_fn(val_y_pred, val_y)
            if False:
                print('train', t, loss.item())
                print('target ', train_y[0:10])
                print('predict', train_y_pred[0:10])
                print('VAL ', t, val_loss.item())
                print('target ', val_y[0:10])
                print('predict', val_y_pred[0:10])
            p0 = val_y_pred.detach().cpu()[:,0]
            g0 = val_y.cpu()[:,0]
            # print(p0[:10], g0[:10])
            r0 = scipy.stats.spearmanr(p0, g0).correlation
            r1 = 0

            if D_out==2:
                p1 = val_y_pred.detach().cpu()[:,1]
                g1 = val_y.cpu()[:,1]
                r1 = scipy.stats.spearmanr(p1, g1).correlation

            if False:
                print('{:7d} {:8.6f} {:8.6f} {:8.6f} {:8.6f}'.
                      format(t, loss.item(), val_loss.item(), r0, r1))

            if r0>r_max[0]:
                r_max[0] = r0

            if r1>r_max[1]:
                r_max[1] = r1

            if False:
                plt.scatter(p0, g0)
                plt.show()

    if ty=='both':
        print('max correlations short={:8.6f} long={:8.6f}'.format(r_max[0], r_max[1]))
    else:
        print('max correlation {}={:8.6f}'.format(ty, r_max[0]))

        
if __name__ == '__main__':
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
    parser.add_argument('--picsom_features', type=str,
                        default=picsom_def_feat, help="PicSOM features used")
    args = parser.parse_args()

    vid, data_x, data_y = read_data(args)
    main(args, vid, data_x, data_y)

    
