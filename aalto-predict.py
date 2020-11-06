#! /usr/bin/env python3

# final results with these:

# ./aalto-predict.py --target short --hidden_size 80 --epochs 750 \
#   --picsom_features i3d-25-128-avg,audioset-527 --output i3d+audio_80_750

# ./aalto-predict.py --target long --hidden_size 260 --epochs 160 \
#   --picsom_features i3d-25-128-avg,audioset-527 --output i3d+audio_260_160


import torch
import numpy as np
import scipy.stats
import re
import csv
import argparse
import sys
import pickle

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

folds_ids = pickle.load(open('folds_ids.pickle', 'br'))
#print(folds_ids)

def read_picsom_features(args):
    year   = '2020'
    dev = 'dev' if year=='2019' else 'training'
    labels = picsom_label_index('picsom/'+year+'/meta/labels.txt')
    dev    = picsom_class('picsom/'+year+'/classes/'+dev)
    test   = picsom_class('picsom/'+year+'/classes/test')

    # devi = sorted([ labels.index_by_label(i) for i in dev.objects() ])

    allobjects = dev.objects() | test.objects()
    
    alli = sorted([ labels.index_by_label(i) for i in allobjects ])

    lab = []
    for i in alli:
        lab.append(labels.label_by_index(i))
    
    print('picsom_features =', args.picsom_features)
    fx = []
    ff = args.picsom_features.split(',')
    for f in ff:
        feat = picsom_bin_data('picsom/'+year+'/features/'+f+'.bin')
        fdat = np.array(feat.get_float_list(alli))
        # print(year, f, fdat.shape)
        fx.append(fdat)

    if len(fx)>1:
        return np.concatenate(fx, axis=1), lab
    else:
        return np.array(fx[0]), lab
    
def read_data(args):
    year = '2020'
    vid    = []
    data_y = []

    if year=='2019':
        with open('data/2019/ground-truth_dev-set.csv', newline='') as csvfile:
            rows = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in rows:
                m = re.match('^(video.*webm).*$', row[0])
                if m:
                    v = m.group(1)
                    vid.append(v)
                    data_y.append([ float(row[1]), float(row[3]) ])

        with open('data/2019/test-set_videos-captions.txt', newline='') as testset:
            for row in testset:
                m = re.match('^(video.*webm).*$', row)
                if m:
                    vid.append(m.group(1))

    else:
        with open('data/2020/scores_v2.csv', newline='') as csvfile:
            rows = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in rows:
                m = re.match('^\d+$', row[0])
                if m:
                    vid.append(row[1])
                    data_y.append([ float(row[4]), float(row[5]) ])
                    
        with open('data/2020/test_urls.csv', newline='') as csvfile:
            rows = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in rows:
                m = re.match('^\d+$', row[0])
                if m:
                    vid.append(row[1])

    data_y      = np.array(data_y)
    #print(data_y[:5,:])
    data_x, lab = read_picsom_features(args)
    return (vid, lab, data_x, data_y)


def train_one(args, iii, t_x, t_y, v_x, v_y, nepochs, val_interval,
              target, output, v_f, jjj):
    D_in  = t_x.shape[1]
    H     = args.hidden_size
    D_out = t_y.shape[1]

    if iii==0:
        print('t_x =', t_x.shape, 't_y =', t_y.shape)
        print('v_x =', v_x.shape, 'v_y =', v_y.shape if v_y is not None else None)
        print('network structure', D_in, H, D_out)
        print('max epochs', nepochs)
    
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

        if val_interval==0 or t%val_interval==0:
            model.eval()
            r0 = 0
            r1 = 0
            v_y_pred = model(v_x)
            if v_y is not None:
                v_loss = loss_fn(v_y_pred, v_y)
                if False:
                    print('train', t, loss.item())
                    print('target ', t_y[0:10])
                    print('predict', t_y_pred[0:10])
                    print('VAL ', t, v_loss.item())
                    print('target ', v_y[0:10])
                    print('predict', v_y_pred[0:10])

            p0 = v_y_pred.detach().cpu()[:,0]
            if v_y is not None:
                g0 = v_y.cpu()[:,0]
                # print(p0[:10], g0[:10])
                r0 = scipy.stats.spearmanr(p0, g0).correlation

            if D_out==2:
                p1 = v_y_pred.detach().cpu()[:,1]
                if v_y is not None:
                    g1 = v_y.cpu()[:,1]
                    r1 = scipy.stats.spearmanr(p1, g1).correlation

            if False:
                print('{:7d} {:8.6f} {:8.6f} {:8.6f} {:8.6f}'.
                      format(t, loss.item(), v_loss.item(), r0, r1))

            if False:
                plt.scatter(p0, g0)
                plt.show()

            res.append([t, r0, r1])

            if output is not None:
                tasks = ['short', 'long'] if target=='both' else [target]
                for taskx in tasks:
                    # taskx = task if task=='long' else 'shor'
                    csv  = taskx+'_'+str(jjj)+'_'
                    csv += output+'.csv'
                    p = p1 if D_out==2 and task=='long' else p0
                    assert p.shape==v_f.shape
                    with open(csv, 'w') as fp:
                        for i in range(len(p)):
                            print(str(v_f[i])+','+str(p[i].item()), file=fp)
                    print('epoch', t, 'stored', p.shape[0], 'in <'+csv+'>') 
    
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

def show_result(i, r0, e0, r1, e1, target, H, f):
    if target=='both':
        print('{} max correlations short={:8.6f} (epoch {:d}) long={:8.6f} (epoch {:d}) h={:d} {}'.
              format(i, r0, e0, r1, e1, H, f))
    else:
        print('{} max correlation {}={:8.6f} (epoch {:d}) h={:d} {}'.format(i, target, r0, e0, H, f))


def fold_old(i, n, x):
    assert i>=0 and n>i
    v = x.nonzero()[0]
    s = len(v)
    a = i*s//n
    b = (i+n-1)*s//n if n>1 else s
    r = np.zeros_like(x, dtype=bool)
    for j in range(a, b):
        r[v[j%s]] = True
    return r


def get_folds_old(nfolds, n):
    folds = [ [False]*n for i in range(nfolds) ]
    #print(len(folds), len(folds[0]))
    for j in range(n):
        i = j*nfolds // n
        folds[i][j] = True
    return folds


def get_folds(nfolds, n, ll):
    folds = [ [False]*n for i in range(nfolds) ]
    # print(len(folds), len(folds[0]), len(ll), ll)
    for j in range(n):
        v = int(ll[j])
        for i in range(nfolds):
            if v in folds_ids[i]:
                folds[i][j] = True
    # a = np.array(folds)
    
    return folds


def main(args, vid, lab, data_x, data_y):    
    global device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')

    target = args.target
    s = [target=='short' or target=='both', target=='long' or target=='both']
    print('data_x =', data_x.shape, 'data_y =', data_y.shape, 'target =', target,
          'train_fold =', args.train_fold, '#vid =', len(vid))

    m = re.match('^(\d+)/(\d+)$', args.train_fold)
    assert m, 'train_fold should be (\d+)/(\d+)'
    f_i = int(m.group(1))
    f_n = int(m.group(2))

    numbers = []
    for l in lab:
        numbers.append(int(l))
    numbers = np.array(numbers, dtype=np.int32)

    nall   = data_x.shape[0]
    ndev   = data_y.shape[0]
    ntrain = ndev*(f_n-1)//f_n if f_n>1 else ndev
    nval   = ndev-ntrain
    assert ntrain>0 and nval>=0, \
        'train-split {} + {} = {} does not make sense'.format(ntrain, nval, ndev)

    iallx   = np.array(range(nall)) ; ially   = iallx[:ndev]
    idevx   = iallx < ndev          ; idevy   = idevx[:ndev]
#   itrainx = fold(f_i, f_n, idevx) ; itrainy = itrainx[:ndev]
    itrainx = np.array([True]*ndev+[False]*(nall-ndev))
    itrainy = itrainx[:ndev]
    ivalx   = idevx & ~itrainx      ; ivaly   = ivalx[:ndev]

    dtype   = torch.float
    train_x = torch.tensor(data_x[itrainx,:],    device=device, dtype=dtype)
    train_y = torch.tensor(data_y[itrainy][:,s], device=device, dtype=dtype)
    val_x   = torch.tensor(data_x[ivalx,:],      device=device, dtype=dtype)
    val_y   = torch.tensor(data_y[ivaly][:,s],   device=device, dtype=dtype)
    val_f   = np.array(vid)[ivalx]

    test_x  = torch.tensor(data_x[~idevx,:],     device=device, dtype=dtype)
    test_f  = np.array(vid)[~idevx]
    
    train_n = numbers[itrainx]
    test_n  = numbers[~itrainx]

    print('train_x =', train_x.shape, 'train_y =', train_y.shape)
    print('val_x =',   val_x.shape,   'val_y =',   val_y.shape)
    print('#val_f =',  len(val_f),    '#test_f =', len(test_f))
    sys.stdout.flush()
    
    epochs = args.epochs
    val_interval = args.val_interval
    nfolds = args.folds
    folds = get_folds(nfolds, train_x.shape[0], lab)

    res = []
    for i in range(nfolds):
        s = folds[i]
        r = [ not j for j in s ]
        t_x = train_x[r,:]
        t_y = train_y[r,:]
        v_x = train_x[s,:]
        v_y = train_y[s,:]
        v_n = train_n[s]
        r = train_one(args, i, t_x, t_y, v_x, v_y, epochs, val_interval, target, args.output, v_n, i)
        res.append(r)
        r0, e0, r1, e1 = solve_max(r)
        show_result(i, r0, e0, r1, e1, target, args.hidden_size, args.picsom_features)
        sys.stdout.flush()

    avg = average_results(res)
    r0, e0, r1, e1 = solve_max(avg)
    show_result('AVER.', r0, e0, r1, e1, target, args.hidden_size, args.picsom_features)
    sys.stdout.flush()
    print(r0, e0, r1, e1)

    # r = train_one(args, 0, train_x, train_y, val_x, val_y, e0, e0, target, args.output, val_f)
    # r0v, e0v, r1v, e1v = solve_max(r)
    # show_result('FINAL', r0v, e0v, r1v, e1v, target, args.hidden_size, args.picsom_features)
    # sys.stdout.flush()

    r = train_one(args, 0, train_x, train_y, test_x, None, epochs, epochs, target, args.output, test_n, 6)
    r0t, e0t, r1t, e1t = solve_max(r)
    show_result('TEST', r0t, e0t, r1t, e1t, target, args.hidden_size, args.picsom_features)
    sys.stdout.flush()

        
if __name__ == '__main__':
    # print(sys.version, torch.__version__)

    pf_rn101   = 'c_in12_rn101_pool5o_d_a'
    pf_rn152   = 'c_in12_rn152_pool5o_d_a'
    pf_sun101  = 'sun-397-c_in12_rn101_pool5o_d_a'
    pf_sun152  = 'sun-397-c_in12_rn152_pool5o_d_a'
    pf_coco101 = 'coco-80-c_in12_rn101_pool5o_d_a'
    pf_coco152 = 'coco-80-c_in12_rn152_pool5o_d_a'
    pf_i3d     = 'i3d-25-128-avg'
    pf_c3d     = 'c3d-rn18-s1m-pool5-a'
    pf_audio   = 'audioset-527'
    
    pf_rs     = ','.join([pf_rn152, pf_sun152])
    pf_rsc    = ','.join([pf_rn152, pf_sun152, pf_coco152])
    pf_rscc   = ','.join([pf_rn152, pf_sun152, pf_coco152, pf_coco101])
    pf_rrss   = ','.join([pf_rn152, pf_rn101, pf_sun152, pf_sun101])
    pf_rrssi  = ','.join([pf_rn152, pf_rn101, pf_sun152, pf_sun101, pf_i3d])
    pf_rrsscc = ','.join([pf_rn152, pf_rn101, pf_sun152, pf_sun101, pf_coco152, pf_coco101])
    picsom_def_feat = pf_rrssi
    picsom_def_feat = pf_i3d

    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action="store_true",
                        help="Use CPU even when GPU is available")
    parser.add_argument('--ntrain', type=int, default=0,
                        help="Number of training samples used")
    parser.add_argument('--hidden_size', type=int, default=430,
                        help="Hidden layer size, default=%(default)s")
    parser.add_argument('--target', type=str, default='both',
                        help="Predicted variable: short, long or both (default)")
    parser.add_argument('--folds', type=int, default=6,
                        help="Number folds in cross-validation, default=%(default)i")
    parser.add_argument('--epochs', type=int, default=1000,
                        help="Number of epochs in training, default=%(default)i")
    parser.add_argument('--val_interval', type=int, default=10,
                        help="Interval between validations, default=%(default)i")
    parser.add_argument('--picsom_features', type=str,
                        default=picsom_def_feat, help="PicSOM features used, default=%(default)s")
    parser.add_argument('--output', type=str,
                        default=None, help="output file for external evaluation, default=%(default)s")
    parser.add_argument('--train_fold', type=str,
                        default='0/4', help="training set fold#/nfolds, default=%(default)s")
    args = parser.parse_args()

    if args.ntrain:
        print('--ntrain deprecated, use instead --train_fold=0/4')
        exit(1)
    
    vid, lab, data_x, data_y = read_data(args)
    main(args, vid, lab, data_x, data_y)

    
