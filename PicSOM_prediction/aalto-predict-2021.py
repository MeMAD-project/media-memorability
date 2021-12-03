#! /usr/bin/env python3

# ./aalto-predict-2021.py --train trecvid/train/short
# ./aalto-predict-2021.py --train trecvid/train/short --test trecvid/dev/short

# ./aalto-predict-2021.py --train memento/train/short --epochs 100

# final results with these:

# Run 2 Early fusion audio + visual + text
# ./aalto-predict-2021.py --train trecvid/train/short --test trecvid/test/short --hidden_size 560 --features i3d-25-128-avg,audioset-527,bert3 --epochs 300 --output run2
# => trecvid_short_260_6_run2.csv
# ./aalto-predict-2021.py --train trecvid/train/norm  --test trecvid/test/norm  --hidden_size 160 --features i3d-25-128-avg,audioset-527,bert3 --epochs 400 --output run2
# => trecvid_norm_320_6_run2.csv
# ./aalto-predict-2021.py --train trecvid/train/long  --test trecvid/test/long  --hidden_size 700 --features i3d-25-128-avg,audioset-527,bert3 --epochs 150 --output run2
# => trecvid_long_80_6_run2.csv
# ./aalto-predict-2021.py --train memento/train/short --test memento/test/short --hidden_size 720 --features i3d-25-128-avg,audioset-527,bert3 --epochs 700 --output run2
# => memento_short_610_6_run2.csv
# ./aalto-predict-2021.py --train memento/train/norm  --test memento/test/norm  --hidden_size 720 --features i3d-25-128-avg,audioset-527,bert3 --epochs 700 --output run2
# => memento_norm_600_6_run2.csv

# Run 3 Early fusion visual + text
# ./aalto-predict-2021.py --train trecvid/train/short --test trecvid/test/short --hidden_size 340 --features i3d-25-128-avg,bert3 --epochs 300 --output run3
# => trecvid_short_300_6_run3.csv
# ./aalto-predict-2021.py --train trecvid/train/norm  --test trecvid/test/norm  --hidden_size 340 --features i3d-25-128-avg,bert3 --epochs 300 --output run3
# => trecvid_norm_260_6_run3.csv
# ./aalto-predict-2021.py --train trecvid/train/long  --test trecvid/test/long  --hidden_size 480 --features i3d-25-128-avg,bert3 --epochs 150 --output run3
# => trecvid_long_70_6_run3.csv
# ./aalto-predict-2021.py --train memento/train/short --test memento/test/short --hidden_size 800 --features i3d-25-128-avg,bert3 --epochs 650 --output run3
# => memento_short_580_6_run3.csv
# ./aalto-predict-2021.py --train memento/train/norm  --test memento/test/norm  --hidden_size 800 --features i3d-25-128-avg,bert3 --epochs 680 --output run3
# => memento_norm_610_6_run3.csv

# Run 5 optimal memento training parameters applied to combined training data 
# ./aalto-predict-2021.py --folds 0 --train combined/train/short --test trecvid/test/short --hidden_size 720 --features i3d-25-128-avg,audioset-527,bert3 --epochs 610 --output run5
# => combined_short_610_6_run5.csv => trecvid_short_610_6_run5.csv
# ./aalto-predict-2021.py --folds 0 --train combined/train/short --test memento/test/short --hidden_size 720 --features i3d-25-128-avg,audioset-527,bert3 --epochs 610 --output run5
# => combined_short_610_6_run5.csv => memento_short_610_6_run5.csv
# ./aalto-predict-2021.py --folds 0 --train combined/train/norm  --test trecvid/test/norm  --hidden_size 720 --features i3d-25-128-avg,audioset-527,bert3 --epochs 600 --output run5
# => combined_norm_600_6_run5.csv => trecvid_norm_600_6_run5.csv
# ./aalto-predict-2021.py --folds 0 --train combined/train/norm  --test memento/test/norm  --hidden_size 720 --features i3d-25-128-avg,audioset-527,bert3 --epochs 600 --output run5
# => combined_norm_600_6_run5.csv => memento_norm_600_6_run5.csv

# sub task
# ./aalto-predict-2021.py --folds 0 --train memento/train/short --test trecvid/test/short --hidden_size 720 --features i3d-25-128-avg,audioset-527,bert3 --epochs 610 --output sub-run2
# => memento_short_610_6_sub-run2.csv => trecvid_short_610_6_sub-run2.csv
# ./aalto-predict-2021.py --folds 0 --train memento/train/norm  --test trecvid/test/norm  --hidden_size 720 --features i3d-25-128-avg,audioset-527,bert3 --epochs 600 --output sub-run2
# => memento_norm_600_6_sub-run2.csv => trecvid_norm_600_6_sub-run2.csv
# ./aalto-predict-2021.py --folds 0 --train trecvid/train/short --test memento/test/short --hidden_size 560 --features i3d-25-128-avg,audioset-527,bert3 --epochs 260 --output sub-run2
# => trecvid_short_260_6_sub-run2.csv => memento_short_260_6_sub-run2.csv
# ./aalto-predict-2021.py --folds 0 --train trecvid/train/norm  --test memento/test/norm  --hidden_size 160 --features i3d-25-128-avg,audioset-527,bert3 --epochs 320 --output sub-run2
# => trecvid_norm_320_6_sub-run2.csv => memento_norm_320_6_sub-run2.csv

import torch
import numpy as np
import scipy.stats
import re
import csv
import argparse
import sys
import pickle
import os

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

def read_features(args, s, t, tt, extra=None):
    # print(args.features, s, t, tt)
    labels = picsom_label_index('picsom/2021/meta/labels.txt')
    cls    = picsom_class('picsom/2021/classes/'+s+'-'+t)
    print('read', labels.path(), cls.path())
    
    # devi = sorted([ labels.index_by_label(i) for i in dev.objects() ])

    allobjects = cls.objects()
    
    alli = sorted([ labels.index_by_label(i) for i in allobjects ])

    lab = []
    for i in alli:
        lab.append(labels.label_by_index(i))
    
    #print('features =', args.features)
    fx = []
    ff = args.features.split(',')
    for f in ff:
        picsfeat = 'picsom/2021/features/'+f+'.bin'
        if os.path.isfile(picsfeat):
            feat = picsom_bin_data('picsom/2021/features/'+f+'.bin')
            fdat = np.array(feat.get_float_list(alli))
            print('read', feat.path(), fdat.shape)
        else:
            assert False, 'feature file <'+picsfeat+'> does not exist'
        fx.append(fdat)

    if len(fx)>1:
        return np.concatenate(fx, axis=1), lab
    else:
        return np.array(fx[0]), lab


def read_scorefile(f, s, o):
    vid    = []
    data_y = []
    with open(f, newline='') as csvfile:
        rows = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in rows:
            m = re.match('^\d+$', row[0])
            if m:
                #print(row[1], row[4], row[5], row[6])
                vid.append(row[1])
                if s=='trecvid' and o=='short':
                    data_y.append([float(row[4])])
                if s=='trecvid' and o=='long':
                    data_y.append([float(row[5])])
                if s=='trecvid' and o=='norm':
                    data_y.append([float(row[6])])
                if s=='memento' and o=='short':
                    data_y.append([float(row[3])])
                if s=='memento' and o=='norm':
                    data_y.append([float(row[4])])

    return (vid, data_y)


def read_data(args, sto):
    s, t, o = sto
    vid    = []
    data_y = []

    print('reading data for', s, t, o)
    ttx = { 'train': 'training', 'dev': 'development', 'test': 'testing' }
    assert t in ttx
    tt = ttx[t]
    xsl = [ s ]
    if s=='combined':
        xsl = ['trecvid', 'memento']
    for xs in xsl:
        f = 'data/2021/{}/{}_set/{}_scores.csv'.format(xs, tt, t)
        e = os.path.isfile(f)
        print(f, e)
        if e:
            zzz = read_scorefile(f, xs, o)
            # print(zzz)
            vid.extend(zzz[0])
            data_y.extend(zzz[1])

    data_y      = np.array(data_y)
    # print(data_y[:5,:])
    data_x, lab = read_features(args, s, t, tt)
    return (vid, lab, data_x, data_y)


def train_one(args, iii, t_x, t_y, v_x, v_y, nepochs, val_interval,
              dset, target, output, v_f, jjj):
    D_in  = t_x.shape[1]
    H     = args.hidden_size
    D_out = 1

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
        # print(t_y_pred.shape, t_y.shape)
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
                g0 = v_y.cpu()
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
                csv  = dset+'_'+target+'_'+str(t)+'_'+str(jjj)+'_'+output+'.csv'
                assert p0.shape==v_f.shape, str(p0.shape)+'!='+str(v_f.shape) 
                with open(csv, 'w') as fp:
                    for i in range(len(p0)):
                        print(str(v_f[i])+','+str(p0[i].item()), file=fp)
                print('epoch', t, 'stored', p0.shape[0], 'in <'+csv+'>') 
    
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

def show_result(i, r0, e0, r1, e1, dset, target, H, f):
    if target=='both':
        print('{} max correlations short {:8.6f} (epoch {:d}) long={:8.6f} (epoch {:d}) h={:d} {}'.
              format(i, r0, e0, r1, e1, H, f))
    else:
        print('{} max correlation {} {} {:8.6f} (epoch {:d}) h={:d} {}'.format(i, dset, target, r0, e0, H, f))


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


def get_folds(s, nfolds, n, ll):
    if s=='memento':
        folds_ids_file = 'memento10k_6-folds_eval_ids.pickle'
    if s=='trecvid':
        folds_ids_file = 'trecvid_6-folds_eval_ids.pickle'

    print('reading fold spec from', folds_ids_file)
    folds_ids = pickle.load(open(folds_ids_file, 'br'))

    folds = [ [False]*n for i in range(nfolds) ]
    #print('get_folds() :', s, len(folds), len(folds[0]), len(ll), ll)
    for j in range(n):
        v = int(ll[j][1:])
        for i in range(nfolds):
            if v in folds_ids[i]:
                folds[i][j] = True
    # a = np.array(folds)
    
    return folds


def main(args, tx, t_vid, t_lab, t_data_x, t_data_y,
         ex, e_vid, e_lab, e_data_x, e_data_y):
    global device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    tset, target = (tx[0], tx[2])
    print('train data_x =', t_data_x.shape, 'data_y =', t_data_y.shape, 'target =', target,
          '#vid =', len(t_vid))

    edys = e_data_y.shape if e_data_y is not None else None
    print('test  data_x =', e_data_x.shape, 'data_y =', edys, 'target =', target,
          '#vid =', len(e_vid))

    assert t_data_x.shape[0]==t_data_y.shape[0], 't_data_x.shape[0]=='+str(t_data_x.shape[0])+' != t_data_y.shape[0]=='+str(t_data_y.shape[0])
    nall = t_data_x.shape[0]
    ndev = nall

    dtype   = torch.float
    train_x = torch.tensor(t_data_x, device=device, dtype=dtype)
    train_y = torch.tensor(t_data_y, device=device, dtype=dtype)

    test_x  = torch.tensor(e_data_x,  device=device, dtype=dtype)
    test_y  = torch.tensor(e_data_y,  device=device, dtype=dtype) if e_data_y is not None else None
    test_f  = np.array(e_vid)
    
    numbers_t = [int(l) for l in t_lab]
    train_n = np.array(numbers_t, dtype=np.int32)
    numbers_e = [int(l[1:]) for l in e_lab]
    test_n = np.array(numbers_e, dtype=np.int32)

    print('train_x =', train_x.shape, 'train_y =', train_y.shape,
          '#test_f =', len(test_f))
    sys.stdout.flush()
    
    epochs = args.epochs
    val_interval = args.val_interval
    nfolds = args.folds
    assert nfolds==0 or nfolds==6

    if nfolds==6:
        folds = get_folds(tset, nfolds, train_x.shape[0], t_lab)

    res = []
    for i in range(nfolds):
        s = folds[i]
        r = [ not j for j in s ]
        t_x = train_x[r,:]
        t_y = train_y[r]
        v_x = train_x[s,:]
        v_y = train_y[s]
        v_n = train_n[s]
        r = train_one(args, i, t_x, t_y, v_x, v_y, epochs, val_interval, tset, target, args.output, v_n, i)
        res.append(r)
        r0, e0, r1, e1 = solve_max(r)
        show_result(i, r0, e0, r1, e1, tset, target, args.hidden_size, args.features)
        sys.stdout.flush()

    avg = average_results(res)
    r0, e0, r1, e1 = solve_max(avg)
    show_result('AVER.', r0, e0, r1, e1, tset, target, args.hidden_size, args.features)
    sys.stdout.flush()
    #print(r0, e0, r1, e1)

    #eps = epochs
    eps = e0
    if nfolds==0 and eps==-1:
        eps = epochs

    r = train_one(args, 0, train_x, train_y, test_x, test_y, eps, eps, tset, target, args.output, test_n, 6)
    r0t, e0t, r1t, e1t = solve_max(r)
    show_result('TEST', r0t, e0t, r1t, e1t, tset, target, args.hidden_size, args.features)
    sys.stdout.flush()
    print('TEST fin correlation {} {} {:8.6f} (epoch {}) h={} {}'.
          format(tset, target, r[-1][1], r[-1][0], args.hidden_size, args.features))
    # print(r)

    # if e_data_x.shape[0]!=0:
    #     print('EXTRA test with', ex)
    #     xtra_data_x = torch.tensor(e_data_x, device=device, dtype=dtype)
    #     xtra_data_y = torch.tensor(e_data_y, device=device, dtype=dtype)
    #     extra_lab = np.array(e_lab)
    #     outf = args.output
    #     if outf is not None:
    #         outf += '-'+'zzz'
    #     r = train_one(args, 0, train_x, train_y, xtra_data_x, xtra_data_y, epochs, epochs, tset, target, outf, extra_lab, 6)
    #     r0x, e0x, r1x, e1x = solve_max(r)
    #     show_result('EXTRA', r0x, e0x, r1x, e1x, tset, target, args.hidden_size, args.features)
    #     sys.stdout.flush()
        
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
    def_feat='C3D'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action="store_true",
                        help="Use CPU even when GPU is available")
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
    parser.add_argument('--features', type=str,
                        default=def_feat, help="Features used, default=%(default)s")
    parser.add_argument('--output', type=str,
                        default=None, help="output file for external evaluation, default=%(default)s")
    parser.add_argument('--train', type=str, help="(trecvid|memento)/(train|dev|test)/(short|long|norm)")
    parser.add_argument('--test',  type=str, help="(trecvid|memento)/(train|dev|test)/(short|long|norm)")
    args = parser.parse_args()

    train_3 = args.train.split('/')
    assert len(train_3)==3
    vid_1, lab_1, x_1, y_1 = read_data(args, train_3)
    vid_2, lab_2, x_2, y_2 = ([], [], np.empty((0, 0)), np.empty((0, 0)))
    test_3 = ('', '', '')

    if args.test is not None and args.test!='//':
        test_3 = args.test.split('/')
        assert len(test_3)==3
        vid_2, lab_2, x_2, y_2 = read_data(args, test_3)
        if y_2.shape==(0,):
            y_2 = None
    main(args, train_3, vid_1, lab_1, x_1, y_1, test_3, vid_2, lab_2, x_2, y_2)

    
