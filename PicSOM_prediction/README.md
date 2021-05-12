# PicSOM score computation

## Short term memorability score computation

Run

```
./aalto-predict.py --target short --hidden_size 80 --epochs 750 \
    --picsom_features i3d-25-128-avg,audioset-527 --output i3d+audio_80_750
```


The data are read and organised as such : 

vid, lab, data_x, data_y = read_data(args)
Vid, data_y list obtained from 'data/2020/scores_v2.csv' and data/2020/test_urls.csv
data_x (for the entire dataset) and lab  from the the picsom features which were first extracted outside of the media-memorability repo and then uploaded to media-memorability/picsom/2020/
The test and train ids are extracted from  
dev    = picsom_class('picsom/'+year+'/classes/'+dev)
test   = picsom_class('picsom/'+year+'/classes/test')
The predictions are saved to --output


## Long term memorability score computation

Run

```
./aalto-predict.py --target long --hidden_size 260 --epochs 160 \
    --picsom_features i3d-25-128-avg,audioset-527 --output i3d+audio_260_160

```


## Applying the model to external data

Run

```
./aalto-predict.py --target short --hidden_size 80 --epochs 750 \
    --picsom_features i3d-25-128-avg,audioset-527 --output i3d+audio_80_750 --extra surrey20
```

which will create file `short_6_i3d+audio_80_750-surrey20.csv` containing the short-term predictions for the `surrey20` data set.


## Predictions for other videos

### Install the PicSOM software

Download https://github.com/aalto-cbir/PicSOM

Read and follow PicSOM's [README.md](https://github.com/aalto-cbir/PicSOM/blob/master/README.md).

### Create a database

Use PicSOM's `analyse=insert` mode.

### Extract features

Use PicSOM's `analyse=create extractfeatures=true` mode.

### Export features for memorability prediction

Use PicSOM's `analyse=exportorderedfeatures` mode.

