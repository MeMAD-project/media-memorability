# Short term memorability score computation

Run

```aalto-predict.py --target short --hidden_size 80 --epochs 750 --picsom_features i3d-25-128-avg,audioset-527 --output i3d+audio_80_750
```


The data are read and organised as such : 

vid, lab, data_x, data_y = read_data(args)
Vid, data_y list obtained from 'data/2020/scores_v2.csv' and data/2020/test_urls.csv
data_x (for the entire dataset) and lab  from the the picsom features which were first extracted outside of the media-memorability repo and then uploaded to media-memorability/picsom/2020/
The test and train ids are extracted from  
dev    = picsom_class('picsom/'+year+'/classes/'+dev)
test   = picsom_class('picsom/'+year+'/classes/test')
The predictions are saved to --output
