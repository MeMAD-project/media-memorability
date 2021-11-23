import pickle
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm.notebook import tqdm
spearman = lambda x,y: spearmanr(x, y).correlation

df_scores = pd.read_csv('/home/semantic/media-memorability/textual_scores/data_2021/TRECVid Data/training_set/train_scores.csv')

                        
                        
 # Ground Truth
scores_st = df_scores['scores_raw_short_term'].values.tolist()
scores_st_norm= df_scores['scores_normalized_short_term'].values.tolist()
scores_lt = df_scores['scores_raw_long_term'].values.tolist()
                    
# video results
video_scores=pd.read_csv('PicSOM_prediction/visual_pred_training_set_trecvid.csv')
#video_scores=pd.read_csv('textual_scores/text_pred_training_set_trecvid.csv')
video_svr_st = video_scores.short_term_pred.values.tolist()
video_svr_st_norm = video_scores.short_term_norm.values.tolist()
video_svr_lt = video_scores.long_term_pred.values.tolist()
#video_svr_st = video_svr_st_df[1].values.tolist()[:590]
#video_svr_lt = video_svr_lt_df[1].values.tolist()[:590]

       
# text's results
text_scores=pd.read_csv('textual_scores/text_pred_training_set_trecvid.csv')
#text_scores=pd.read_csv('PicSOM_prediction/visual_pred_training_set_trecvid.csv')
text_svr_st = text_scores.short_term_pred.values.tolist()
text_svr_st_norm = text_scores.short_term_norm.values.tolist()
text_svr_lt = text_scores.long_term_pred.values.tolist()

"""
# video_svr's results
video_svr_st_df = pd.read_csv('me_2020/short_i3d+audio_80_750.csv', header=None)
video_svr_lt_df = pd.read_csv('me_2020/long_i3d+audio_260_160.csv', header=None)
video_svr_st = video_svr_st_df[1].values.tolist()[:590]
video_svr_lt = video_svr_lt_df[1].values.tolist()[:590]
"""

combined_df = pd.DataFrame({'gt_st': scores_st,
                            'gt_lt': scores_lt,
                            'gt_st_norm': scores_st_norm,
                            'text_svr_st': text_svr_st,
                            'text_svr_st_norm': text_svr_st_norm,
                            'text_svr_lt': text_svr_lt,

                            'video_svr_st': video_svr_st,
                            'video_svr_st_norm': video_svr_st_norm,
                            'video_svr_lt': video_svr_lt,
                           })

combined_df['video_id'] = df_scores['video_id']

# combined_df.to_csv('me_2020/all_predictions_trainset.csv')
# combined_df = pd.read_csv('me_2020/all_predictions_trainset.csv')


#n = 8
"""
ast =  combined_df['text_st'].round(n)
ist =  combined_df['ismail_st'].round(n)
jst =  combined_df['video_svr_st'].round(n)
alt =  combined_df['text_lt'].round(n)
ilt =  combined_df['ismail_lt'].round(n)
jlt =  combined_df['video_svr_lt'].round(n)
"""
ast_scaler = MinMaxScaler()
ist_scaler = MinMaxScaler()
jst_scaler = MinMaxScaler()
alt_scaler = MinMaxScaler()
ilt_scaler = MinMaxScaler()
jlt_scaler = MinMaxScaler()
ast_norm_scaler = MinMaxScaler()
jst_norm_scaler = MinMaxScaler()

ast = ast_scaler.fit_transform(combined_df['text_svr_st'].values.reshape(-1, 1))
ast_norm = ast_norm_scaler.fit_transform(combined_df['text_svr_st'].values.reshape(-1, 1))
alt = alt_scaler.fit_transform(combined_df['text_svr_lt'].values.reshape(-1, 1))


ist=ast
ist_norm=ast_norm
ilt=alt

jst = jst_scaler.fit_transform(combined_df['video_svr_st'].values.reshape(-1, 1))
jst_norm = jst_norm_scaler.fit_transform(combined_df['video_svr_st'].values.reshape(-1, 1))

jlt = jlt_scaler.fit_transform(combined_df['video_svr_lt'].values.reshape(-1, 1))


score = spearman(combined_df['video_svr_st'], combined_df['gt_st'])
print(score)
## Finetuning the linear combination

# Short Term
increment = 0.01
steps = int(1 / increment)
print("Steps:", steps)

best_combo = None
best_score = 0

for a in tqdm(range(steps+1)):
    for b in range(steps+1 - a):
        c = steps - (a + b)
        al = a / steps
        bl = b / steps
        cl = c / steps
        
        # print(al, bl, cl)
        score = spearman(al * ast + bl * ist + cl * jst, combined_df['gt_st'])
        #score = spearman(al * ast + bl  * jst, combined_df['gt_st'])
        if score > best_score:
            best_score = score
            best_combo = al, bl, cl
            # print('New best combo', best_combo, ':\t', best_score)
            print('Best combo st', best_combo, ':\t', best_score)
            
            
# Short Term Norm
increment = 0.01
steps = int(1 / increment)
print("Steps:", steps)

best_combo = None
best_score = 0

for a in tqdm(range(steps+1)):
    for b in range(steps+1 - a):
        c = steps - (a + b)
        al = a / steps
        bl = b / steps
        #cl = c / steps
        
        # print(al, bl, cl)
        #score = spearman(al * ast + bl * ist + cl * jst, combined_df['gt_st'])
        score = spearman(al * ast_norm + bl  * jst_norm, combined_df['gt_st_norm'])
        if score > best_score:
            best_score = score
            best_combo = al, bl, cl
            # print('New best combo', best_combo, ':\t', best_score)
            print('Best combo st norm', best_combo, ':\t', best_score)

# Long Term
increment = 0.01
steps = int(1 / increment)
print("Steps:", steps)

best_combo = None
best_score = 0

for a in tqdm(range(steps+1)):
    for b in range(steps+1 - a):
        c = steps - (a + b)
        al = a / steps
        bl = b / steps
        cl = c / steps
        
        # print(al, bl, cl)
        score = spearman(al * alt + bl * ilt + cl * jlt, combined_df['gt_lt'])
        #score = spearman(al * alt + bl  * jlt, combined_df['gt_lt'])
        if score > best_score:
            best_score = score
            best_combo = al, bl, cl
            # print('New best combo', best_combo, ':\t', best_score)
            print('Best combo lt', best_combo, ':\t', best_score)

# Short Term for Long Term predictions
            
increment = 0.01
steps = int(1 / increment)
print("Steps:", steps)

best_combo = None
best_score = 0

for a in tqdm(range(steps+1)):
    for b in range(steps+1 - a):
        c = steps - (a + b)
        al = a / steps
        bl = b / steps
        cl = c / steps
        
        # print(al, bl, cl)
        score = spearman(al * ast + bl * ist + cl * jst, combined_df['gt_lt'])
        #score = spearman(al * ast + bl  * jst, combined_df['gt_lt'])
        if score > best_score:
            best_score = score
            best_combo = al, bl, cl
            # print('New best combo', best_combo, ':\t', best_score)
            print('Best combo slt', best_combo, ':\t', best_score)
