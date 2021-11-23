import pickle
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler, MinMaxScaler

spearman = lambda x,y: spearmanr(x, y).correlation

df_scores = pd.read_csv('me_2020/gt_scores.csv')

# Ground Truth
scores_st = df_scores['part_1_scores'].values.tolist()
scores_lt = df_scores['part_2_scores'].values.tolist()


# Ismail's results
ismail_df = pd.read_csv('me_2020/ismail_best_st.csv', header=None)
ismail_lt_df = pd.read_csv('me_2020/ismail_best_lt.csv', header=None)
ismail_st = ismail_st_df[1].values.tolist()[:590]
ismail_lt = ismail_lt_df[1].values.tolist()[:590]

       
# Alison's results
alison_st_pkl = pickle.load(open('me_2020/6folds_st.pkl', 'rb'))
alison_lt_pkl = pickle.load(open('me_2020/6folds_lt.pkl', 'rb'))
alison_st = [v for fold in alison_st_pkl for v in fold[0]]
alison_lt = [v for fold in alison_lt_pkl for v in fold[0]]


# Jorma's results
jorma_st_df = pd.read_csv('me_2020/short_i3d+audio_80_750.csv', header=None)
jorma_lt_df = pd.read_csv('me_2020/long_i3d+audio_260_160.csv', header=None)
jorma_st = jorma_st_df[1].values.tolist()[:590]
jorma_lt = jorma_lt_df[1].values.tolist()[:590]

combined_df = pd.DataFrame({'gt_st': scores_st,
                            'gt_lt': scores_lt,
                            'ismail_st': ismail_st,
                            'ismail_lt': ismail_lt,
                            'alison_st': alison_st,
                            'alison_lt': alison_lt,
                            'jorma_st': jorma_st,
                            'jorma_lt': jorma_lt,
                           })

combined_df['video_id'] = df_scores['video_id']

# combined_df.to_csv('me_2020/all_predictions_trainset.csv')
# combined_df = pd.read_csv('me_2020/all_predictions_trainset.csv')


n = 8
ast =  combined_df['alison_st'].round(n)
ist =  combined_df['ismail_st'].round(n)
jst =  combined_df['jorma_st'].round(n)
alt =  combined_df['alison_lt'].round(n)
ilt =  combined_df['ismail_lt'].round(n)
jlt =  combined_df['jorma_lt'].round(n)

ast_scaler = MinMaxScaler()
ist_scaler = MinMaxScaler()
jst_scaler = MinMaxScaler()
alt_scaler = MinMaxScaler()
ilt_scaler = MinMaxScaler()
jlt_scaler = MinMaxScaler()

ast = ast_scaler.fit_transform(combined_df['alison_st'].values.reshape(-1, 1))
ist = ist_scaler.fit_transform(combined_df['ismail_st'].values.reshape(-1, 1))
jst = jst_scaler.fit_transform(combined_df['jorma_st'].values.reshape(-1, 1))
alt = alt_scaler.fit_transform(combined_df['alison_lt'].values.reshape(-1, 1))
ilt = ilt_scaler.fit_transform(combined_df['ismail_lt'].values.reshape(-1, 1))
jlt = jlt_scaler.fit_transform(combined_df['jorma_lt'].values.reshape(-1, 1))


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
        if score > best_score:
            best_score = score
            best_combo = al, bl, cl
            # print('New best combo', best_combo, ':\t', best_score)
            print('Best combo', best_combo, ':\t', best_score)

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
        if score > best_score:
            best_score = score
            best_combo = al, bl, cl
            # print('New best combo', best_combo, ':\t', best_score)
            print('Best combo', best_combo, ':\t', best_score)

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
        if score > best_score:
            best_score = score
            best_combo = al, bl, cl
            # print('New best combo', best_combo, ':\t', best_score)
            print('Best combo', best_combo, ':\t', best_score)
