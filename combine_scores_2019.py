
from language_vectors import *
from models import *
from sklearn  import preprocessing



if __name__ == '__main__':

    #if ran from windows machine
    #titles_path = '/medias/db3/MM_GrandChallenge/MediaMemorability/dev-set/dev-set_video-captions.txt'
    # These paths are hard-coded , TODO replace by arguments
    alto_captions_path='alto_captions.txt'
    titles_path = '/home/portege24/Documents/devset/dev-set/dev-set_video-captions.txt'
    titles= read_titles(titles_path)
    alto_captions=read_alto_captions(alto_captions_path)
    alto_captions_train= alto_captions[0:8000]
    alto_and_titles=alto_and_titles(alto_captions_train, titles)
    X_resnet=pd.read_csv('X_resnet.csv')

    captions = pd.read_json('captions_list.json')
    danny_captions = pd.DataFrame(data=captions)
    danny_captions['captions_danny'] = captions
    video_name = pd.read_json('names_list.json')
    danny_captions['video'] = video_name
    danny_captions['video']=danny_captions['video'].str.strip("video")
    danny_embeddings = np.load('embeddings_list.npy')

    alto_titles_danny = adding_danny_to_rest(alto_and_titles,danny_captions)
    alto_titles_danny.to_csv('alto_titles_danny.csv')


    ismail1 = pd.read_csv('results_count_log_with_deepcaptions.csv')
    #ismail2 = pd.read_csv('results_bertmaxpavgpca350.csv')
    #ismail3 = pd.read_csv('results_rcnn_with_deepcaptions.csv')
    #ismail4 = pd.read_csv('results_selfattention_with_deepcaptions.csv')
    ismail_st=ismail1['short_term_predictions']
    ismail_lt = ismail1['long_term_predictions']

    #ismail_st=ismail_st.astype(float)
    #ismail_st = ismail_st.astype(float)







    labels=pd.read_csv('/home/portege24/Documents/devset/dev-set/ground-truth/ground-truth_dev-set.csv')
    labels[['video', 'poubelle']] = labels['video'].str.split(".webm", expand=True, )
    labels[['poubelle2', 'video']] = labels['video'].str.split("video", expand=True, )

    #df = pd.merge(labels,alto_titles_danny, on="video")
    df=pd.merge(labels,alto_titles_danny,on='video')

    #df=pd.concat([alto_titles_danny,labels], axis=1, join='inner')
    #df.to_csv('zuco-nlp/sentiment-analysis/SST_data/titles_and_alto_captions.csv')
    #df.to_csv('zuco-nlp/sentiment-analysis/SST_data/alto_captions.csv')

    df_tf=obtaining_tfidf_vectors(df)
    df_bow=obtaining_BOW_vectors(df)

    #df_bow['video']=df['video']


    #to compute Bert client vectors, if already calculated can just charge file
    #Bert_client_vectors= getting_Bertclient_vectors(alto_captions_train["caption"])


    #to charge Bert client vectors
    #Bert_client_vectors= pd.read_csv("bert_alto.csv")

    Y=labels[['short-term_memorability','long-term_memorability']].values #targets
    Y_st=labels['short-term_memorability']
    #print(Y_st)
    Y_lt = labels['long-term_memorability']
    df2=pd.read_csv('et_max_alto_titles.csv')
    #df2['video'] = labels['video']
    df3=pd.read_csv('et_max_titles.csv')

    #df3['video'] = labels['video']
    df4=pd.read_csv('eeg_max_titles_pca.csv')
    #df4['video'] = labels['video']
    df5=pd.read_csv('eeg_max_alto_titles_pca.csv')
    df6=pd.read_csv('et_max_titles_danny_alto.csv')
    df7 = pd.read_csv('eeg_max_titles_pca_danny_alto.csv')

    danny_embeddings = np.load('embeddings_list.npy')
    print(danny_embeddings.shape)

    mean_embeddings=[]
    last_column_embedding = []
    for x in danny_embeddings:
        mean_embeddings.append(np.mean(x, axis=0))
        last_column_embedding.append(x[-1])

    df_embeddings = pd.DataFrame(data=mean_embeddings)
    df_embeddings['video'] =danny_captions['video']
    df_emb = pd.merge(labels,df_embeddings, on='video')
    #print(df_emb.columns)
    df_subset=pd.DataFrame()
    for i in range(1024,2047):
        df_subset['{}'.format(i)]=df_emb[i]
    #hmp=pd.read_csv('features.csv')



    #print(df_subset)

    #Y_st=df_emb['short-term_memorability']
    #Y_lt = df_emb['long-term_memorability']
    #print(Y_st)


    df_vec_st=np.concatenate([df_subset,df7,df6], axis=1)
    #print(len(df_vec_st))
    df_vec_lt=np.concatenate([df_subset,df6,df7], axis=1)
    #df_vec_st=hmp
    #df_vec_lt=hmp

    #print(len(df_vec_lt))

    X_st = preprocessing.normalize(df_vec_st)
    X=preprocessing.normalize(df_vec_st)
    X_lt = preprocessing.normalize(df_vec_lt)
    #Y=preprocessing.normalize(Y)

    #X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
                                                        #random_state=42)  # random state for reproducability

    #The predictions coming from resnet need to be checked = reran to be sure there has not been a confusion in the names
    folds=[1,2,3,4]
    retrain=True
    two_output=True
    scores_st_combi=[]
    scores_lt_combi=[]
    scores_st_text=[]
    scores_lt_text=[]
    scores_st_resnet=[]
    scores_lt_resnet=[]

    df_pred_text = pd.DataFrame(data=np.arange(2000), columns=['trash'])


    for fold in folds:
        if fold==1:
            X_train = X[0:6000]
            X_train_st=X_st[0:6000]
            X_train_lt = X_lt[0:6000]
            X_test_st=X_st[6000:8000]
            X_test_lt = X_lt[6000:8000]
            X_test = X[6000:8000]
            Y_train_st = Y_st[0:6000]
            Y_train_lt= Y_lt[0:6000]
            Y_train = Y[0:6000]
            Y_test_st=Y_st[6000:8000]
            Y_test_lt = Y_lt[6000:8000]
            Y_test=Y[6000:8000]
            ismail_st_test = ismail_st[6000:8000]
            ismail_lt_test = ismail_lt[6000:8000]
            #res_st = pd.read_csv('po_maxlt.csv')
            res_st = pd.read_csv('po_max_st_1test.csv')
            #res_st['ismail']=ismail_st_test.values
            res_st = res_st['0']
            res_lt = pd.read_csv('p1_max_lt_1test.csv')
            res_lt = res_lt['0']
            column_st_name='st-1'
            column_lt_name='lt-1'



        elif fold==2:
            # to test on first 2000
            X_train_st=X_st[2000:]
            X_train = X[2000:]
            X_train_lt=X_lt[2000:]
            X_test_st = X_st[0:2000]
            X_test_lt=X_lt[0:2000]
            X_test = X[0:2000]
            Y_train_st=Y_st[2000:]
            Y_train_lt=Y_lt[2000:]
            Y_train = Y[2000:]
            Y_test_st = Y_st[0:2000]
            Y_test_lt=Y_lt[0:2000]
            Y_test = Y[0:2000]
            ismail_st_test = ismail_st[0:2000]
            ismail_lt_test = ismail_lt[0:2000]
            res_st = pd.read_csv('po_max_st_2test.csv')
            res_st = res_st['0']
            res_lt = pd.read_csv('p1_max_lt_2test.csv')
            res_lt = res_lt['0']
            column_st_name='st-2'
            column_lt_name='lt-2'



        elif fold==3:
            # to test on 2nd 2000
            X_train_st = np.concatenate((X_st[:2000], X_st[4000:]))
            X_train = np.concatenate((X[:2000], X[4000:]))
            X_train_lt = np.concatenate((X_lt[:2000], X_lt[4000:]))
            Y_train_st = np.concatenate((Y_st[:2000], Y_st[4000:]))
            Y_train_lt = np.concatenate((Y_lt[:2000], Y_lt[4000:]))
            Y_train = np.concatenate((Y[:2000], Y[4000:]))
            X_test_st = X_st[2000:4000]
            X_test = X[2000:4000]
            X_test_lt = X_lt[2000:4000]
            Y_test_st = Y_st[2000:4000]
            Y_test_lt = Y_lt[2000:4000]
            Y_test=Y[2000:4000]
            ismail_st_test = ismail_st[2000:4000]
            ismail_lt_test = ismail_lt[2000:4000]
            print(len(Y_test))
            print(len(Y_train_lt))
            res_st = pd.read_csv('po_max_st_3test.csv')
            res_st = res_st['0']
            res_lt = pd.read_csv('p1_max_lt_3test.csv')
            res_lt = res_lt['0']
            column_st_name='st-3'
            column_lt_name='lt-3'


        elif fold==4:
            # np.concatenate((data_x[:4000, :], data_x[6000:, :]))
            X_train_st = np.concatenate((X_st[:4000], X_st[6000:]))
            X_train = np.concatenate((X[:4000], X[6000:]))
            X_train_lt = np.concatenate((X_lt[:4000], X_lt[6000:]))
            Y_train_st = np.concatenate((Y_st[:4000], Y_st[6000:]))
            Y_train = np.concatenate((Y[:4000], Y[6000:]))
            Y_train_lt = np.concatenate((Y_lt[:4000], Y_lt[6000:]))
            X_test_st = X_st[4000:6000]
            X_test_lt = X_lt[4000:6000]
            Y_test_st = Y_st[4000:6000]
            Y_test_lt = Y_lt[4000:6000]
            Y_test = Y[4000:6000]
            X_test = X[4000:6000]
            ismail_st_test=ismail_st[4000:6000]
            ismail_lt_test = ismail_st[4000:6000]

            print(len(Y_test))
            print(len(Y_train_lt))
            res_st = pd.read_csv('po_max_st_4test.csv')
            res_st = res_st['0']

            res_lt = pd.read_csv('p1_max_lt_4test.csv')
            res_lt = res_lt['0']
            column_st_name='st-4'
            column_lt_name='lt-4'


        if retrain==True:
            if two_output==True:
                predictions=three_dense_layers_model_2_outuput(X_train, X_test, Y_train, Y_test)
                predictions_st=predictions[:,0]
                predictions_lt = predictions[:, 0]
            else:
                predictions_st=three_dense_layers_model_1_outuput(X_train_st, X_test_st, Y_train_st, Y_test_st)
                predictions_lt=three_dense_layers_model_1_outuput(X_train_lt, X_test_lt, Y_train_lt, Y_test_lt)
            #predictions_lt=three_dense_layers_model_1_outuput(X_train_lt, X_test_lt, Y_train_lt, Y_test_lt)
        else:
            df_pred_text=pd.read_csv('danny_embedding.csv')
            predictions_st=df_pred_text[column_st_name]
            predictions_lt = df_pred_text[column_lt_name]

        #score_st_text = Get_score_ind(predictions_st,Y_test_st)
        #score_lt_text = Get_score_ind(predictions_lt, Y_test_st)
        #score_lt = Get_score_ind(res_text_lt, Y_test_lt)
        #predictions_st

        #res_st.type()
        #ismail_st_test.type()
        res_text_st = (0.4 * res_st) + (0.3 * np.squeeze(predictions_st)) + (0.3* ismail_st_test.values)

        #res_text_st = (0.4 * ismail_st_test.values) + (0.1 * np.squeeze(predictions_st)) + (0.6 * res_st)
        res_text_lt = 0.4* res_lt + 0.3* np.squeeze(predictions_st)+(0.3 * ismail_st_test.values)
        score_st_combi = Get_score_ind(res_text_st, Y_test_st)
        score_lt_combi = Get_score_ind(res_text_lt, Y_test_lt)
        scores_st_combi.append(score_st_combi)
        scores_lt_combi.append(score_lt_combi)
        #scores_st_text.append(score_st_text)
        df_pred_text[column_st_name] = predictions_st
        df_pred_text[column_lt_name] = predictions_lt


    average_st=np.mean(scores_st_combi)
    print(average_st)
    average_lt=np.mean(scores_lt_combi)
    print(average_lt)

    if retrain==True:
        print('done')
        #df_pred_text.to_csv('pred_text_1output_MSE.csv')
        #df_pred_text.to_csv('pred_text_1output_MSE_15pca.csv')
        #df_pred_text.to_csv('pred_text_1output_MSE_10_pca_abs.csv')
        #df_pred_text.to_csv('pred_text_1output_MSE_10_pca_abs.csv')
        #df_pred_text.to_csv('pred_text_2output_noeeg.csv')
        #df_pred_text.to_csv('pred_text_2output_nodanny.csv')
        #df_pred_text.to_csv('pred_text_2output.csv')
        #df_pred_text.to_csv('pred_2output_absolute_error.csv')
        #df_pred_text.to_csv('only_titles_bow')
        #df_pred_text.to_csv('tf_ngrams_2output_allcap.csv')
        #df_pred_text.to_csv('2output_eegall_et_danny_alto.csv')
        #df_pred_text.to_csv('1output_eegall_et_danny_alto.csv')
        #df_pred_text.to_csv('1output_eegall_et_danny_alto_200,100.csv')
        #df_pred_text.to_csv('1output_eegall_et_danny_alto_100,50.csv')
        #df_pred_text.to_csv('bow_et_2output.csv')
        #df_pred_text.to_csv('pred_text_1output_noeeg.csv')
        #df_pred_text.to_csv('pred_text_1output_noet.csv')
        #df_pred_text.to_csv('Lasso_best_ST.csv')
        #df_pred_text.to_csv('danny_embedding.csv')
        #df_pred_text.to_csv('danny_embedding_last_frame.csv')
        #df_pred_text.to_csv('hmp_2output.csv')




    """scores_st_combi=[]
    scores_lt_combi=[]
    scores_st_text=[]
    scores_lt_text=[]
    scores_st_resnet=[]
    scores_lt_resnet=[]"""



