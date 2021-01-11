
from sklearn.svm import SVR
import tensorflow_ranking as tfr
from keras import optimizers
from tensorflow.python.keras.layers import Dropout
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

from sklearn import linear_model

from numpy.random import seed

SEED = 1  # use this constant seed everywhere

seed(SEED)  # numpy pseudo-random generator
tf.set_random_seed(SEED)
#seed(1)



def Get_score(Y_pred,Y_true):
    '''Calculate the Spearmann"s correlation coefficient'''
    Y_pred = np.squeeze(Y_pred)
    Y_true = np.squeeze(Y_true)
    if Y_pred.shape != Y_true.shape:
        print('Input shapes don\'t match!')
    else:
        if len(Y_pred.shape) == 1:
            Res = pd.DataFrame({'Y_true':Y_true,'Y_pred':Y_pred})
            your_spearman = Res[['Y_true','Y_pred']].corr(method='spearman',min_periods=1)
            print('The Spearman\'s correlation coefficient is: %.3f' % your_spearman.iloc[1][0],3)
            result=your_spearman.iloc[1][0]
            #results.append(result)
        else:
            for ii in range(Y_pred.shape[1]):
                Get_score(Y_pred[:,ii],Y_true[:,ii])

def Get_score_ind(pred,Y_true):
    Y_pred = pd.DataFrame(data=[pred])
    Y_pred=Y_pred.T
    Y_pred = np.squeeze(Y_pred.values)
    Y_true=np.squeeze(Y_true)
    short_term = pd.DataFrame({'Y_pred': Y_pred, 'Y_test': Y_true})
    your_spearman = short_term[['Y_pred', 'Y_test']].corr(method='spearman', min_periods=1)
    print('The Spearman\'s correlation coefficient is: %.3f' % your_spearman.iloc[1][0], 3)
    return your_spearman.iloc[1][0]


def do_pca(sentence_features, n_dimensions):
        # Make everything into one big matrix (words x features_per_word)
        sentence_lengths = [sf.shape[0] for sf in sentence_features]
        X = np.concatenate(sentence_features, axis=0)

        # Transform with PCA
        pca = PCA(n_components=n_dimensions)
        X_transformed = pca.fit_transform(X)

        # Put everything back into shape
        reduced_dim_sentence_features = np.split(X_transformed, np.cumsum(sentence_lengths)[:-1])
        return reduced_dim_sentence_features




def three_dense_layers_model_1_outuput(X_train, X_test, Y_train, Y_test):

    print('X_train', X_train.shape)
    print('X_test', X_test.shape)
    print('Y_train', Y_train.shape)
    print('Y_test', Y_test.shape)
    n_cols = X_train.shape[1]
    # Save the number of columns in predictors: n_cols
    # Set up the model: modele
    model = Sequential()
    # Add the first layer
    model.add(Dense(100, activation='relu', input_shape=(n_cols,)))
    #model.add(Dropout(0.2))

    # Add the second layer
    model.add(Dense(50, activation='relu'))

    # Add the output layer
    model.add(Dense(1))
    # Compile the model
    #adam = optimizers.Adam(lr=0.001)
    #model.compile(optimizer='Adam', loss='mean_absolute_percentage_error')
    model.compile(optimizer='Adam', loss='mean_squared_error')

    # Define early_stopping_monitor
    early_stopping_monitor = EarlyStopping(patience=10)


    # Fit the model

    model.fit(X_train, Y_train, validation_split=0.2, epochs=200, callbacks=[early_stopping_monitor])


    # Verify that model contains information from compiling
    print("Loss function: " + model.loss)
    predictions = model.predict(X_test)

    return predictions


def three_dense_layers_model_2_outuput(X_train, X_test, Y_train, Y_test):

    print('X_train', X_train.shape)
    print('X_test', X_test.shape)
    print('Y_train', Y_train.shape)
    print('Y_test', Y_test.shape)
    n_cols = X_train.shape[1]
    # Save the number of columns in predictors: n_cols
    # Set up the model: modele
    model = Sequential()
    # Add the first layer
    model.add(Dense(100, activation='relu', input_shape=(n_cols,)))
    # Add the second layer
    model.add(Dense(50, activation='relu'))
    # Add the output layer
    model.add(Dense(2))

    # Compile the model
    #adam = optimizers.Adam(lr=0.001)
    model.compile(optimizer='adam', loss='mean_squared_error')
    #model.compile(optimzer=AdamOptimizer(0.001), loss='mean_squarred_error')

    # Define early_stopping_monitor
    early_stopping_monitor = EarlyStopping(patience=20)

    # Fit the model

    model.fit(X_train, Y_train, validation_split=0.2, epochs=40, callbacks=[early_stopping_monitor])


    # Verify that model contains information from compiling
    print("Loss function: " + model.loss)
    predictions = model.predict(X_test)

    return predictions



def SVR_imp(X_train, X_test, Y_train, Y_test):
    model = SVR()
    model.fit(X_train,Y_train,)
    predictions = model.predict(X_test)
    return predictions


def Lasso_imp(X_train, X_test, Y_train, Y_test):
    model = linear_model.Lasso(alpha=0.1)
    model.fit(X_train,Y_train,)
    predictions = model.predict(X_test)
    return predictions
