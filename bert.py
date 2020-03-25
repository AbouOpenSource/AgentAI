
# 0.9952390599675851
# [[19050    15]
# [   79   600]]



# @title Constants

SEQ_LEN = 128
BATCH_SIZE = 128
EPOCHS = 100
LR = 4e-5

# @title Environment
import os
import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn.metrics import confusion_matrix, precision_recall_curve

from numpy.random import seed
seed(123)
from tensorflow import set_random_seed
set_random_seed(123)
lookbac k =20


# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequences[i:end_ix, 1:], sequences[end_i x -1, 0]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


# In[3]:


train_file s =['aircraft_28_3c09ee_attacked.csv' ,'aircraft_31_3c0fe6_attacked.csv' ,'aircraft_33_3c4907_attacked.csv',
             #     'aircraft_34_3c496a_attacked.csv',  => cause NAN
             'aircraft_35_3c49eb_attacked.csv' ,'aircraft_36_3c56e6_attacked.csv',
             'aircraft_37_3c5ee1_attacked.csv' ,'aircraft_38_3c6441_attacked.csv',
             'aircraft_45_3c6624_attacked.csv' ,'aircraft_48_3c6676_attacked.csv' ,'aircraft_49_3c66a5_attacked.csv']

firs t =True
for file in train_files:
    if first:
        df = pd.read_csv(file)
        # suppress time
        d f =df.iloc[: ,1:]
        print(df.shape)
        firs t =False
    else:
        df2 = pd.read_csv(file)
        df 2 =df2.iloc[: ,1:]
        d f =pd.concat([df ,df2] ,axis=0)
print(df.shape)

# scaler on all train data
# scaler = preprocessing.MinMaxScaler().fit(df)

firs t =True
for file in train_files:
    if first:
        df = pd.read_csv(file)
        # suppress time
        d f =df.iloc[: ,1:]
        trai n =df.iloc[: ,:].to_numpy()
        #        train=scaler.transform(train)
        X_train ,y_trai n =split_sequences(train ,lookback)
        y_trai n =y_train.reshape(y_train.shape[0] ,1)
        firs t =False
    else:
        df = pd.read_csv(file)
        # suppress time
        d f =df.iloc[: ,1:]
        trai n =df.iloc[: ,:].to_numpy()
        #        train=scaler.transform(train)

        X_train2 ,y_train 2 =split_sequences(train ,lookback)
        #        X_train2=X_train2.reshape(X_train2.shape[0],X_train2.shape[1]*X_train2.shape[2])
        y_train 2 =y_train2.reshape(y_train2.shape[0] ,1)

        print(X_train.shape)
        print(X_train2.shape)


        print(y_train.shape)
        print(y_train2.shape)


        X_trai n =np.vstack((X_train ,X_train2))
        y_trai n =np.vstack((y_train ,y_train2))

print(X_train.shape)
print(y_train.shape)


test_file s =['aircraft_0_00a2e4_attacked.csv' ,'aircraft_20_394c12_attacked.csv',
            'aircraft_24_39bd27_attacked.csv' ,'aircraft_23_3991e2_attacked.csv']
firs t =True
for file in test_files:
    if first:
        df = pd.read_csv(file)
        # suppress time
        d f =df.iloc[: ,1:]
        tes t =df.iloc[: ,:].to_numpy()
        #        test=scaler.transform(test)

        X_test ,y_tes t =split_sequences(test ,lookback)
        y_tes t =y_test.reshape(y_test.shape[0] ,1)
        firs t =False
    else:
        df = pd.read_csv(file)
        # suppress time
        d f =df.iloc[: ,1:]
        test 2 =df.iloc[: ,:].to_numpy()
        #        test2=scaler.transform(test2)

        X_test2 ,y_test 2 =split_sequences(test2 ,lookback)
        y_test 2 =y_test2.reshape(y_test2.shape[0] ,1)

        X_tes t =np.vstack((X_test ,X_test2))
        y_tes t =np.vstack((y_test ,y_test2))


X_trai n =X_train.reshape(X_train.shape[0] ,X_train.shape[1 ] *X_train.shape[2])
X_tes t =X_test.reshape(X_test.shape[0] ,X_test.shape[1 ] *X_test.shape[2])

print("laa")
print(X_train.shape)
print(np.zeros_like(X_train).shape)



X_trai n =[X_train ,np.zeros_like(X_train)]
print(X_test.shape)
# print(X_train.shape)
X_tes t =X_test[:19744]
y_tes t =y_test[:19744]
print(X_test.shape)


X_tes t =[X_test ,np.zeros_like(X_test)]

# print(X_train.shape)
# print(y_train.shape)


# In[6]:







pretrained_path = 'uncased_L-12_H-768_A-12'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

# TF_KERAS must be added to environment variables in order to use TPU
os.environ['TF_KERAS'] = '1'

# @title Initialize TPU Strategy

import tensorflow as tf
from keras_bert import get_custom_objects

strategy = tf.distribute.MirroredStrategy()

# @title Load Basic Model
import codecs
from keras_bert import load_trained_model_from_checkpoint, get_model, compile_model

token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

with strategy.scope():

    model = get_model(
        token_num=200000,
        #        head_num=25,
        #        transformer_num=20,
        embed_dim=4 8 *2,
        feed_forward_dim=512,
        seq_len=100,
        pos_num=100,  # 128
        training=True,
        trainable=None,
        dropout_rate=0.1,
    )
    compile_model(model)

#    model = load_trained_model_from_checkpoint(
#        config_path,
#        checkpoint_path,
#        training=True,
#        trainable=True,
#        seq_len=SEQ_LEN,
#    )


print(model.summary())
# @title Download IMDB Data
import tensorflow as tf

dataset = tf.keras.utils.get_file(
    fname="aclImdb.tar.gz",
    origin="http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
    extract=True,
)

# @title Convert Data to Array
import os
import numpy as np
from tqdm import tqdm
from keras_bert import Tokenizer

tokenizer = Tokenizer(token_dict)




# @title Build Custom Model
from tensorflow.python import keras
from keras_radam import RAdam

# from keras.optimizers import Adam, SGD, RMSprop, Adadelta, Adagrad, Nadam
# from keras import optimizers

with strategy.scope():
    inputs = model.inputs[:2]
    dense = model.get_layer('NSP-Dense').output
    outputs = keras.layers.Dense(units=2, activation='softmax')(dense)

    model = keras.models.Model(inputs, outputs)
    model.compile(
        # RAdam(lr=LR),
        #        RAdam(total_steps=10000, warmup_proportion=0.1, min_lr=1e-5, lr=1e-4),
        keras.optimizers.SGD(lr=LR),
        loss='sparse_categorical_crossentropy',
        metrics=['sparse_categorical_accuracy'],
    )

# @title Initialize Variables
import tensorflow as tf
import tensorflow.keras.backend as K

sess = K.get_session()
uninitialized_variables = set([i.decode('ascii') for i in sess.run(tf.report_uninitialized_variables())])
init_op = tf.variables_initializer(
    [v for v in tf.global_variables() if v.name.split(':')[0] in uninitialized_variables]
)
sess.run(init_op)

# @title Fit

model.fit(
    X_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
)

# @title Predict

predicts = model.predict(X_test, verbose=True).argmax(axis=-1)

# @title Accuracy

y_tes t =np.transpose(y_test)
y_tes t =np.squeeze(y_test)
print(y_test)
print(predicts)
print(y_test.shape)
print(np.sum(y_test == predicts))
print(y_test.shape[0])
print(np.sum(y_test == predicts) / y_test.shape[0])



conf_matrix = confusion_matrix(y_test ,predicts)

print(conf_matrix)
