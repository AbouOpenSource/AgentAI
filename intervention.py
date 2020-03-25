# coding: utf-8
from sys import argv
nb_neurons = int(argv[-1])
nb_epochs = 4000
#5.963405057290431


import warnings; warnings.simplefilter('ignore')
import pandas as pd
df=pd.read_csv('data/interventionsEtInfos.csv')
del df["nb_accouchement"]
del df['nb_autre']
del df['nb_feu']
del df['nb_noyade']
del df['nb_route']
del df['nb_suicide']



from sklearn.preprocessing import StandardScaler
def transforme(df, num_list = [], cat_list = [], copie = []):
    df_out = pd.DataFrame()
    # On normalise les variables numériques
    scaler = StandardScaler()
    for col in num_list:
        df_out[col] = scaler.fit_transform(df[col].values.reshape(-1, 1)).flatten()
    # On fait un codage disjonctif complet des variables qualitatives
    for col in cat_list:
        pd1 = pd.get_dummies(df[col],prefix=col)
        for col1 in pd1.columns:
            df_out[col1] = pd1[col1]
    # On recopie dans le nouveau dataframe les colonnes contenues dans copie
    for col in copie:
        df_out[col] = df[col].copy()
    return df_out




# Variables numériques
num_attribs = ['annee', 'heure',
               'directionVentBale', 'directionVentDijon', 'directionVentNancy',
               'humiditeBale', 'humiditeDijon', 'humiditeNancy',
               'nebulositeBale',
               'phaseLune',
               'pointRoseeBale', 'pointRoseeDijon', 'pointRoseeNancy',
               'precipitations1hBale', 'precipitations1hDijon', 'precipitations1hNancy',
               'precipitations3hBale', 'precipitations3hDijon', 'precipitations3hNancy',
               'pressionBale', 'pressionDijon', 'pressionNancy',
               'pressionMerBale', 'pressionMerDijon', 'pressionMerNancy',
               'pressionVar3hBale', 'pressionVar3hDijon', 'pressionVar3hNancy',
               'rafalesSur1perBale', 'rafalesSur1perDijon', 'rafalesSur1perNancy',
               'temperatureBale', 'temperatureDijon', 'temperatureNancy',
               'visibiliteBale', 'visibiliteDijon', 'visibiliteNancy',
               'vitesseVentBale', 'vitesseVentDijon', 'vitesseVentNancy',
               'diarrhee_inc', 'diarrhee_inc100', 'diarrhee_inc100_low',
               'diarrhee_inc100_up', 'diarrhee_inc_low', 'diarrhee_inc_up',
               'grippe_inc', 'grippe_inc100', 'grippe_inc100_low',
               'grippe_inc100_up', 'grippe_inc_low', 'grippe_inc_up',
               'varicelle_inc', 'varicelle_inc100', 'varicelle_inc100_low',
               'varicelle_inc100_up', 'varicelle_inc_low', 'varicelle_inc_up',
               'hAllanCourcelle', 'hAllanCourcelleNb', 'hAllanCourcelleStd',
               'hDoubsBesancon', 'hDoubsBesanconNb', 'hDoubsBesanconStd',
               'hDoubsVoujeaucourt', 'hDoubsVoujeaucourtNb', 'hDoubsVoujeaucourtStd',
               'hLoueOrnan', 'hLoueOrnanNb', 'hLoueOrnanStd',
               'hOgnonBonna', 'hOgnonBonnaNb', 'hOgnonBonnaStd',
               'hOgnonMontessau', 'hOgnonMontessauNb','hOgnonMontessauStd']

# Variables qualitatives
cat_attribs = ['bisonFuteDepart', 'bisonFuteRetour',
               'debutFinVacances', 'ferie',
               'jourSemaine',
               'jour', 'jourAnnee', 'mois',
               'luneApparente', 'nuit',
               'tempsPresentBale', 'tempsPresentDijon', 'tempsPresentNancy',
               'tendanceBaromBale', 'tendanceBaromDijon', 'tendanceBaromNancy',
               'vacances', 'veilleFerie']


df_trans = transforme(df, num_attribs, cat_attribs, ['nbInterventions'])
dflt = df_trans.loc[df_trans['annee']<max(df_trans['annee'])]
dfv = df_trans.loc[df_trans['annee']==max(df_trans['annee'])]



from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(dflt, test_size = 0.2, random_state = 42)

X_train = train_set.drop('nbInterventions', axis=1)
y_train = train_set['nbInterventions'].copy()

X_test = test_set.drop('nbInterventions', axis=1)
y_test = test_set['nbInterventions'].copy()

X_val = dfv.drop('nbInterventions', axis=1)
y_val = dfv['nbInterventions'].copy()




from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error
import numpy as np
from time import time



n_inputs = X_train.shape[1]




learning_rate = 0.02
activ = 'selu'
init = 'he_normal'
momentum = 0.9
sgd = SGD(lr=learning_rate,
          momentum=momentum,
          #decay=decay_rate,
          nesterov=False)
model = Sequential()
model.add(Dense(nb_neurons,
                input_dim=n_inputs,
                activation=activ,
                kernel_initializer=init,
                #name='dense0_'+str(arguments[0])+'_'+str(arguments[1])
               ))
model.add(Dropout(0.2))
'''model.add(Dense(arguments[1], 
                activation=activ, 
                kernel_initializer=init,
                #name='dense1_'+str(arguments[0])+'_'+str(arguments[1])+'_'+str(arguments[2])
               ))
model.add(Dropout(0.2))
'''
model.add(Dense(1, activation=activ, kernel_initializer=init))
model.compile(optimizer=sgd, loss='mse')
history = model.fit(X_train, y_train, batch_size=50, epochs=nb_epochs, verbose = 1)

y_pred = model.predict(X_test)
print(mean_squared_error(y_test,y_pred))

fic = open('errors/Layer_1_Neurons_'+str(nb_neurons)+'.txt','w')
fic.write(str(np.sqrt(mean_squared_error(y_test,y_pred))))
fic.close()
