#Nous utilisons ici le modèle ARIMA de base en séries temporelles pour prédire l'évolution de US Inflation Rate en 2024.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pylab import style
import seaborn as sns
import datetime


style.use('ggplot')    #Utiliser le style 'ggplot'
plt.rcParams['font.sans-serif'] = ['SimHei'] # Définir le style de police
plt.rcParams['axes.unicode_minus'] = False  # Utiliser le Unicode minus:-1


# read data
excel_file_path = 'C:/Users/XueXi/Desktop/Macro Projet/Inflation Rate.xlsx'
df_data = pd.read_excel(excel_file_path)

#Étape 1. Obtenir les données de séries temporelles
# On prend les données  du 1er janvier 2020 au June 2023
#L'ensemble d'entraînement est constitué de données de 2020 à 2022

df_data['Date'] = pd.to_datetime(df_data['Date'], format='%Y-%m-%d')
df_data.set_index('Date', inplace=True)
df_train_data = df_data.loc['2000-01-31':'2023-11-30']
#Visualization
df_train_data.plot()
plt.title='Train Data'
plt.show()

# Étape 2. Vérifiez si la série est stationnaire
# Utiliser la méthode de First Difference, c'est-à-dire la ligne suivante moins la ligne précédente, afin de la convertir en une série stationnaire
df_diff_data = df_train_data.diff()
df_diff_data.dropna(inplace=True)

#Calculer p-value et vérifer la stationnarité des données 
from statsmodels.tsa.stattools import adfuller
def Teststationarity(df_diff_data):
    dfinput = adfuller(df_diff_data)
    dfoutput = pd.Series(dfinput[0:4], index=['Test Statistic','p-value','Lags Used','Number of Observations Used'])
    for key,value in dfinput[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    return dfoutput
ts_us=df_diff_data.iloc[:, 0]
ts_eu=df_diff_data.iloc[:, 1]

Teststationarity(ts_us)
Teststationarity(ts_eu)

print(Teststationarity(ts_us))
print(Teststationarity(ts_eu))

print("La p-value est bien inférieure à la valeur critique，nous pouvons donc dire que notre série est stationnaire.")

#Garphique de First Difference
df_diff_data.plot()
plt.title='First Difference'
plt.show()

# Étape 3.Trouver les trois paramètres du modèle ARIMA: p, d, q en observant le graphique de l'ACF et la PACF
# ACF est une fonction d'autocorrélation complète qui décrit le degré de corrélation entre la valeur actuelle
# de la séquence et sa valeur passée. 
# PACF est une fonction d'autocorrélation partielle, 
# il s'agit de trouver la corrélation entre le résidu et la valeur de retard suivante

# Tracer l'ACF et la PACF
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
acf = plot_acf(ts_eu,lags=25)
plt.title='ACF'
acf.show()
pacf = plot_pacf(ts_eu,lags=25)
plt.title='PACF'
pacf.show()
plt.show()

# Étape 4.Etablir le modèles ARIMA
# p est la valeur décalée, également appelé auto-régressif
# d représente le nombre de fois que la série doivent être différenciées pour être stables, ce que l'on appelle également le terme intégré
# q représente la valeur décalée de l'erreur de prédiction utilisée dans le modèle de prédiction, également appelé la moyenne mobile

from statsmodels.tsa.arima.model import ARIMA
model = ARIMA(df_train_data['EU'], order=(2,1,0),freq='M')
result = model.fit() 
print(result.summary())



#Étape 5.Prédiction
#Prédire l'évolution future
pred = result.predict('2023-11-30', '2024-5-31',dynamic=True, typ='levels')
print (pred)

#Faire un graphique pour voir la différence entre la valeur prédite et la valeur réelle
df_comparison = pd.concat([df_data,pred,],axis=1,keys=['ori','pre'])
df_comparison.plot()
plt.title='Prediction'
plt.show()

