# Beta Bank
Project Description: Beta Bank's customers are leaving, every month, little by little. Bankers have discovered that it is cheaper to save existing customers than to attract new ones.

Target: Predict whether a customer will leave the bank soon. 
----------

Descripción del proyecto: 
Los clientes de Beta Bank se están yendo, cada mes, poco a poco. Los banqueros descubrieron que es más barato salvar a los clientes existentes que atraer nuevos.

Objetivo:
Necesitamos predecir si un cliente dejará el banco pronto.
# TOC:
* [Previsualización de la información](#Initialization)
* [Preparación de los datos](#DataPreprocessing)
* [Analysis](#Analysis)
* [Modelos: Árbol de decision, Bosque Aleatorio, Regresión Logística](#Models)
* [Fixing Imbalance Models](#FixingImbalance)
* [Model Conclusions](#Conclusions)

# Plan de trabajo:
1. Initialization
2. Data preprocessing
3. Analysis
4. Models
5. Fixing Imbalance
6. Conclusions

# Initialization
import pandas as pd
import numpy as np
import math 
import seaborn as sns
from matplotlib import pyplot as plt
from scipy import stats as st
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,roc_auc_score

# DataPreprocessing
2.1 Data Info
df_betabank=pd.read_csv('/datasets/Churn.csv')

print("Data Frame Beta Bank \n")
print(df_betabank.info())
print()
print("Data Head Beta Bank \n")
print(df_betabank.head())

2.2 Duplicated
print('Duplicated Data in the Data Frame:', df_betabank.duplicated().sum())
Additional remarks:

Duplicate values were not found
2.3 Null Data
print('Null Data in the Data Frame\n\n',df_betabank.isnull().sum())
    
print("Tenure Average:",df_betabank['Tenure'].mean())
print("Tenure Median:",df_betabank['Tenure'].median())
df_betabank['Tenure']=df_betabank['Tenure'].fillna(df_betabank['Tenure'].median())
print('Null Data in the Data Frame:\n\n',df_betabank.isnull().sum())
Additional remarks:

Data completed
# Analysis 
3.1 Credit Score vs Geography
plt.figure(figsize=(8, 6))
sns.boxplot(x='CreditScore', y='Geography', data=df_betabank)
plt.title('Boxplot Credit Score vs Geography')
plt.xlabel('Credit Score')
plt.ylabel('Geography')
plt.show()

#Whitout outliers

df_betabank_cs=df_betabank['CreditScore'].describe()
print(df_betabank_cs)
#Quartile
Q1=584
Q3=718

IQR=Q3-Q1
print('IQR: ',IQR)
print()
upper_limit=Q3+(1.5*IQR)
lower_limit=Q1-(1.5*IQR)
print('Upper Limit',upper_limit)
print('Lower Limit',lower_limit)

df_no_csvsg=df_betabank.query('CreditScore <= @upper_limit')
df_no_csvsg=df_betabank.query('CreditScore >= @lower_limit')

plt.figure(figsize=(8, 6))
sns.boxplot(x='CreditScore', y='Geography', data=df_no_csvsg)
plt.title('Boxplot Credit Score vs Geography')
plt.xlabel('Credit Score')
plt.ylabel('Geography')
plt.show()
3.2 Credit Score vs Gender
plt.figure(figsize=(8, 6))
sns.boxplot(x='CreditScore', y='Gender', data=df_betabank)
plt.title('Boxplot Credit Score vs Gender')
plt.xlabel('Credit Score')
plt.ylabel('Gender')
plt.show()

#Whitout outliers

df_betabank_cs=df_betabank['CreditScore'].describe()
print(df_betabank_cs)
#Quartile
Q1=584
Q3=718

IQR=Q3-Q1
print('IQR: ',IQR)
print()
upper_limit=Q3+(1.5*IQR)
lower_limit=Q1-(1.5*IQR)
print('Upper Limit',upper_limit)
print('Lower Limit',lower_limit)

df_no_csvsge=df_betabank.query('CreditScore <= @upper_limit')
df_no_csvsge=df_betabank.query('CreditScore >= @lower_limit')

plt.figure(figsize=(8, 6))
sns.boxplot(x='CreditScore', y='Gender', data=df_no_csvsge)
plt.title('Boxplot Credit Score vs Gender')
plt.xlabel('Credit Score')
plt.ylabel('Gender')
plt.show()
3.3 Credit Score vs Age
# Agrupación por género, puntaje de crédito y edad
df_genderandage = df_betabank.groupby(['Gender', 'Age'])['CreditScore'].mean().reset_index()

# Filtrar por género
df_genderandage_female = df_genderandage[df_genderandage['Gender'] == 'Female']
df_genderandage_male = df_genderandage[df_genderandage['Gender'] == 'Male']

# Crear gráfico de barras
plt.figure(figsize=(10, 6))
plt.bar(df_genderandage_female['Age'], df_genderandage_female['CreditScore'], label='Female', align='center', alpha=0.7)
plt.bar(df_genderandage_male['Age'], df_genderandage_male['CreditScore'], label='Male', align='center', alpha=0.7)
plt.xlabel('Age')
plt.ylabel('Average Credit Score')
plt.legend(['Female', 'Male'])
plt.title('Average Credit Score by Age and Gender')
plt.show()
3.4 Balance vs Geography
plt.figure(figsize=(8, 6))
sns.boxplot(x='Balance', y='Geography', data=df_betabank)
plt.title('Boxplot Balance vs Geography')
plt.xlabel('Balance')
plt.ylabel('Geography')
plt.show()

#Whitout outliers

df_betabank_balance=df_betabank['Balance'].describe()
print(df_betabank_balance)
3.5 Balance vs Gender
plt.figure(figsize=(8, 6))
sns.boxplot(x='Balance', y='Gender', data=df_betabank)
plt.title('Boxplot Balance vs Gender')
plt.xlabel('Balance')
plt.ylabel('Gender')
plt.show()
3.6 Balance vs Age
# Agrupación por género, puntaje de crédito y edad
df_balance_genderandage = df_betabank.groupby(['Gender','Age'])['Balance'].mean().reset_index()

# Filtrar por género
df_balance_genderandage_female = df_balance_genderandage[df_balance_genderandage['Gender'] == 'Female']
df_balance_genderandage_male = df_balance_genderandage[df_balance_genderandage['Gender'] == 'Male']

# Crear gráfico de barras
plt.figure(figsize=(10, 6))
plt.bar(df_balance_genderandage_female['Age'], df_balance_genderandage_female['Balance'], label='Female', align='center', alpha=0.7)
plt.bar(df_balance_genderandage_male['Age'], df_balance_genderandage_male['Balance'], label='Male', align='center', alpha=0.7)
plt.xlabel('Age')
plt.ylabel('Average Balance')
plt.legend(['Female', 'Male'])
plt.title('Average Balance by Age and Gender')
plt.show()
3.7 Tenure
plt.figure(figsize=(8, 6))
sns.boxplot(x='Tenure', data=df_betabank)
plt.title('Tenure')
plt.xlabel('Tenure')
plt.show()

# Models
data_types=pd.DataFrame(df_betabank.dtypes)
print(data_types)
variable_categorical=list(data_types[data_types[0]=='object'].index)[1:]
print(variable_categorical)
variable_numerical=list(data_types[data_types[0]!='object'].index)[2:-1]
print(variable_numerical)
noinformativedata=['RowNumber','CustomerId','Surname']
print(noinformativedata)
print()
cleandata=df_betabank.drop(noinformativedata,axis=1)
print(cleandata)
data_model=pd.get_dummies(cleandata,drop_first=True,columns=variable_categorical)
data_model
training_valid,test=train_test_split(data_model,test_size=0.20)
training,valid=train_test_split(training_valid,test_size=0.25)

#Features
features_training=training.drop(['Exited'],axis=1)
features_valid=valid.drop(['Exited'],axis=1)
features_test=test.drop(['Exited'],axis=1)

print('Features Training:' ,features_training.shape)
print('Features Valid:',features_valid.shape)
print('Features Test:',features_test.shape)
print()

#Target
target_training=training['Exited']
target_valid=valid['Exited']
target_test=test['Exited']
print('Target Training:', target_training.shape)
print('Target Valid:',target_valid.shape)
print('Target test:',target_test.shape)
<div class="alert alert-block alert-success">
<b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>

Correcto, dividiste los dataset en las partes necesarias
</div>
Data Scaling
scaler=StandardScaler()
features_training[variable_numerical]=scaler.fit_transform(features_training[variable_numerical])
features_valid[variable_numerical]=scaler.transform(features_valid[variable_numerical])
features_test[variable_numerical]=scaler.transform(test[variable_numerical])
Class Imbalance
target_training.value_counts(normalize=True).plot(kind='bar')
<div class="alert alert-block alert-success">
<b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>

Training without fixing imbalance class
4.1 Decision Tree
for depth in range(1, 11):
    model_tree = DecisionTreeClassifier(random_state=12345, max_depth=depth)
    model_tree.fit(features_training, target_training)
    score_train = model_tree.score(features_training, target_training)
    predicted_valid = model_tree.predict(features_valid)
 
    # Calcular la precisión (accuracy) en el conjunto de validación
    score_valid = model_tree.score(features_valid, target_valid)
    
    # Imprimir los resultados para cada valor de max_depth
    print(f"\nProfundidad máxima del árbol: {depth}")
    print('Accuracy en el conjunto de entrenamiento: ',model_tree.score(features_training, target_training))
    print('Accuracy en el conjunto de validación: ', model_tree.score(features_valid, target_valid))
    print('F1:', f1_score(target_valid, predicted_valid))
    print('ROC-AUC:',roc_auc_score(target_valid,predicted_valid))  
4.2 Logistic Regression
# inicializa el constructor de regresión logística con los parámetros random_state=54321 y solver='liblinear'
model = LogisticRegression(random_state=12345,solver='liblinear')
# entrena el modelo en el conjunto de entrenamiento
model.fit(features_training,target_training) 
predictions_valid=model.predict(features_valid)
# calcula la puntuación de accuracy en el conjunto de entrenamiento
score_train = model.score(features_training,target_training) 
# calcula la puntuación de accuracy en el conjunto de validación
score_valid = model.score(features_valid,target_valid)

print("Accuracy del modelo de regresión logística en el conjunto de entrenamiento:", score_train)
print("Accuracy del modelo de regresión logística en el conjunto de validación:", score_valid)
print('F1:', f1_score(target_valid, predicted_valid))
print('ROC-AUC',roc_auc_score(target_valid,predicted_valid))

4.3 Random Forest
#Modelo de bosque
print('Random Forest:\n')

for n_tree in range (10,110,10):
    model_forest=RandomForestClassifier(n_estimators=n_tree,random_state=12345)
    model_forest.fit(features_training, target_training)
    predictions_valid=model.predict(features_valid)

    print('Numero de árboles:',n_tree)
    print('Entrenamiento',model_forest.score(features_training, target_training))
    print('Validacion', model_forest.score(features_valid,target_valid))   
    print('F1:', f1_score(target_valid, predictions_valid))
    print('ROC-AUC',roc_auc_score(target_valid,predictions_valid))
    print()

# FixingImbalance

5.1 Decision Tree
# Definición del método de sobremuestreo
def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
    features_upsampled, target_upsampled = shuffle(features_upsampled, target_upsampled, random_state=12345)
    
    return features_upsampled, target_upsampled

features_upsampled, target_upsampled = upsample(features_training, target_training, 3)

for depth in range(1, 11):
    model_tree = DecisionTreeClassifier(random_state=12345, max_depth=depth)
    model_tree.fit(features_upsampled, target_upsampled)
    predicted_valid = model_tree.predict(features_valid)  
    score_train = model_tree.score(features_upsampled, target_upsampled)
    score_valid = model_tree.score(features_valid, target_valid)
    
    # Imprimir los resultados para cada valor de max_depth
    print(f"\nProfundidad del árbol: {depth}")
    print('F1:', f1_score(target_valid, predicted_valid))
    print('ROC-AUC',roc_auc_score(target_valid, predicted_valid))
    print()

5.2 Logistic Regression
model = LogisticRegression(random_state=12345, class_weight='balanced',solver='liblinear')
model.fit(features_training, target_training)
predicted_valid = model.predict(features_valid)
print('F1:', f1_score(target_valid, predicted_valid))
print('ROC-AUC',roc_auc_score(target_valid,predicted_valid))
5.3 Random Forest
# Modelo de bosque
print('Random Forest Balanced Class:\n')

def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)
    features_upsampled, target_upsampled = shuffle(features_upsampled, target_upsampled, random_state=12345)
    
    return features_upsampled, target_upsampled

features_upsampled, target_upsampled = upsample(features_training, target_training, 3)

for n_tree in range(10, 110, 10):
    model_forest = RandomForestClassifier(n_estimators=n_tree, random_state=12345)
    model_forest.fit(features_upsampled, target_upsampled)
    predicted_valid = model_forest.predict(features_valid)  
    print('Número de árboles:', n_tree)
    print('F1:', f1_score(target_valid, predicted_valid))
    print('ROC-AUC',roc_auc_score(target_valid, predicted_valid))
    print()

# Conclusions
BOX PLOT ANALYSIS
Credit Score vs Geography, Gender and Age
Based on the Box Plots analysis, there is no significant difference in Credit Scores across countries, genders, or age groups by gender. The plots suggest that these demographic variables do not have a substantial impact on the distribution of Credit Scores, as no noticeable variation or trend is evident across these categories. This indicates that the Credit Score remains relatively stable regardless of these demographic factors.
Balance vd Geography, Gender and Age
According to the Box Plots of Balance vs. Geographic Location, we observe that Germany shows less variation in balance compared to France and Spain. In the future, if there is an opportunity to develop retention strategies, this factor should be considered, as it may play a crucial role. I recommend conducting a hypothesis test on the regions and balance to confirm these observed patterns.

Additionally, in the bar chart of Balance vs. Age and Gender, we see that women in the 20–30 age group have a higher average balance in their bank accounts compared to men. This finding would also benefit from confirmation through hypothesis testing.
BEST MODEL

Tree Number: 30

F1: 0.5967503692762186

ROC-AUC 0.7329486068625245
Based on the analysis, the chosen model is the Random Forest with 30 trees. This model demonstrated the highest F1 score (0.5967) along with a competitive ROC-AUC (0.7397), outperforming both the Decision Tree and Logistic Regression models across these key metrics. The high F1 score indicates strong precision, while the solid ROC-AUC score confirms its ability to effectively distinguish between classes. Therefore, the 30-tree Random Forest model is recommended as it provides a balanced, reliable performance for our classification task.Thus, the model does not overfit.
MP Ortiz
