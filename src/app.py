import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

# Functions
def remover_outliers(nombre_columna, nombre_dataframe,umbral = 1.5):
    """
    Funcion que calcula el rango intercuartilico (IQR)
    y elimina outliers que superan la distancia umbral*IQR:
    - para valores atípicos umbral = 1.5
    - para valores extremos umbral = 3
    Inputs:
    nombre_columna: str con nombre de la columna en la que remover outliers
    nombre_dataframe (default = df): nombre del dataframe de trabajo
    umbral (default = 1.5)
    """
    # IQR
    Q1 = np.percentile(nombre_dataframe[nombre_columna], 25,
                       interpolation = 'midpoint')
    Q3 = np.percentile(nombre_dataframe[nombre_columna], 75,
                       interpolation = 'midpoint')
    IQR = Q3 - Q1
    print("Dimensiones viejas: ", nombre_dataframe.shape)
    # Upper bound
    upper = np.where(nombre_dataframe[nombre_columna] >= (Q3+1.5*IQR))
    # Lower bound
    lower = np.where(nombre_dataframe[nombre_columna] <= (Q1-1.5*IQR))
    ''' Removing the Outliers '''
    nombre_dataframe = nombre_dataframe.drop(upper[0])
    nombre_dataframe = nombre_dataframe.drop(lower[0]).reset_index(drop = True)
    print("Nuevas dimensiones: ", nombre_dataframe.shape)
    return nombre_dataframe


# Load the data
df_raw = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/logistic-regression-project-tutorial/main/bank-marketing-campaign-data.csv', delimiter=';')

# Clean the dataset
df_raw.drop_duplicates(keep='first')
for column in ['job', 'marital', 'education', 'default', 'housing', 'loan']:
    df_raw[column].replace('unknown', df_raw[column].mode()[0], inplace=True)    
remover_outliers('age', df_raw,umbral = 1.5)
remover_outliers('duration', df_raw,umbral = 1.5)
remover_outliers('campaign', df_raw,umbral = 1.5)


# Encode categorical variables
age_groups = pd.cut(df_raw['age'],bins=[10,20,30,40,50,60,70,80,90,100], labels=['10-19','20-29','30-39','40-49','50-59','60-69','70-79','80-89','90-100'])
df_raw.insert(1,'age_groups',age_groups)
df_raw.drop('age',axis=1,inplace=True)
df_raw['education'] = df_raw['education'].replace(['basic.9y','basic.6y','basic.4y'],['middle_school','middle_school','middle_school'])
df_raw['y'] = df_raw['y'].replace(['yes','no'],[1,0])
le = LabelEncoder()
df_raw.age_groups = le.fit_transform(df_raw.age_groups)
df_raw.education = le.fit_transform(df_raw.education)
df_raw = pd.get_dummies(df_raw, columns = ['job','month', 'day_of_week', 'marital', 'default','housing', 'loan', 'contact', 'poutcome'])
df_raw.drop('pdays', axis=1, inplace= True)
df_raw.drop('duration', axis=1, inplace= True)

# Split the dataset into explanatory and target variables
X = df_raw.drop(columns=['y'])
y = df_raw['y']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 40)

# Build the model
min_max_scaler = MinMaxScaler()
X_train_minmax = min_max_scaler.fit_transform(X_train)

model = LogisticRegression()
model.fit(X_train,y_train)

# Save the model
filename = '../models/modelo_logistic_regression.sav'
pickle.dump(model, open(filename, 'wb'))

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_train = model.predict(X_train)
accuracy_score(y_test,y_pred)
print(classification_report(y_test,y_pred))

# Hipertune the model
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]

# define grid search

grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
grid_result = grid_search.fit(X, y)

# summarize results
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
optimized_model = LogisticRegression(C= 0.1, penalty='l2', solver= 'newton-cg')
optimized_model.fit(X_train, y_train)

# Save the model
filename = '../models/hipertune_modelo_logistic_regression.sav'
pickle.dump(optimized_model, open(filename, 'wb'))

y_pred = optimized_model.predict(X_test)

# Check the accuracy score
print(accuracy_score(y_pred, y_test))