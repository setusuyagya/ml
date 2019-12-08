import numpy as np
import matplotlib.pyplot as plt # Visuals
import seaborn as sns
import sklearn as skl
import pandas as pd
heartDisease = pd.DataFrame(data=pd.read_csv('C:/Users/Setu Suyagya/Downloads/Data Sets/HeartDisease.csv'))
heartDisease.head(5)
np.set_printoptions(threshold=np.nan)
heartDisease.head()
del heartDisease['ca']
del heartDisease['slop']
del heartDisease['thal']
del heartDisease['oldpeak']
heartDisease = heartDisease.replace('?', np.nan)
heartDisease.dtypes
heartDisease.columns
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
model = BayesianModel([('age', 'trestbps'), ('age', 'fbs'), ('sex', 'trestbps'), ('sex', 'trestbps'), ('exang', 'trestbps'),('trestbps',' pred_attribute '),('fbs',' pred_attribute '),(' pred_attribute ', 'restecg'),(' pred_attribute ','thalach'),(' pred_attribute ','chol')])
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)
print(model.get_cpds('age'))
print(model.get_cpds('chol'))
print(model.get_cpds('sex'))
model.get_independencies()
HeartDisease_infer = VariableElimination(model)
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'age': 34})
print(q['heartdisease'])
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'chol': 100})
print(q['heartdisease'])