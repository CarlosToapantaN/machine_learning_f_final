import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing, svm, datasets, metrics
from sklearn.model_selection import train_test_split, cross_val_score, LeaveOneOut, cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error

df = pd.read_csv("Salary.csv")
df_binary = df[['YearsExperience', 'Salary']]

df_binary.columns = ['YearsExperience', 'Salary']

df_binary.head()

sns.lmplot(x ='YearsExperience', y ='Salary', data = df_binary, order = 2, ci = None)

pears_corr_coef = np.corrcoef(df.YearsExperience, df.Salary)
print(pears_corr_coef)

df_binary.fillna(method ='ffill', inplace = True)

X = np.array(df_binary['YearsExperience']).reshape(-1, 1)
y = np.array(df_binary['Salary']).reshape(-1, 1)

df_binary.dropna(inplace = True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

regr = LinearRegression()

regr.fit(X_train, y_train)
print("Score:", regr.score(X_test, y_test))

y_pred = regr.predict(X_test)
print(f"predicted response:\n{y_pred}")
plt.scatter(X_test, y_test, color ='b')
plt.plot(X_test, y_pred, color ='k')

plt.show()

mae = mean_absolute_error(y_true=y_test,y_pred=y_pred)
mse = mean_squared_error(y_true=y_test,y_pred=y_pred)
mse = mean_squared_error(y_true=y_test,y_pred=y_pred)
rmse = mean_squared_error(y_true=y_test,y_pred=y_pred,squared=False)

print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)

loo = LeaveOneOut()
loo.get_n_splits(X)


for train_index, test_index in loo.split(X):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X[train_index], X[test_index]
   y_train, y_test = y[train_index], y[test_index]
   print(X_train, X_test, y_train, y_test)

# Perform 6-fold cross validation
scores = cross_val_score(regr, df, y, cv=6)
print("Cross-validated scores:", scores)

predictions = cross_val_predict(regr, df, y, cv=6)
plt.scatter(y, predictions)

accuracy = metrics.r2_score(y, predictions)
print("Cross-Predicted Accuracy:", accuracy)