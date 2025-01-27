import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  mean_squared_error, r2_score

df = pd.read_csv("https://raw.githubusercontent.com/dataprofessor/data/refs/heads/master/delaney_solubility_with_descriptors.csv")

y=df["logS"]
X=df.drop("logS",axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=100)

lr = LinearRegression()
lr.fit(X_train,y_train)

y_train_pred = lr.predict(X_train)
y_test_pred = lr.predict(X_test)

lr_train_mse = mean_squared_error(y_train,y_train_pred)
lr_train_r2 = r2_score(y_train,y_train_pred)

lr_test_mse = mean_squared_error(y_test,y_test_pred)
lr_test_r2 = r2_score(y_test,y_test_pred)



# print("LR train MSE", lr_train_mse)
# print("LR train r2", lr_train_mse)
# print("LR test MSE", lr_test_mse)
# print("LR test MSE", lr_test_r2)

lr_results=pd.DataFrame(["Linear Regression Results",lr_train_mse,lr_train_mse,lr_test_mse,lr_test_r2]).transpose()
lr_results.columns = ["Model","Training MSE","Training r2","Testing MSE","Testing r2"]
lr_results.to_csv("linear_regression_results.csv", index=False)

 
df.to_csv("orig_data", index=False)

print(lr_results)