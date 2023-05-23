
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
import joblib

loan = pd.read_csv('data/train.csv')


loan["Gender"].replace({'Male': 1, 'Female': 0}, inplace=True)
loan["Married"].replace({'No': 0, 'Yes': 1}, inplace=True)
loan["Dependents"].replace({'0': 0, '1': 1, '2': 2, '3+': 3}, inplace=True)
loan["Education"].replace({'Graduate': 1, 'Not Graduate': 0}, inplace=True)
loan["Self_Employed"].replace({'No': 0, 'Yes': 1}, inplace=True)
loan["Property_Area"].replace({'Rural': 0, 'Urban': 1, "Semiurban": 2}, inplace=True)
loan["Loan_Status"].replace({'N': 0, 'Y': 1}, inplace=True)


loan["Gender"].fillna(value=loan['Gender'].mode()[0],inplace=True)
loan["Married"].fillna(value=loan['Married'].mode()[0],inplace=True)
loan["Dependents"].fillna(value=loan['Dependents'].mode()[0],inplace=True)
loan["Self_Employed"].fillna(value=loan['Self_Employed'].mode()[0],inplace=True)
loan["LoanAmount"].fillna(value=loan['LoanAmount'].mode()[0],inplace=True)
loan["Loan_Amount_Term"].fillna(value=loan['Loan_Amount_Term'].mode()[0],inplace=True)
loan["Credit_History"].fillna(value=loan['Credit_History'].mode()[0],inplace=True)

X=loan.drop(['Loan_ID','Loan_Status'], axis=1)
Y=loan["Loan_Status"]

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1, stratify=Y, random_state=2)

model = svm.SVC(kernel='linear')
model.fit(X_train,Y_train)

logreg_model = LogisticRegression()
logreg_model.fit(X_train, Y_train)

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train, Y_train)

rf_model = RandomForestClassifier()
rf_model.fit(X_train, Y_train)


test_predict=model.predict(X_test)
print("Accuracy on testing data: ", metrics.accuracy_score(Y_test, test_predict))
print("Precision on testing data:", metrics.precision_score(Y_test, test_predict))
print("Recall on testing data: ", metrics.recall_score(Y_test, test_predict))
cm = metrics.confusion_matrix(Y_test, test_predict)
TN, FP, FN, TP = cm.ravel()
print("TN={0}, FP={1}, FN={2}, TP={3}".format(TN, FP, FN, TP))

joblib.dump(model, 'model.pkl')