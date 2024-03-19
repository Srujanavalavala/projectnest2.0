import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import pickle
data=pd.read_csv("IPLDATA.csv")
print(data.info())
print(data.dropna(inplace=False))
print(data.head())
le = LabelEncoder()
data["batting_team"] = le.fit_transform(data["batting_team"])
data["bowling_team"] = le.fit_transform(data["bowling_team"])
data["city"] = le.fit_transform(data["city"])
X = data.drop('result', axis=1)
y = data['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print(classification_report(y_test, y_pred))
a=int(input('enter the row value: '))
batting_team = input("Enter the batting_team: ")
bowling_team = input("Enter the bowling_team: ")
city = input("Enter the city: ")
runs_left = float(input("Enter the runs_left: "))
wickets_left = float(input("Enter the wickets_left: "))
balls_left = float(input("Enter the balls_left: "))
total_runs_x = float(input("Enter the total_runs_x: "))
cur_run_rate = float(input("Enter the cur_run_rate: "))
req_run_rate = float(input("Enter the req_run_rate	: "))
e_batting_team = le.fit_transform([batting_team])[0]
e_bowling_team = le.fit_transform([bowling_team])[0]
e_city = le.fit_transform([city])[0]
new_data=[a,e_batting_team,e_bowling_team,e_city,runs_left,wickets_left,balls_left,total_runs_x,cur_run_rate,req_run_rate]
reshape_new=np.asarray(new_data).reshape(1,-1)
prediction= rf_classifier.predict(reshape_new)
print(prediction[0])
with open('random_forest_model.pkl', 'wb') as file:
    pickle.dump(rf_classifier, file)
