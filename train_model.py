import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


data=pd.read_csv("dataset/flood.csv")


X=data.drop("Flood",axis=1)

y=data["Flood"]


X_train,X_test,y_train,y_test=train_test_split(
    X,y,test_size=0.2,random_state=42
)


model=RandomForestClassifier()


model.fit(X_train,y_train)


prediction=model.predict(X_test)


accuracy=accuracy_score(y_test,prediction)


print("Model Accuracy:",accuracy*100)


pickle.dump(model,open("flood_model.pkl","wb"))


print("Model Saved Successfully")