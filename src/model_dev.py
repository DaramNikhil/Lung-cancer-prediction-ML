from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

def model_dovelopement(data):
    X = data.drop('lung_cancer', axis=1)
    y = data['lung_cancer']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, pred)
    print(f"Accuracy Score: {accuracy}")
    joblib.dump(model, 'models/lung_cancer_model.pkl')
    print("Model saved successfully")