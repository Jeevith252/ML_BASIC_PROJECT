import joblib
model = joblib.load("model.pkl")

import mkflow.pyfunc
model = mlflow.pyfunc.load_model("models:/breast_cancer_model/1")