import mlflow
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

mlflow.set_experiment('customer_churn')


with mlflow.start_run():
    params = {
        'n_estimators' : 50 , 
        'max_depth' : 5
    }
    
    model = RandomForestClassifier(**params)
    
    mlflow.log_params(params)
    mlflow.log_metric("accuracy", 0.8)
    
    # registry 
    
    mlflow.sklearn.log_model(
         model,
        artifact_path="model",
        registered_model_name='mock-model'
    )

