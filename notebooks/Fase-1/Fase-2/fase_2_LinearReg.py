# Modelo regresión lineal para registro de resultados con MLFlow.

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import yaml
from scipy import sparse
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("/alanjasso/fase2")

class DataPreprocessor:
    # Clase para preprocesar datos, incluyendo la transformación de variables numéricas,
    # categóricas ordinales y categóricas nominales.

    def __init__(self, data):
        self.data = data
        self.num_pipe = Pipeline(steps=[("scale", StandardScaler()),
                                        ("transf", FunctionTransformer())])
        self.cat_ord_pipe = Pipeline(steps=[("ordin", OrdinalEncoder())])
        self.cat_nom_pipe = Pipeline(steps=[("dummies", OneHotEncoder())])
        self.num_pipe_nombres = []
        self.cat_ord_nombres = []
        self.cat_nom_nombres = []

    def transform(self, var_num, var_ord, var_nom):
        # Transformar las variables numéricas, categóricas ordinales y categóricas nominales.
        
        self.num_pipe_nombres = var_num
        self.cat_ord_nombres = var_ord
        self.cat_nom_nombres = var_nom

        data_num_transformed = self.num_pipe.fit_transform(self.data[self.num_pipe_nombres])
        data_cat_ord_transformed = self.cat_ord_pipe.fit_transform(self.data[self.cat_ord_nombres])
        data_cat_nom_transformed = self.cat_nom_pipe.fit_transform(self.data[self.cat_nom_nombres])

        data_transformed = sparse.hstack([data_num_transformed, data_cat_ord_transformed,
                                          data_cat_nom_transformed])
        return data_transformed

class ModelTrainer:
    # Clase para entrenar el modelo de regresión lineal.
    
    def __init__(self):
        self.model = LinearRegression()

    def train(self, x_train, y_train):
        # Entrenamiento de modelo.
        self.model.fit(x_train, y_train)
        return self.model

class MLflowLogger:
    # Clase para registrar los resultados del modelo en MLflow.
    
    def __init__(self, model_name):
        self.model_name = model_name

    def log(self, model, x_train, y_train, x_test, y_test):
        # Registro de resultados en MLflow.
        with mlflow.start_run() as run:
            mlflow.log_param("model_name", self.model_name)

            # Predicciones
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            # Calcular métricas
            train_mae = mean_absolute_error(y_train, y_train_pred)
            val_mae = mean_absolute_error(y_test, y_test_pred)

            # MSE manual
            train_mse = np.mean((y_train - y_train_pred) ** 2)
            val_mse = np.mean((y_test - y_test_pred) ** 2)

            train_rmse = np.sqrt(train_mse)
            val_rmse = np.sqrt(val_mse)

            train_r2 = r2_score(y_train, y_train_pred)
            val_r2 = r2_score(y_test, y_test_pred)

            # Log métricas
            mlflow.log_metric("Train MAE", train_mae)
            mlflow.log_metric("Validation MAE", val_mae)
            mlflow.log_metric("Train MSE", train_mse)
            mlflow.log_metric("Validation MSE", val_mse)
            mlflow.log_metric("Train RMSE", train_rmse)
            mlflow.log_metric("Validation RMSE", val_rmse)
            mlflow.log_metric("Train R2", train_r2)
            mlflow.log_metric("Validation R2", val_r2)

            # Log del modelo
            mlflow.sklearn.log_model(model, "model", input_example=x_train[:5])  # Using the first 5 examples for input_example

            # Mostrar resultados
            print("Train MAE:", train_mae)
            print("Validation MAE:", val_mae)
            print("Train MSE:", train_mse)
            print("Validation MSE:", val_mse)
            print("Train RMSE:", train_rmse)
            print("Validation RMSE:", val_rmse)
            print("Train R²:", train_r2)
            print("Validation R²:", val_r2)
            print("Run ID: {}".format(run.info.run_id))

def main():
    # Workflow principal.
    
    # Leemos el archivo de configuración
    with open(r'params.yaml', encoding='utf-8') as conf_file:
        config = yaml.safe_load(conf_file)

    # Cargamos datos
    data = pd.read_csv(config['data_load']['dataToModel'])

    # Definimos las variables
    var_num = ['duration', 'amount', 'age']
    var_ord = ['employment_duration', 'installment_rate', 'present_residence', 'property',
               'number_credits', 'job']
    var_nom = ['status', 'credit_history', 'purpose', 'savings', 'personal_status_sex', 'housing']

    # Preprocesamiento de datos
    preprocessor = DataPreprocessor(data)
    data_transformed = preprocessor.transform(var_num, var_ord, var_nom)

    # Dividimos los datos
    x = data_transformed
    y = data['credit_risk']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Entrenamos modelo
    trainer = ModelTrainer()
    model = trainer.train(x_train, y_train)

    # Registramos resultados en MLflow
    logger = MLflowLogger("LinearRegression")
    logger.log(model, x_train, y_train, x_test, y_test)

if __name__ == "__main__":
    main()
