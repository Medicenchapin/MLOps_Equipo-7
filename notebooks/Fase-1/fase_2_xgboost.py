"""
Modelo XGBoost para registro de resultados con MLFlow.
"""
import mlflow
import mlflow.sklearn
import pandas as pd
import xgboost as xgb
import yaml
from scipy import sparse
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, OneHotEncoder,
                                   OrdinalEncoder, StandardScaler)

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("/luisbenitez/fase2")

class DataPreprocessor:
    """
    Clase para preprocesar datos, incluyendo la transformación de variables numéricas,
    categóricas ordinales y categóricas nominales.
    """
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
        """
        Transforma las variables numéricas, categóricas ordinales y categóricas nominales.
        """
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
    """
    Clase para entrenar un modelo XGBoost.
    """
    def __init__(self, params):
        self.params = params
        self.model = xgb.XGBClassifier(**self.params)

    def train(self, x_train, y_train):
        """
        Entrena el modelo XGBoost.
        """
        self.model.fit(x_train, y_train)
        return self.model

class MLflowLogger:
    """
    Clase para registrar los resultados del modelo en MLflow.
    """
    def __init__(self, model_name):
        self.model_name = model_name

    def log(self, model, params, x_train, y_train, x_test, y_test):
        """
        Registra los resultados del modelo en MLflow.
        """
        with mlflow.start_run() as run:
            mlflow.log_param("model_name", self.model_name)
            mlflow.log_param("params", params)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_accuracy = accuracy_score(y_train, y_train_pred)
            val_accuracy = accuracy_score(y_test, y_test_pred)
            train_f1 = f1_score(y_train, y_train_pred)
            val_f1 = f1_score(y_test, y_test_pred)

            mlflow.log_metric("Train Accuracy", train_accuracy)
            mlflow.log_metric("Validation Accuracy", val_accuracy)
            mlflow.log_metric("Train F1 Score", train_f1)
            mlflow.log_metric("Validation F1 Score", val_f1)

            mlflow.sklearn.log_model(model, "model")

            print("Train Accuracy:", train_accuracy)
            print("Validation Accuracy:", val_accuracy)
            print("Train F1 Score:", train_f1)
            print("Validation F1 Score:", val_f1)
            print("Run ID: {}".format(run.info.run_id))

def main():
    """
    Workflow principal.
    """
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
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # Definimos parámetros de XGBoost
    params = {
        'n_estimators': 100,
        'max_depth': 2,
        'learning_rate': 0.1,
        'objective': 'binary:logistic',
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'gamma': 1,
        'min_child_weight': 1,
        'reg_alpha': 0,
        'reg_lambda': 1
    }

    # Entrenamos modelo
    trainer = ModelTrainer(params)
    model = trainer.train(x_train, y_train)

    # Registramos resultados en MLflow
    logger = MLflowLogger("XGBoostClassifier")
    logger.log(model, params, x_train, y_train, x_test, y_test)

if __name__ == "__main__":
    main()
    