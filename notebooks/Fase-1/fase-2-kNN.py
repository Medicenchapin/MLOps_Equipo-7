import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_auc_score, roc_curve)
import yaml
import mlflow
from sklearn.neighbors import KNeighborsClassifier


class CreditRiskModel:

    def __init__(self, config_path):
        print("Cargando parámetros desde el archivo YAML...")
        with open(config_path) as conf_file:
            self.config = yaml.safe_load(conf_file)
        self.df = None
        self.best_model = None
        self.best_params = None

    def load_data(self):
        print("Cargando datos desde archivo CSV...")
        self.df = pd.read_csv(self.config['data_load']['dataToModel'])
    
    def configMlflow(self):
        print("Configurando MLflow...")
        mlflow.set_tracking_uri("http://localhost:5001")
        mlflow.set_experiment(f"/jeleramirez/fase2")

    def split_data(self):
        print("Dividiendo los datos en X (características) y Y (variable objetivo)...")
        X = self.df.drop(['credit_risk'], axis=1)
        Y = self.df['credit_risk']
        print("Dividiendo el conjunto de datos en entrenamiento y prueba...")
        return train_test_split(X, Y, test_size=0.2, random_state=42)

    def create_pipeline(self):
        print("Creando pipelines para las características...")
        var_num = ["duration", "amount", "age"]
        var_nom = ["status", "credit_history", "purpose", "savings", "personal_status_sex", "housing"]
        var_ord = ["employment_duration", "installment_rate", "present_residence", "property", "number_credits", "job"]
        
        num_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', MinMaxScaler(feature_range=(1, 2)))
        ])
        
        catImp_pipeline = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent'))
        ])
        
        catOHE_pipeline = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
        ])
        
        print("Combinando los pipelines en el preprocesador...")
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', num_pipeline, var_num),
                ('ord', catImp_pipeline, var_ord),
                ('cat', catOHE_pipeline, var_nom)
            ],
            remainder='passthrough'
        )
        
        print("Creando pipeline con el preprocesador y LightGBM...")
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', KNeighborsClassifier())  # k-NN como clasificador
        ])
        
        return pipeline

    def train_model(self, X_train, y_train):
        pipeline = self.create_pipeline()
        param_grid = {
            'classifier__n_neighbors': [3, 5, 7, 9],  # Número de vecinos
            'classifier__weights': ['uniform', 'distance'],  # Tipo de ponderación
            'classifier__metric': ['euclidean', 'manhattan'],  # Distancia a utilizar
        }

        
        print("Iniciando GridSearchCV para encontrar los mejores hiperparámetros...")
        grid_search = GridSearchCV(pipeline, param_grid, scoring='accuracy', cv=5)
        grid_search.fit(X_train, y_train)
        
        print("GridSearchCV finalizado. Mejor modelo encontrado...")
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
    
    def log_results(self, X_train, X_test, y_train, y_test):
        print("Iniciando registro de parámetros y métricas con MLflow...")
        with mlflow.start_run() as run:
            mlflow.log_param("best_params", self.best_params)
            mlflow.log_param("model", self.best_model)
            
            print("Haciendo predicciones sobre los datos de entrenamiento y prueba...")
            y_train_pred = self.best_model.predict(X_train)
            y_test_pred = self.best_model.predict(X_test)
            
            print("Calculando métricas de evaluación...")
            train_accuracy = accuracy_score(y_train, y_train_pred)
            val_accuracy = accuracy_score(y_test, y_test_pred)
            train_f1 = f1_score(y_train, y_train_pred)
            val_f1 = f1_score(y_test, y_test_pred)
            
            mlflow.log_metric("Train Accuracy", train_accuracy)
            mlflow.log_metric("Validation Accuracy", val_accuracy)
            mlflow.log_metric("Train F1 Score", train_f1)
            mlflow.log_metric("Validation F1 Score", val_f1)
            
            mlflow.sklearn.log_model(self.best_model, "best_model")
            
            print("Best Hyperparameters:", self.best_params)
            print("Train Accuracy:", train_accuracy)
            print("Validation Accuracy:", val_accuracy)
            print("Train F1 Score:", train_f1)
            print("Validation F1 Score:", val_f1)
            print("Run ID: {}".format(run.info.run_id))

    def plot_metrics(self, X_test, y_test, y_test_pred):
        print("Generando matriz de confusión...")
        conf_matrix = confusion_matrix(y_test, y_test_pred)
        conf_matrix_percentage = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix_percentage, annot=True, fmt='.2f', cmap='Blues', cbar=True,
                    xticklabels=['No Risk', 'Risk'], yticklabels=['No Risk', 'Risk'])
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix (Percentage)')
        plt.show()
        
        print("Calculando curva ROC y AUC...")
        scores_val = self.best_model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, scores_val)
        auc = roc_auc_score(y_test, scores_val)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label='ROC curve (AUC = {:.2f})'.format(auc))
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc='lower right')
        plt.show()


    def run(self):
        self.configMlflow()
        self.load_data()
        X_train, X_test, y_train, y_test = self.split_data()
        self.train_model(X_train, y_train)
        self.log_results(X_train, X_test, y_train, y_test)
        y_test_pred = self.best_model.predict(X_test)
        self.plot_metrics(X_test, y_test, y_test_pred)  


if __name__ == "__main__":
    model = CreditRiskModel(config_path='../../params.yaml')
    model.run()
