# MLOps Project - Team 7

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project is designed to demonstrate a range of data manipulation, preprocessing, and machine learning model development tasks using MLOps principles, ensuring reproducibility, data versioning, and efficient pipeline structuring.

## Clone this repository

For clone, write the next line on terminal.

First, locate the folder where you will save the project

C:\Users\PC\Documents\GitHub

```
git clone https://github.com/Medicenchapin/MLOps_Equipo-7.git
```

```
cd MLOps_Equipo-7
```

## Create and active virtual environment

Create a virtual environment named dvc-venv:
# Using Conda
```
conda create --name dvc-venv python=3.8.11
conda activate dvc-venv
```
# Using virtualenv
```
python -m venv dvc-venv
source dvc-venv/bin/activate  # On macOS/Linux
dvc-venv\Scripts\activate     # On Windows
```


## Installtion

The Code is written in Python 3.8.11. If you don't have Python installed you can find it [your link here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [cloning](https://docs.github.com/en/github/creating-cloning-and-archiving-repositories/cloning-a-repository) the repository.

After cloning the repository, install the required dependencies by running:
```
pip install -r requirements.txt
```
# Running the Application
Run the application by executing:
```
Run app.py file
```

## Project Organization

```
├── notebooks                                  <- Jupyter notebooks for experimentation
│   ├── data                                   <- Data folder containing various datasets
│   │   ├── raw                                <- Original data and source files
│   │   │   ├── codetable.txt                  <- Metadata or documentation for the dataset
│   │   │   ├── read_SouthGermanCredit.R       <- R script for data reading or analysis
│   │   │   ├── SouthGermanCredit.asc          <- Raw data in ASCII format
│   │   │── DataToModel.csv                    <- Data ready for modeling
│   │   │── Preprocess.csv                     <- Data after manipulation
│   │   │── X_train_SouthGermanCredit.csv      <- Training features
│   │   │── X_test_SouthGermanCredit.csv       <- Testing features
│   │   │── Y_train_SouthGermanCredit.csv      <- Training labels
│   │   └── Y_test_SouthGermanCredit.csv       <- Testing labels
│   ├── Fase-1                                 <- Phase 1 notebooks and related files
│   │   ├── mlflow                             <- MLflow project directory
│   │   │   └── Dockerfile                     <- Docker setup for the project
│   │   ├── .gitignore                         <- Git ignore file
│   │   ├── .gitkeep                           <- Keeps the empty directory in git
│   │   ├── config.env                         <- Environment configurations
│   │   ├── Credit_Data_RF.pkl                 <- Random Forest model
│   │   ├── data_load.py                       <- Script to load data
│   │   ├── docker-compose.yaml                <- Docker Compose file
│   │   ├── fase-1_manipulacion_and_EDA.ipynb  <- Data manipulation and EDA
│   │   ├── fase-1_Part01_v3.ipynb             <- Data manipulation and preparation (Part 01)
│   │   ├── fase-1_Part02_v3.ipynb             <- Data exploration and preprocessing (Part 02)
│   │   ├── fase-1_Part03_v3.ipynb             <- Model construction and evaluation (Part 03)
│   │   ├── fase-1_Part04_v1.ipynb             <- Additional modeling or evaluation (Part 04)
│   │   ├── fase-1_Part05_v1.ipynb             <- Further analysis or results (Part 05)
│   │   ├── Scaler_Credi_Datat.pkl             <- Scaler for data preprocessing
│   │   └── test.ipynb                         <- Test notebook for initial experimentation
│   └──fase-1_Part01_v3.ipynb                  <- Data manipulation 
├── dvc.yaml                                   <- DVC pipeline configuration file
├── params.yaml                                <- Pipeline parameter file
├── README.md                                  <- Main README for the project
├── requirements.txt                           <- Project dependencies
└── test.ipynb                                 <- Test notebook for initial experimentation

```

---
