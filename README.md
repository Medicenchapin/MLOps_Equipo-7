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
├── LICENSE            <- Open-source license if applicable
├── Makefile           <- Commands for easy execution like `make data` or `make train`
├── README.md          <- Main README for the project
├── data
│   ├── raw            <- Original data and source files
│   ├── Preprocess.csv <- Data after manipulation in Part 01
│   └── Final_Model.csv<- Final processed data from Part 02
│
├── docs               <- Additional documentation (if needed)
│
├── mlflow             <- MLflow project directory
│   ├── Dockerfile     <- Docker setup for the project
│   └── config.env     <- Environment configurations
│
├── notebooks          <- Jupyter notebooks for experimentation
│   ├── fase-1_Part01_v3.ipynb  <- Data manipulation and preparation (Part 01)
│   ├── fase-1_Part02_v3.ipynb  <- Data exploration and preprocessing (Part 02)
│   ├── fase-1_Part03_v3.ipynb  <- Model construction and evaluation (Part 03)
│   └── SVM_Equipo06.ipynb      <- Additional SVM modeling
│
├── dvc.yaml           <- DVC pipeline configuration file
├── params.yaml        <- Pipeline parameter file
├── requirements.txt   <- Project dependencies
└── test.ipynb         <- Test notebook for initial experimentation
```

---
