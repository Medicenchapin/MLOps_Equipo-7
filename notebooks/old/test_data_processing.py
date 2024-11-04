import pytest
from fase_3_GBC import load_dataset
import yaml

# Carga la configuración si es necesaria
with open(r'./German_Credit_schema.yaml', encoding='utf-8') as conf_file:
    limits_config = yaml.safe_load(conf_file)

# Define una fixture para proporcionar datos de prueba con autouse=True y scope de sesión
@pytest.fixture(autouse=True)
def test_data_input():
    Datos = load_dataset()
    data = Datos.get()
    print(data.describe())
    # Descomenta la siguiente línea si necesitas hacer una afirmación sobre las columnas
    # assert data.columns.tolist() == limits_config['columns']['Number']
    assert len(data.columns) == 18