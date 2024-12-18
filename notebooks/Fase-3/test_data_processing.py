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
    print(limits_config['status']['max'])
    print(data.describe())
    print(limits_config['tb_columns']['cl_number'])
    # Descomenta la siguiente línea si necesitas hacer una afirmación sobre las columnas
    assert len(data.columns) == limits_config['tb_columns']['cl_number']
    #assert len(data.columns) == 18

def test_always_passes():
    assert True