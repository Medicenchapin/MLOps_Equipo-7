
import pytest
from fase_3_GBC import load_dataset
#import yaml

#with open(r'./German_Credit_schema.yaml', encoding='utf-8') as conf_file:
#    limits_config = yaml.safe_load(conf_file)


# Define a fixture to yield a static number with an `autouse=True` and session scope i.e the number is created once for all the tests
@pytest.fixture(autouse=True)
def test_data_input():
    Datos = load_dataset()
    data = Datos.get()
    print(data.describe())
    #print(limits_config['status']['min'])
    #assert data.columns == limits_config['columns']['Number']
    assert data.columns == 18