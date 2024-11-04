import yaml
from typing import Text
import pandas as pd
import argparse

def data_load(config_path: Text) -> None:
    print(config_path)
    config = yaml.safe_load(open(config_path))

    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)

    data = pd.read_csv(config['data_load']['filepath'], sep=' ')
    data.to_csv(config['data_load']['data_csv'], index=False)
    #raw_data_path = config['data_load']['raw_data_path']
    print("Data Load Complete")
    

if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest = 'config', required=True)
    args = args_parser.parse_args()

    data_load(config_path=args.config)