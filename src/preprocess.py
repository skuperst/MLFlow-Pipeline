import pandas as pd
import sys
import yaml
import os

## Load parameters from param.yaml
curr_dir = os.path.dirname(os.path.abspath(__file__)) 
params=yaml.safe_load(open(os.path.join(curr_dir, '..', "params.yaml")))['preprocess']

def preprocess(input_path,output_path):

    # Load the raw input data
    data = pd.read_csv(input_path)
    
    # Ceate the directory to save the preprocessed file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Save the preprocessed data
    data.to_csv(output_path, index=False)
    
    print(f"Preprocesses data saved to {output_path}")

if __name__ == "__main__":
    preprocess(input_path = params["input"], output_path = params["output"])