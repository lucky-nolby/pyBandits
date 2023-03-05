import pandas as pd
import numpy as np

def load_dataset(filename):
    path='./data/'+filename
    dataset = pd.read_csv(path)
    return dataset


