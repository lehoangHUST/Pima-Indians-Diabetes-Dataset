import numpy as np
import pandas as pd
import csv
import os

from pathlib import Path


class Dataset:
    @staticmethod
    def preprocess(data):
        if isinstance(data, pd.core.frame.DataFrame):
            data = data.values

        return data

    @staticmethod
    def load(file: str):
        if os.path.isfile(file):
            ext = file.split('/')[-1].split('.')[-1]
            print("Loading dataset ...")
            if ext == 'csv':
                df = pd.read_csv(file, header=None)
            elif ext == 'xlsx':
                df = pd.read_excel(file, header=None)
            print("Loaded dataset ...")
            return df
        else:
            raise FileNotFoundError
