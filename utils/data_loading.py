import numpy as np
import pandas as pd
import csv
import os

from pathlib import Path
from sklearn.preprocessing import normalize, StandardScaler


class Dataset:
    @staticmethod
    def preprocess(data, mode=None):
        """
        :param data: Dữ liệu chưa qua xử lý (kiểu dữ liệu có thể pd, np, list, v.v.v)
        :param mode: Liên quan đến vấn đề Handle Missing
        :return: Dữ liệu đã qua xử lý
        """

        if isinstance(data, pd.core.frame.DataFrame):
            data = data.values

        # Replace value 0 -> NaN
        data[:, 1:6] = np.where(data[:, 1:6] == 0, np.nan, data[:, 1:6])
        if mode == 'mean':
            mean_data = np.nanmean(data[:, 1:6], axis=0)
            for idx, mean in enumerate(mean_data):
                data[:, idx+1] = np.where(np.isnan(data[:, idx + 1]), mean, data[:, idx + 1])
                print(data[:, idx+1])
        elif mode == 'median':
            median_data = np.nanmedian(data[:, 1:6], axis=0)
            for idx, median in enumerate(median_data):
                data[:, idx+1] = np.where(np.isnan(data[:, idx + 1]), median, data[:, idx + 1])
        elif mode == 'remove':
            data = data[~np.isnan(data).any(axis=1)]
        elif mode is None:
            pass

        # Chuẩn hóa dữ liệu
        stx = StandardScaler()
        data[:, :-1] = stx.fit_transform(data[:, :-1])
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


if __name__ == '__main__':
    Data = Dataset().load('D:/Machine_Learning/Pima-Indians-Diabetes-Dataset/pima-indians-diabetes.csv')
    data = Dataset().preprocess(Data, mode='mean')