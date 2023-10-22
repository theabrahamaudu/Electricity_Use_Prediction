# Regular imports
import os
import random
import numpy as np
import pandas as pd
import json
import time
import pickle
import joblib
from numpy import ndarray
from pandas import DataFrame
import matplotlib.pyplot as plt
from src.data.make_dataset import makeDataset
from src.utils.pipeline_log_config import pipeline as logger

# Metrics and 
import math
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler


class preprocessPipeline(makeDataset):
    def __init__(self, raw_path: str,
                 interim_path: str,
                 processed_path: str,
                 n_future: int = 1,
                 n_past: int = 48*30):
        super().__init__(raw_path, interim_path)
        self.processed_path = processed_path
        self.n_future = n_future
        self.n_past = n_past


    def filterData(self, dataframe: DataFrame) -> DataFrame:
        # Load filter codes data
        label_data = pd.read_excel(f'{self.raw_path}/CER_Electricity_Documentation/SME and Residential allocations.xlsx',
                           sheet_name='Sheet1',
                           usecols=['ID', 'Code', 'Residential - Tariff allocation', 'Residential - stimulus allocation', 'SME allocation']
                        )
        
        # Get control meters
        control_meters = []
        for i in range(len(label_data)):
            if label_data['Residential - Tariff allocation'][i] == 'E' or\
            label_data['Residential - stimulus allocation'][i] == 'E' or\
            label_data['SME allocation'][i] == 'C':
                control_meters.append(str(label_data['ID'][i]))

        # Filter out control Meters from concatenated data
        filtered_data = dataframe.drop(columns=control_meters)
        return filtered_data

    @staticmethod
    def removeMissingValues(dataframe: DataFrame, threshold = 0.20) -> DataFrame:
        cols = []
        for i in dataframe.columns:
            if dataframe[i].isna().sum() > len(dataframe)*threshold:
                cols.append(i)
        dataframe = dataframe.drop(columns=cols)
        return dataframe
    
    @staticmethod
    def sumAllRows(dataframe: pd.DataFrame) -> DataFrame:

        # Sum all rows along axis 1
        row_sums = dataframe.sum(axis=1)

        # Create a new DataFrame with only the row sums
        result_df = pd.DataFrame({'kWh': row_sums})
    
        return result_df
    
    def scaleData(self, dataframe: DataFrame, train=True) -> DataFrame:

        if train:
            # Fit scaler
            scaler = MinMaxScaler()
            scaler = scaler.fit(dataframe)

            # Save scaler
            joblib.dump(scaler, f'scaler.pkl')

            # Load scaler and transform data
            scaler = joblib.load(f'scaler.pkl')
            scaled_data = scaler.transform(dataframe)
            return scaled_data
        else:
            # Load scaler and transform data
            scaler = joblib.load(f'scaler.pkl')
            scaled_data = scaler.transform(dataframe)
            return scaled_data
        

    def structure_data(self, dataset: DataFrame, train=True) -> tuple[ndarray, ndarray] | ndarray:
        """
        This function is used to split the data into features and targets, structuring the training
        data to look like the number of days backward the model should look, and the number of days forward
        the model should predict.
        
        Returns:
            dataX: features
            dataY: targets
        """


        if train:
            dataX, dataY = [], []
            for i in range(self.n_past, len(dataset) - self.n_future +1):
                dataX.append(dataset[i - self.n_past:i, 0:dataset.shape[1]])
                dataY.append(dataset[i + self.n_future - 1:i + self.n_future, -1])
                
            dataX, dataY = np.array(dataX), np.array(dataY)
            
            return dataX, dataY
        else:
            dataX = []
            for i in range(self.n_past, len(dataset) - self.n_future +1):
                dataX.append(dataset[i - self.n_past:i, 0:dataset.shape[1]])
                
            dataX = np.array(dataX)
            
            return dataX
    

    @staticmethod
    def splitData(X: ndarray, y: ndarray,
                  test_size: float = 0.2, 
                  ) -> tuple[ndarray, ndarray, ndarray, ndarray]:
        # Time series train test split
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=test_size,
                                                            shuffle=False)
        return X_train, X_test, y_train, y_test
    

    def saveArray(self, filename: str, array: ndarray):
            np.save(f'{self.processed_path}/{filename}', array)
            logger.info(f"Saved {filename} to {self.processed_path}")


    def trainPreprocess(self, filename: str = "concatenated_data.csv", train=True):
        self.loadTransform()
        self.loadGroup()
        self.loadConcatenate()
        data = self.loadData(self.interim_path,
                            filename,
                            names=None, sep=',',
                            parse_dates=['DateTime'],
                            index_col=0)
        data = self.filterData(data)
        data = self.removeMissingValues(data)
        data = self.sumAllRows(data)
        data = self.scaleData(data, train=train)
        dataX, dataY = self.structure_data(data, train=train)
        X_train, X_test, y_train, y_test = self.splitData(dataX, dataY)

        # Save train and test data
        self.saveArray("X_train.npy", X_train)
        self.saveArray("X_test.npy", X_test)
        self.saveArray("y_train.npy", y_train)
        self.saveArray("y_test.npy", y_test)

    
    def inferencePreprocess(self, filename: str, train=False):
        data = self.loadData(self.interim_path,
                            filename,
                            names=None, sep=',',
                            parse_dates=['DateTime'],
                            index_col=0)
        data = self.filterData(data)
        data = self.removeMissingValues(data)
        data = self.sumAllRows(data)
        data = self.scaleData(data, train=train)
        dataX = self.structure_data(data, train=train)
        return dataX