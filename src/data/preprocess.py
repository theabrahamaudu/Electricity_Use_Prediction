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
        logger.info("Loading filter codes data")
        try:
            label_data = pd.read_excel(f'{self.raw_path}/CER_Electricity_Documentation/SME and Residential allocations.xlsx',
                            sheet_name='Sheet1',
                            usecols=['ID', 'Code', 'Residential - Tariff allocation', 'Residential - stimulus allocation', 'SME allocation']
                            )
        except Exception as e:
            logger.error(f"Failed to load filter codes data: {e}")
            raise e
        
        logger.info("Filtering out control meters")
        try:
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
        except Exception as e:
            logger.error(f"Failed to filter data: {e}")
            raise e

    @staticmethod
    def removeMissingValues(dataframe: DataFrame, threshold = 0.20) -> DataFrame:
        logger.info(f"Removing meter records with missing values count greater than {threshold*100}%")
        try:
            cols = []
            for i in dataframe.columns:
                if dataframe[i].isna().sum() > len(dataframe)*threshold:
                    cols.append(i)
            dataframe = dataframe.drop(columns=cols)
            return dataframe
        except Exception as e:
            logger.error(f"Failed to remove missing values: {e}")
            raise e
    
    @staticmethod
    def sumAllRows(dataframe: pd.DataFrame) -> DataFrame:
        logger.info("Summing meter records by timestamp")
        try:
            # Sum all rows along axis 1
            row_sums = dataframe.sum(axis=1)

            # Create a new DataFrame with only the row sums
            result_df = pd.DataFrame({'kWh': row_sums})
        
            return result_df
        except Exception as e:
            logger.error(f"Failed to sum all meter records: {e}")
            raise e
    
    def scaleData(self, dataframe: DataFrame, train=True) -> DataFrame:
        scaler_path = f'{self.processed_path}/scaler.pkl'
        if train:
            logger.info("Scaling train data")
            try:
                # Fit scaler
                scaler = MinMaxScaler()
                scaler = scaler.fit(dataframe)

                # Save scaler
                joblib.dump(scaler, scaler_path)

                # Load scaler and transform data
                scaler = joblib.load(scaler_path)
                scaled_data = scaler.transform(dataframe)
                return scaled_data
            except Exception as e:
                logger.error(f"Failed to scale data: {e}")
                raise e
        else:
            logger.info("Scaling inference data")
            try:
                # Load scaler and transform data
                scaler = joblib.load(scaler_path)
                scaled_data = scaler.transform(dataframe)
                return scaled_data
            except Exception as e:
                logger.error(f"Failed to scale data: {e}")
                raise e
        

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
            logger.info("Structuring training data")
            dataX, dataY = [], []
            for i in range(self.n_past, len(dataset) - self.n_future +1):
                dataX.append(dataset[i - self.n_past:i, 0:dataset.shape[1]])
                dataY.append(dataset[i + self.n_future - 1:i + self.n_future, -1])
                
            dataX, dataY = np.array(dataX), np.array(dataY)
            
            return dataX, dataY
        else:
            logger.info("Structuring inference data")
            dataX = []
            for i in range(self.n_past, len(dataset) - self.n_future +1):
                dataX.append(dataset[i - self.n_past:i, 0:dataset.shape[1]])
                
            dataX = np.array(dataX)
            
            return dataX
    

    @staticmethod
    def splitData(X: ndarray, y: ndarray,
                  test_size: float = 0.2, 
                  ) -> tuple[ndarray, ndarray, ndarray, ndarray]:
        logger.info("Splitting data into train and test sets")
        # Time series train test split
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=test_size,
                                                            shuffle=False)
        return X_train, X_test, y_train, y_test
    

    def saveArray(self, filename: str, array: ndarray):
            np.save(f'{self.processed_path}/{filename}', array)
            logger.info(f"Saved {filename} to {self.processed_path}")


    def trainPreprocess(self, filename: str = "concatenated_data.csv", train=True, skip_transformed=True, skip_grouped=True, skip_concatenated=True):
        self.loadTransform(skip_transformed=skip_transformed)
        self.loadGroup(skip_grouped=skip_grouped)
        self.loadConcatenate(skip_concatenated=skip_concatenated)
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

    
    def inferencePreprocess(self, filename: str, train_scale=False, train_structure=True):
        data = self.loadData(self.interim_path,
                            filename,
                            names=None, sep=',',
                            parse_dates=['DateTime'],
                            index_col=0)
        data = self.filterData(data)
        data = self.removeMissingValues(data)
        data = self.sumAllRows(data)
        data = self.scaleData(data, train=train_scale)
        if train_structure:
            dataX, dataY = self.structure_data(data, train=train_structure)
            return dataX, dataY
        else:
            dataX = self.structure_data(data, train=train_structure)
            return dataX