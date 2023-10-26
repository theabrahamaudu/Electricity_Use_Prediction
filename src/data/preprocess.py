"""
preprocess Module

This module provides a preprocessing pipeline for electricity consumption data. 
It extends the 'makeDataset' class for loading, filtering, and preprocessing electricity consumption data. 
The pipeline includes methods for filtering control meters, removing missing values, 
summing meter records, scaling the data, and structuring it for training or inference.


Classes:
    preprocessPipeline(makeDataset)
    
    Attributes:
    - raw_path (str): Path to raw data directory.
    - interim_path (str): Path to interim data directory.
    - processed_path (str): Path to processed data directory.
    - n_future (int, optional): Number of future time steps to predict. Defaults to 1.
    - n_past (int, optional): Number of past time steps to consider for prediction. Defaults to 48 * 30.

    Methods:
    - filterData(dataframe: DataFrame) -> DataFrame
    - removeMissingValues(dataframe: DataFrame, threshold: float = 0.20) -> DataFrame
    - sumAllRows(dataframe: pd.DataFrame) -> DataFrame
    - scaleData(dataframe: DataFrame, train: bool = True) -> DataFrame
    - structure_data(dataset: DataFrame, train: bool = True) -> Union[Tuple[ndarray, ndarray], ndarray]
    - splitData(X: ndarray, y: ndarray, test_size: float = 0.2) -> Tuple[ndarray, ndarray, ndarray, ndarray]
    - saveArray(filename: str, array: ndarray)
    - trainPreprocess(filename: str = "concatenated_data.csv", train: bool = True, skip_transformed: bool = True, skip_grouped: bool = True, skip_concatenated: bool = True)
    - inferencePreprocess(filename: str, train_scale: bool = False, train_structure: bool = True) -> Union[Tuple[ndarray, ndarray], ndarray]
"""


# Regular imports
import numpy as np
import pandas as pd
import joblib
from numpy import ndarray
from pandas import DataFrame
from src.data.make_dataset import makeDataset
from src.utils.pipeline_log_config import pipeline as logger

# Splitting and scaling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


class preprocessPipeline(makeDataset):
    """
    A preprocessing pipeline for electricity consumption data.
    
    This class extends the 'makeDataset' class for loading, filtering, and preprocessing
    electricity consumption data. It provides methods for filtering control meters,
    removing missing values, summing meter records, scaling the data, and structuring
    it for training or inference.
    
    Args:
        raw_path (str): Path to raw data directory.
        interim_path (str): Path to interim data directory.
        processed_path (str): Path to processed data directory.
        n_future (int, optional): Number of future time steps to predict. Defaults to 1.
        n_past (int, optional): Number of past time steps to consider for prediction. Defaults to 48 * 30.

    Attributes:
        raw_path (str): Path to raw data directory.
        interim_path (str): Path to interim data directory.
        processed_path (str): Path to processed data directory.
        n_future (int): Number of future time steps to predict.
        n_past (int): Number of past time steps to consider for prediction.
    
    Methods:
        - filterData(dataframe: DataFrame) -> DataFrame
        - removeMissingValues(dataframe: DataFrame, threshold: float = 0.20) -> DataFrame
        - sumAllRows(dataframe: pd.DataFrame) -> DataFrame
        - scaleData(dataframe: DataFrame, train: bool = True) -> DataFrame
        - structure_data(dataset: DataFrame, train: bool = True) -> Union[Tuple[ndarray, ndarray], ndarray]
        - splitData(X: ndarray, y: ndarray, test_size: float = 0.2) -> Tuple[ndarray, ndarray, ndarray, ndarray]
        - saveArray(filename: str, array: ndarray)
        - trainPreprocess(filename: str = "concatenated_data.csv", train: bool = True, skip_transformed: bool = True, 
            skip_grouped: bool = True, skip_concatenated: bool = True)
        - inferencePreprocess(filename: str, train_scale: bool = False, train_structure: bool = True) -> Union[Tuple[ndarray, ndarray], ndarray]

    """

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
        """
        Filter out control meters from the input DataFrame.

        This method loads filter codes data from an Excel file and checks for control meters based on specific conditions.
        It filters out the control meters from the input DataFrame and returns the filtered DataFrame.

        Args:
            dataframe (DataFrame): The input DataFrame containing electricity consumption data.

        Returns:
            DataFrame: A new DataFrame with control meters removed.

        Raises:
            Exception: If any error occurs during the filtering process.

        Example:
            make_data = preprocessPipeline(raw_path, interim_path, processed_path)
            data = make_data.loadData(dataframe)
            filtered_data = make_data.filterData(data)
            # Filter out control meters from the input data and get the filtered DataFrame.
        """

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
        """
        Remove meter records with missing values from the DataFrame.

        This method removes rows (meter records) from the input DataFrame if the number of missing values in a row
        exceeds a specified threshold. It returns a new DataFrame with missing records removed.

        Args:
            dataframe (DataFrame): The input DataFrame containing electricity consumption data.
            threshold (float, optional): The threshold for the maximum allowable proportion of missing values in a row. 
                Defaults to 0.20 (20%).

        Returns:
            DataFrame: A new DataFrame with rows containing excessive missing values removed.

        Raises:
            Exception: If any error occurs during the removal process.

        Example:
            make_data = preprocessPipeline(raw_path, interim_path, processed_path)
            data = make_data.loadData(dataframe)
            threshold = 0.10  # Set the threshold for missing values to 10%
            clean_data = make_data.removeMissingValues(data, threshold)
            # Remove rows with missing values exceeding the specified threshold and get the cleaned DataFrame.
        """
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
        """
        Sum the meter records by timestamp and create a new DataFrame.

        This method sums the meter records for each timestamp by calculating the row sums of the input DataFrame
        and creates a new DataFrame with the summed values.

        Args:
            dataframe (pd.DataFrame): The input DataFrame containing electricity consumption data.

        Returns:
            DataFrame: A new DataFrame with row sums for each timestamp.

        Raises:
            Exception: If any error occurs during the summation process.

        Example:
            make_data = preprocessPipeline(raw_path, interim_path, processed_path)
            data = make_data.loadData(dataframe)
            summed_data = make_data.sumAllRows(data)
            # Sum the meter records by timestamp and obtain a new DataFrame with the summed values.
        """

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
        """
        Scale the input data using Min-Max scaling.

        This method scales the input DataFrame using Min-Max scaling, which scales the data to a specific range,
        typically [0, 1]. It allows the option to fit and transform the data for training or just transform it
        during inference. The scaler is saved to a file for later use during inference if used for training.

        Args:
            dataframe (DataFrame): The input DataFrame containing electricity consumption data.
            train (bool, optional): Whether to fit the scaler to the data during training (default is True).

        Returns:
            DataFrame: The scaled data.

        Raises:
            Exception: If any error occurs during the scaling process.

        Example:
            make_data = preprocessPipeline(raw_path, interim_path, processed_path)
            data = make_data.loadData(dataframe)
            scaled_data = make_data.scaleData(data)
            # Scale the input data using Min-Max scaling and obtain the scaled data.
        """
        
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
        Structure the input data for training or inference.

        This method structures the input DataFrame to create feature (dataX) and target (dataY) datasets for training
        a predictive model. The data can be structured differently based on whether it's used for training or inference.

        Args:
            dataset (DataFrame): The input DataFrame containing electricity consumption data.
            train (bool, optional): Whether the data should be structured for training (default is True).

        Returns:
            Union[tuple[ndarray, ndarray], ndarray]: A tuple containing feature and target arrays for training, or just feature
            array for inference.

        Raises:
            Exception: If any error occurs during the data structuring process.

        Example:
            make_data = preprocessPipeline(raw_path, interim_path, processed_path)
            data = make_data.loadData(dataset)
            # Structure data for training
            dataX, dataY = make_data.structure_data(data, train=True)
            # Structure data for inference
            dataX_inference = make_data.structure_data(data, train=False)
            # Perform data structuring for training or inference based on the use case.
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
        """
        Split the data into training and testing sets.

        This method performs a time series-based split of the feature (X) and target (y) arrays into training and testing sets.

        Args:
            X (ndarray): Feature data.
            y (ndarray): Target data.
            test_size (float, optional): The proportion of data to include in the testing set (default is 0.2).

        Returns:
            tuple[ndarray, ndarray, ndarray, ndarray]: A tuple containing the following arrays:
            - X_train: Features for training.
            - X_test: Features for testing.
            - y_train: Targets for training.
            - y_test: Targets for testing.

        Raises:
            Exception: If any error occurs during the data splitting process.

        Example:
            from preprocess import preprocessPipeline

            # Load your feature and target data (X and y)

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = preprocessPipeline.splitData(X, y, test_size=0.2)

            # Perform further data processing or modeling using the training and testing sets.
        """
        logger.info("Splitting data into train and test sets")
        # Time series train test split
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=test_size,
                                                            shuffle=False)
        return X_train, X_test, y_train, y_test
    

    def saveArray(self, filename: str, array: ndarray):
        """
        Save a NumPy array to a specified file location.

        This method allows you to save a NumPy array to a file in a specific directory, typically used for saving processed data or model outputs.

        Args:
            filename (str): The name of the file where the array will be saved.
            array (ndarray): The NumPy array to be saved.

        Returns:
            None

        Example:
            from preprocessPipeline import preprocessPipeline
            import numpy as np

            # Create or obtain a NumPy array (e.g., array_data)
            array_data = np.array([1, 2, 3, 4, 5])

            # Specify the filename and save the array
            filename = "my_array.npy"
            processor = preprocessPipeline(raw_path, interim_path, processed_path)
            processor.saveArray(filename, array_data)

            # The array will be saved in the specified directory with the given filename.

        """
        np.save(f'{self.processed_path}/{filename}', array)
        logger.info(f"Saved {filename} to {self.processed_path}")


    def trainPreprocess(self, filename: str = "concatenated_data.csv", train=True, skip_transformed=True, skip_grouped=True, skip_concatenated=True):
        """
        Preprocess and prepare data for training.

        This method is used to preprocess the data for training a model. It includes loading and transforming data, 
        filtering control meters, removing missing values, summing meter records, scaling the data, 
        and splitting it into train and test sets. The processed data is then saved to separate files.

        Args:
            filename (str, optional): The name of the file containing raw data. Defaults to "concatenated_data.csv".
            train (bool, optional): If True, perform training data preprocessing. If False, preprocess data for inference. Defaults to True.
            skip_transformed (bool, optional): If True, skip loading transformed data. Defaults to True.
            skip_grouped (bool, optional): If True, skip loading grouped data. Defaults to True.
            skip_concatenated (bool, optional): If True, skip loading concatenated data. Defaults to True.

        Returns:
            None

        Example:
            from preprocessPipeline import preprocessPipeline

            # Initialize the preprocessing pipeline
            processor = preprocessPipeline(raw_path, interim_path, processed_path)

            # Perform data preprocessing for training (optional skip parameters can be used)
            processor.trainPreprocess(filename="my_data.csv", train=True, skip_transformed=True, skip_grouped=True, skip_concatenated=False)

            # The preprocessed data is saved in the specified directory as "X_train.npy", "X_test.npy", "y_train.npy", and "y_test.npy".

        """

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
        """
        Preprocess and prepare data for inference.

        This method is used to preprocess the data for making predictions or inferences with a trained model.
        It includes loading data, filtering control meters, removing missing values, summing meter records,
        and scaling the data. The data can be structured for inference if needed.

        Args:
            filename (str): The name of the file containing the data to be preprocessed.
            train_scale (bool, optional): If True, set train parameter to true in `scaleData`. If False, set it to false. Defaults to False.
            train_structure (bool, optional): If True, set train parameter to true in `structure_data`. If False, set it to false. Defaults to True.

        Returns:
            Union[tuple[numpy.ndarray, numpy.ndarray], numpy.ndarray]: Depending on the train_structure parameter, this method returns either structured (dataX, dataY) or unstructured dataX for making inferences.

        Example:
            from preprocessPipeline import preprocessPipeline

            # Initialize the preprocessing pipeline
            processor = preprocessPipeline(raw_path, interim_path, processed_path)

            # Preprocess data for inference, optionally specifying whether to structure the data
            structured_data = processor.inferencePreprocess(filename="my_data.csv", train_scale=False, train_structure=True)

            # The structured_data variable contains preprocessed data for making inferences.

        """
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