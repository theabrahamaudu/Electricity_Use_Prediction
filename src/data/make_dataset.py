"""
Module: make_dataset

Description:
    This module provides a class, `makeDataset`, for loading, transforming, 
    and processing time-series data. 
    It offers methods for loading data from text files, converting date codes to datetime objects, 
    grouping data by MeterID, and concatenating multiple data files into a single dataset.

Classes:
    makeDataset

Functions:
    - scanDir(directory: str, extension: str) -> list[str]: 
        Check a specified directory and return a list of files with a specified extension.

    - loadData(directory: str, filename: str | list[str], parse_dates: list[str] | bool = False, 
               index_col: str | int | None = None, sep: str = ' ', names: list[str] | None = ['MeterID', 'codeDateTime', 'kWh']) -> DataFrame: 
        Load data from a file or a list of files with options for parsing dates, specifying column names, and more.

    - saveInterimData(filename: str, dataframe: DataFrame, sep: str = ',', index: bool = True): 
        Save a DataFrame to an interim directory with the given filename and options.

    - code_to_datetime(code: int) -> datetime: 
        Convert a 5-digit integer code to a datetime object.

    - group_by_meter_id(df: DataFrame) -> DataFrame: 
        Group data by MeterID and create a new DataFrame.

    - applyDateTimeTransform(dataframe: DataFrame): 
        Apply the datetime transformation to a DataFrame.

    - loadTransform(skip_transformed: bool = True): 
        Load data from raw files, apply datetime transformation, and save to the interim directory.

    - loadGroup(skip_grouped: bool = True): 
        Load transformed data, group it by MeterID, and save the grouped data.

    - loadConcatenate(skip_concatenated: bool = True): 
        Load grouped data, concatenate it into a single dataset, and save the concatenated data.

Usage:
    if __name__ == "__main__":
        make_data = makeDataset("./data/raw", "./data/interim")
        make_data.loadTransform()
        make_data.loadGroup(skip_grouped=False)
        make_data.loadConcatenate()

"""

import os
import warnings
import pandas as pd
from datetime import datetime, timedelta
from pandas import DataFrame
import numpy as np
from tqdm import tqdm
from multiprocessing import cpu_count
from src.utils.pipeline_log_config import pipeline as logger
# Dask
import dask.dataframe as dd
from dask import delayed, compute
from dask.diagnostics import ProgressBar

# Set the warning filter to 'ignore' to suppress all warnings
warnings.filterwarnings('ignore')


class makeDataset:
    """
    A class for loading, transforming, and processing electricity consumption data.

    This class provides methods to load data from raw files, transform the date and time information,
    group data by MeterID, and concatenate the grouped data into a single dataset.

    Args:
        raw_path (str): The path to the directory containing the raw data files.
        interim_path (str): The path to the directory where interim data will be saved.

    Example:
        make_data = makeDataset(raw_path="./data/raw", interim_path="./data/interim")
        # Initialize the makeDataset object with the specified paths.

    Methods:
        scanDir(directory: str, extension: str) -> list[str]:
        loadData(directory: str, filename: str | list[str], parse_dates: list[str] | bool = False, index_col: str | int | None = None, sep: str = ' ', names: list[str] | None = ['MeterID', 'codeDateTime', 'kWh']) -> DataFrame:
        saveInterimData(filename: str, dataframe: DataFrame, sep: str = ',', index: bool = True) -> None:
        code_to_datetime(code: int) -> datetime:
        group_by_meter_id(df: DataFrame) -> DataFrame:
        applyDateTimeTransform(dataframe: DataFrame) -> DataFrame:
        loadTransform(skip_transformed: bool = True) -> None:
        loadGroup(skip_grouped: bool = True) -> None:
        loadConcatenate(skip_concatenated: bool = True) -> None:

    """

    def __init__(self, raw_path: str, interim_path: str):
        self.raw_path = raw_path
        self.interim_path = interim_path

    @staticmethod
    def scanDir(directory: str, extension: str) -> list[str]:
        """Check specified directory and return list of files with
        specified extension

        Args:
            extension (str): extension type to be searched for e.g. ".txt"

        Returns:
            list: strings of file names with specified extension
        """    
        files: list = []
        for filename in os.listdir(directory):
            if filename.endswith(extension):
                files.append(filename)
        files.sort()
        return files
    

    @staticmethod
    def loadData(directory: str,
                 filename: str | list[str],
                 parse_dates: list[str] | bool = False,
                 index_col: str | int | None = None,
                 sep=' ',
                 names: list[str] | None=['MeterID', 'codeDateTime', 'kWh']) -> DataFrame:
        """
        Load data from files into a Pandas DataFrame.

        Args:
            directory (str): The path to the directory containing the data files.
            filename (str | list[str]): The name of a single data file or a list of data file names.
            parse_dates (list[str] | bool, optional): List of column names to parse as dates, or False to not parse dates.
            index_col (str | int | None, optional): Column to set as the DataFrame index, or None for no index.
            sep (str, optional): Separator used in the data files.
            names (list[str] | None, optional): List of column names or None to infer column names.

        Returns:
            DataFrame: A Pandas DataFrame containing the loaded data.

        Raises:
            Exception: If there is an error loading the data.

        Example:
            data = makeDataset.loadData("data/raw", "example.txt", parse_dates=['DateTime'])
            # Load data from the "example.txt" file in the "data/raw" directory, parsing the 'DateTime' column as dates.
        """
        # if the given filename is a string, load without looping
        if type(filename) == str:
            logger.info(f"Loading {filename}")
            try:
                # if the names of columns in the data is given
                if type(names)==list:
                    dataframe = pd.read_csv(directory+"/"+filename, 
                                            header=None, sep=sep, 
                                            names=names,
                                            parse_dates=parse_dates,
                                            index_col=index_col)
                # if no column names are given
                else:
                    dataframe = pd.read_csv(directory+"/"+filename,
                                            sep=sep,
                                            parse_dates=parse_dates,
                                            index_col=index_col)
                logger.info(f"Loaded {len(dataframe)} rows from {filename}")
            except Exception as e:
                logger.error(f"Failed to load {filename}: {e}")
                raise e

        # if the given filename is a list of files  
        else:
            logger.info(f"Loading files {filename}")
            try:
                # initialize dataframe where all the data will be loaded to
                dataframe = pd.DataFrame(columns=names)

                # initialize counter to verify total number of rows loaded from all files
                total_rows = 0
                
                # if the names of columns to be loaded is given
                if type(names)==list:
                    # loop through each file, load it and add it to the dataframe initialized above
                    for file in filename:
                        # for the first iteration, load directly to the dataframe
                        if len(dataframe)==0:
                            dataframe = pd.read_csv(directory+"/"+file,
                                                    header=None, sep=sep,
                                                    names=names,
                                                    parse_dates=parse_dates,
                                                    index_col=index_col)
                            total_rows += len(dataframe)
                            logger.info(f"Loaded {len(dataframe)} rows from {file}")
                        # for subsequent iterations, load and concatenate to the dataframe
                        else:    
                            dataframe = pd.concat([dataframe, pd.read_csv(directory+"/"+file,
                                                                    header=None, sep=sep,
                                                                    names=names,
                                                                    parse_dates=parse_dates,
                                                                    index_col=index_col)],
                                                                    ignore_index=True)
                            total_rows += len(dataframe)
                            logger.info(f"Loaded {len(dataframe)} rows from {file}")
                # If column names are not specified, do same as above but without specifying the column names 
                else:
                    for file in filename:
                        if len(dataframe)==0:
                            dataframe = pd.read_csv(directory+"/"+file,
                                                    sep=sep,
                                                    parse_dates=parse_dates,
                                                    index_col=index_col)
                            total_rows += len(dataframe)
                            logger.info(f"Loaded {len(dataframe)} rows from {file}")
                        else:    
                            dataframe = pd.concat([dataframe, pd.read_csv(directory+"/"+file,
                                                                          sep=sep,
                                                                          parse_dates=parse_dates,
                                                                          index_col=index_col)],
                                                                          ignore_index=True)
                            total_rows += len(dataframe)
                logger.info(f"Loaded {total_rows} rows from files{filename}")
            except Exception as e:
                logger.error(f"Failed to load files {filename}: {e}")
                raise e

        return dataframe
    
    def saveInterimData(self, filename: str, dataframe: DataFrame, sep = ',', index = True):
        """
        Save data to an interim file in a specified format.

        Args:
            filename (str): The name of the interim file to be saved.
            dataframe (DataFrame): The Pandas DataFrame containing the data to be saved.
            sep (str, optional): Separator used in the data file (default is ',').
            index (bool, optional): Whether to include the index in the saved data (default is True).

        Returns:
            None

        Example:
            makeData = makeDataset("data/raw", "data/interim")
            data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
            makeData.saveInterimData("interim_data.csv", data, sep=';', index=False)
            # Save the 'data' DataFrame as 'interim_data.csv' with a ';' separator and without an index.
        """
        dataframe.to_csv(self.interim_path+"/"+filename, index=index, header=True, sep=sep)
        logger.info(f"Saved {filename} to {self.interim_path}")
    

    @staticmethod
    def code_to_datetime(code: int) -> datetime:
        """
        Convert a 5-digit code to a corresponding datetime object.

        This method takes a 5-digit code, where the first 3 digits represent the day of the year, 
        and the last 2 digits represent the time in half-hour increments, and converts it into
        a datetime object.

        Args:
            code (int): A 5-digit code representing date and time.

        Returns:
            datetime: The datetime object corresponding to the input code.

        Raises:
            ValueError: If the input code is not a 5-digit integer.

        Example:
            dt = makeDataset.code_to_datetime(12000)
            # Convert the code 12000 to a datetime object.
        """

        if len(str(code)) != 5:
            raise ValueError("Input code must be a 5-digit integer.")

        # split the code according to the dataset documentation
        day_code = int(str(code)[:3])
        time_code = int(str(code)[3:5])

        # Calculate the date
        base_date = datetime(2009, 1, 1) # reference date according to documentation
        delta = timedelta(days=day_code)
        target_date = base_date + delta

        # Calculate the time
        hours = (time_code) // 2
        minutes = 30 * (time_code % 2)

        target_time = timedelta(hours=hours, minutes=minutes)

        # Combine the date and time to create the datetime object
        result_datetime = target_date + target_time

        return result_datetime
    
    @staticmethod
    def group_by_meter_id(df: DataFrame) -> DataFrame:
        """
        Group data by MeterID and create a new DataFrame.

        This method groups data by the 'MeterID' column and creates a new DataFrame 
        where each column represents a unique MeterID, and each row represents a unique
        'DateTime'. Missing values are filled with NaN.

        Args:
            df (DataFrame): The input DataFrame containing data with 'MeterID' and 'DateTime' columns.

        Returns:
            DataFrame: A new DataFrame where each column represents a unique MeterID, and each row 
            represents a unique 'DateTime' with corresponding 'kWh' values.

        Raises:
            Exception: If any error occurs during the grouping process.

        Example:
            data = makeDataset.group_by_meter_id(input_dataframe)
            # Group data by 'MeterID' and create a new DataFrame.
        """

        logger.info("Grouping data by MeterID")
        try:
            # Get the number of CPU cores and use 75%
            cores = int(cpu_count() * 0.75)
            # Create a Dask DataFrame from the Pandas DataFrame
            dask_df = dd.from_pandas(df, npartitions=cores)  # You can adjust the number of partitions as needed

            # Get unique datetimes
            unique_datetimes = dask_df['DateTime'].unique().compute()
            meter_ids = dask_df['MeterID'].unique().compute()

            # Scatter the Dask DataFrame ahead of time
            dask_df = dask_df.repartition(npartitions=cores)  # You can adjust the number of partitions as needed

            # Create a list of futures for the results
            results = []

            # Function to process a single meter ID. Wraped with Dask `delayed` function
            @delayed
            def process_meter(meter_id):
                # Get the data for the meter ID 
                meter_data = dask_df[dask_df['MeterID'] == meter_id]

                # Group the data by DateTime and kWh columns and return the result
                result = meter_data.groupby('DateTime')['kWh'].first().compute()
                result = result.reindex(unique_datetimes, fill_value=np.nan)
                return result

            # Use Dask's delayed function for parallel processing
            with ProgressBar():
                # Initialize all the jobs in a list
                futures = [process_meter(meter_id) for meter_id in meter_ids]
                # Compute jobs in parallel
                results = compute(*futures, num_workers=cores)
            
            # Combine the results into a Pandas DataFrame
            new_df = pd.concat(results, axis=1)
            new_df.columns = meter_ids
            new_df.index = unique_datetimes
            new_df = new_df.sort_index()
            logger.info("Data grouped by MeterID")
        except Exception as e:
            logger.error(f"Failed to group data by MeterID: {e}")
            raise e

        return new_df

    
    def applyDateTimeTransform(self, dataframe: DataFrame) -> DataFrame:
        """
        Apply a datetime transformation to the DataFrame.

        This method takes a DataFrame with a 'codeDateTime' column and applies a transformation
        to convert it into a 'DateTime' column. The progress is tracked using tqdm.

        Args:
            dataframe (DataFrame): The input DataFrame with a 'codeDateTime' column.

        Returns:
            DataFrame: A new DataFrame with the 'codeDateTime' column transformed into a 'DateTime' column.

        Raises:
            Exception: If any error occurs during the transformation process.

        Example:
            transformed_data = makeDataset.applyDateTimeTransform(input_dataframe)
            # Apply the datetime transformation to the DataFrame.
        """

        # Use tqdm to track progress
        tqdm.pandas(desc="Converting", unit=" rows")

        logger.info("Applying datetime transform")
        try:
            # Apply the conversion function to the 'codeDateTime' column
            dataframe['DateTime'] = dataframe['codeDateTime'].progress_apply(self.code_to_datetime)
            # Drop the 'codeDateTime' column
            dataframe.drop(columns=['codeDateTime'], inplace=True)
            logger.info("Datetime transformed")
        except Exception as e:
            logger.error(f"Failed to apply datetime transform: {e}")
            raise e

        return dataframe
    
    def loadTransform(self, skip_transformed = True):
        """
        Load data from raw files and transform the 'codeDateTime' column into 'DateTime'.

        This method scans the 'raw_path' directory for '.txt' files, and if not already transformed,
        applies a datetime transformation to convert the 'codeDateTime' column into a 'DateTime' column.
        The transformed data is saved in the 'interim_path' directory.

        Args:
            skip_transformed (bool, optional): If True, skip already transformed files. Defaults to True.

        Raises:
            Exception: If any error occurs during loading, transformation, or saving.

        Example:
            make_data = makeDataset("./data/raw", "./data/interim")
            make_data.loadTransform()
            # Load raw data, apply datetime transformation, and save transformed data.
        """

        # Load Data and Transform DateTime
        # Get list of all the raw files in the raw files path
        raw_files = self.scanDir(self.raw_path, ".txt")

        # Get the list of all transformed files in the interim files path
        transformed_files = self.scanDir(self.interim_path, ".txt")

        logger.info(f"Loading and transforming {len(raw_files)} files")
        try:
            # loop through each raw file
            for i, file in enumerate(raw_files):
                # Skip the file if its tranformed version already exists and `skip_transformed` is True
                if f"transformed_{raw_files[i][:-4]}.txt" in transformed_files and skip_transformed:
                    logger.info(f"Skipping {file}, already transformed")
                # if not, load the data, call the transformation method on the data and save the outcome    
                else:
                    data = self.loadData(self.raw_path, file)
                    transformed_data = self.applyDateTimeTransform(data)
                    self.saveInterimData(f"transformed_{raw_files[i][:-4]}.txt",
                                         transformed_data, index=False)
            logger.info(f"Loaded and transformed {len(raw_files)} files")
        except Exception as e:
            logger.error(f"Failed to load and transform files: {e}")
            raise e

    def loadGroup(self, skip_grouped = True):
        """
        Load transformed data and group it by 'MeterID'.

        This method scans the 'interim_path' directory for transformed '.txt' files, loads the data, and
        groups it by 'MeterID'. The grouped data is saved in the 'interim_path' directory.

        Args:
            skip_grouped (bool, optional): If True, skip already grouped files. Defaults to True.

        Raises:
            Exception: If any error occurs during loading, grouping, or saving.

        Example:
            make_data = makeDataset("./data/raw", "./data/interim")
            make_data.loadGroup()
            # Load transformed data, group by 'MeterID', and save grouped data.
        """

        # Load Data and Group by MeterID
        # Get lists of the data in different states
        raw_files = self.scanDir(self.raw_path, ".txt")
        transformed_files = self.scanDir(self.interim_path, ".txt")
        grouped_files = self.scanDir(self.interim_path, ".csv")

        logger.info(f"Loading and grouping {len(raw_files)} files")
        try:
            for i, file in enumerate(transformed_files):
                # check if the file has already been grouped and skip it if `skip_grouped` is True
                if f"grouped_data_{raw_files[i][:-4]}.csv" in grouped_files and skip_grouped:
                    logger.info(f"Skipping {file}, already grouped")
                # load the data and apply the grouping method, then save it to the interim dir
                else:
                    transformed_data = self.loadData(self.interim_path,
                                                        file, 
                                                        names=None,
                                                        sep=',',
                                                        parse_dates=['DateTime'])
                    grouped_data = self.group_by_meter_id(transformed_data)
                    self.saveInterimData(f"grouped_data_{raw_files[i][:-4]}.csv", grouped_data)
            logger.info(f"Loaded and grouped {len(raw_files)} files")
        except Exception as e:
            logger.error(f"Failed to load and group files: {e}")
            raise e

    def loadConcatenate(self, skip_concatenated = True):
        """
        Load grouped data and concatenate it into a single dataset.

        This method scans the 'interim_path' directory for grouped '.csv' files, loads the data,
        and concatenates it into a single dataset. The concatenated data is saved in the 'interim_path' directory.

        Args:
            skip_concatenated (bool, optional): If True, skip concatenation if the result file already exists. Defaults to True.

        Raises:
            Exception: If any error occurs during loading, concatenation, or saving.

        Example:
            make_data = makeDataset("./data/raw", "./data/interim")
            make_data.loadConcatenate()
            # Load grouped data, concatenate it, and save the concatenated dataset.
        """

        # Load Data and Concatenate
        # Get list of the different stages of the data
        files = self.scanDir(self.interim_path, ".csv")
        grouped_files = [x for x in files if "grouped_data_" in x]
        concatenated_file = [x for x in files if "concatenated_data.csv" in x]

        logger.info(f"Loading and concatenating {len(grouped_files)} files")
        try:
            # skip if the files have already been concatentated and `skip_concatenated` is true
            if f"concatenated_data.csv" in concatenated_file and skip_concatenated:
                logger.info(f"Skipping concatenation, `concatenated_data.csv` already exists")
            else:
                # load load the dataframes into a list and concatente them
                grouped_dataframes = []
                for i, file in enumerate(grouped_files):
                    grouped_dataframes.append(self.loadData(self.interim_path,
                                                            file,
                                                            names=None, sep=',',
                                                            parse_dates=['DateTime'],
                                                            index_col=0))
                concatenated_data = pd.concat(grouped_dataframes, axis=1)
                concatenated_data.sort_index(inplace=True)
                self.saveInterimData(f"concatenated_data.csv", concatenated_data)
        except Exception as e:
            logger.error(f"Failed to load and concatenate files: {e}")
            raise e  

    

# if __name__ == "__main__":
#     make_data = makeDataset("./data/raw", "./data/interim")
#     make_data.loadTransform()
#     make_data.loadGroup(skip_grouped=False)
#     make_data.loadConcatenate()


    


