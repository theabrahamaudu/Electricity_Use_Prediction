import os
import pandas as pd
from datetime import datetime, timedelta
from pandas import DataFrame
import numpy as np
from tqdm import tqdm
from src.utils.pipeline_log_config import pipeline as logger

class makeDataset:
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
    def loadData(directory: str, filename: str | list[str], sep=' ', names: list[str] | None=['MeterID', 'codeDateTime', 'kWh']) -> DataFrame:
        if type(filename) == str:
            logger.info(f"Loading {filename}")
            try:
                if type(names)==list[str]:
                    dataframe = pd.read_csv(directory+"/"+filename, 
                                            header=None, sep=sep, 
                                            names=names)
                else:
                    dataframe = pd.read_csv(directory+"/"+filename, sep=sep)
                logger.info(f"Loaded {len(dataframe)} rows from {filename}")
            except Exception as e:
                logger.error(f"Failed to load {filename}: {e}")
                raise e

            
        else:
            logger.info(f"Loading files {filename}")
            try:
                dataframe = pd.DataFrame(columns=names)
                total_rows = 0
                if type(names)==list[str]:
                    for file in filename:
                        if len(dataframe)==0:
                            dataframe = pd.read_csv(directory+"/"+file,
                                                    header=None, sep=sep,
                                                    names=names)
                            total_rows += len(dataframe)
                            logger.info(f"Loaded {len(dataframe)} rows from {file}")
                        else:    
                            dataframe = pd.concat([dataframe, pd.read_csv(directory+"/"+file,
                                                                    header=None, sep=sep,
                                                                    names=names)],
                                                                    ignore_index=True)
                            total_rows += len(dataframe)
                            logger.info(f"Loaded {len(dataframe)} rows from {file}")
                else:
                    for file in filename:
                        if len(dataframe)==0:
                            dataframe = pd.read_csv(directory+"/"+file, sep=sep)
                            total_rows += len(dataframe)
                            logger.info(f"Loaded {len(dataframe)} rows from {file}")
                        else:    
                            dataframe = pd.concat([dataframe, pd.read_csv(directory+"/"+file, sep=sep)],
                                                 ignore_index=True)
                            total_rows += len(dataframe)
                logger.info(f"Loaded {total_rows} rows from files{filename}")
            except Exception as e:
                logger.error(f"Failed to load files {filename}: {e}")
                raise e

        return dataframe
    
    def saveInterimData(self, filename: str, dataframe: DataFrame, sep = ','):
        dataframe.to_csv(self.interim_path+"/"+filename, index=True, header=True, sep=sep)
        logger.info(f"Saved {filename} to {self.interim_path}")
    

    @staticmethod
    def code_to_datetime(code: int):
        if len(str(code)) != 5:
            raise ValueError("Input code must be a 5-digit integer.")

        day_code = int(str(code)[:3])
        time_code = int(str(code)[3:5])

        # Calculate the date
        base_date = datetime(2009, 1, 1)
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
    def group_by_meter_id(df: DataFrame):

        logger.info("Grouping data by MeterID")
        try:
            # Initialize an empty DataFrame with 'DateTime' as the index
            unique_datetimes = df['DateTime'].unique()
            new_df = pd.DataFrame(index=unique_datetimes)

            # Iterate through MeterIDs and populate the new DataFrame
            meter_ids = df['MeterID'].unique()
            for meter_id in tqdm(meter_ids, desc="Processing MeterIDs", unit=" MeterID"):
                meter_data = df[df['MeterID'] == meter_id]
                new_df[meter_id] = new_df.index.map(
                    lambda dt: meter_data[meter_data['DateTime'] == dt]['kWh'].values[0] if len(meter_data[meter_data['DateTime'] == dt]['kWh']) > 0 else np.nan
                )

            # Reset the index of the new DataFrame
            new_df.reset_index(inplace=True)
            new_df.set_index('index', inplace=True)
            new_df.sort_index()
            logger.info("Data grouped by MeterID")
        except Exception as e:
            logger.error(f"Failed to group data by MeterID: {e}")
            raise e

        return new_df

    
    def applyDateTimeTransform(self, dataframe: DataFrame):
        # Use tqdm to track progress
        tqdm.pandas(desc="Converting", unit=" rows")

        logger.info("Applying datetime transform")
        try:
            # Apply the conversion function to the 'code' column
            dataframe['DateTime'] = dataframe['codeDateTime'].progress_apply(self.code_to_datetime)
            logger.info("Datetime transformed")
        except Exception as e:
            logger.error(f"Failed to apply datetime transform: {e}")
            raise e

        return dataframe
    
    def loadTransform(self, skip_transformed = True):
        # Load Data and Transform DateTime
        raw_files = self.scanDir(self.raw_path, ".txt")
        transformed_files = self.scanDir(self.interim_path, ".txt")
        logger.info(f"Loading and transforming {len(raw_files)} files")
        try:
            for i, file in enumerate(raw_files):
                if f"transformed_{raw_files[i][:-4]}.txt" in transformed_files and skip_transformed:
                    logger.info(f"Skipping {file}, already transformed")
                else:
                    data = self.loadData(self.raw_path, file)
                    transformed_data = self.applyDateTimeTransform(data)
                    self.saveInterimData(f"transformed_{raw_files[i][:-4]}.txt", transformed_data)
            logger.info(f"Loaded and transformed {len(raw_files)} files")
        except Exception as e:
            logger.error(f"Failed to load and transform files: {e}")
            raise e

    def loadGroup(self, skip_grouped = True):
        # Load Data and Group by MeterID
        raw_files = self.scanDir(self.raw_path, ".txt")
        transformed_files = self.scanDir(self.interim_path, ".txt")
        grouped_files = self.scanDir(self.interim_path, ".csv")

        logger.info(f"Loading and grouping {len(raw_files)} files")
        try:
            for i, file in enumerate(transformed_files):
                if f"grouped_data_{raw_files[i][:-4]}.csv" in grouped_files and skip_grouped:
                    logger.info(f"Skipping {file}, already grouped")
                else:
                    transformed_data = self.loadData(self.interim_path,
                                                        file, 
                                                        names=None, sep=',')
                    grouped_data = self.group_by_meter_id(transformed_data)
                    self.saveInterimData(f"grouped_data_{raw_files[i][:-4]}.csv", grouped_data)
            logger.info(f"Loaded and grouped {len(raw_files)} files")
        except Exception as e:
            logger.error(f"Failed to load and group files: {e}")
            raise e

    def loadConcatenate(self, skip_concatenated = True):
        # Load Data and Concatenate
        grouped_files = self.scanDir(self.interim_path, ".csv")
        concatenated_file = self.scanDir(self.interim_path, ".csv")

        logger.info(f"Loading and concatenating {len(grouped_files)} files")
        try:
            if f"concatenated_data.csv" in concatenated_file and skip_concatenated:
                logger.info(f"Skipping concatenation, `concatenated_data.csv` already exists")
            else:
                grouped_dataframes = []
                for i, file in enumerate(grouped_files):
                    grouped_dataframes.append(self.loadData(self.interim_path,
                                                            file,
                                                            names=None, sep=','))
                concatenated_data = pd.concat(grouped_dataframes, axis=1)
                concatenated_data.sort_index()
                self.saveInterimData(f"concatenated_data.csv", concatenated_data)
        except Exception as e:
            logger.error(f"Failed to load and concatenate files: {e}")
            raise e

    

if __name__ == "__main__":
    make_data = makeDataset("./data/raw", "./data/interim")
    make_data.loadTransform()
    make_data.loadGroup()
    make_data.loadConcatenate()


    


