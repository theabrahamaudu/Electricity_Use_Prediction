import os
import pandas as pd
from datetime import datetime, timedelta
from pandas import DataFrame
import numpy as np
from tqdm import tqdm
from src.utils.pipeline_log_config import pipeline as logger

class makeDataset:
    def __init__(self, raw_path: str, interim_path: str, processed_path):
        self.raw_path = raw_path
        self.interim_path = interim_path
        self.processed_path = processed_path


    def scan_directory(self, extension: str) -> list:
        """Check `raw_path` directory and return list of files with
        specified extension

        Args:
            extension (str): extension type to be searched for e.g. ".txt"

        Returns:
            list: strings of file names with specified extension
        """
        directory = self.raw_path    
        files: list = []
        for filename in os.listdir(directory):
            if filename.endswith(extension):
                files.append(filename)
        return files

    def loadRawData(self, filename: str):
        dataframe = pd.read_csv(self.raw_path+"/"+filename, header=None, sep=' ', names=['MeterID', 'codeDateTime', 'kWh'])

        return dataframe
    
    def saveInterimData(self, filename: str, dataframe: DataFrame):
        dataframe.to_csv(self.interim_path+"/"+filename, index=True, header=True)
    
    def saveProcessedData(self, filename: str, dataframe: DataFrame):
        dataframe.to_csv(self.processed_path+"/"+filename, index=True, header=True)
    
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
        # Initialize an empty DataFrame with 'DateTime' as the index
        unique_datetimes = df['DateTime'].unique()
        new_df = pd.DataFrame(index=unique_datetimes)

        # Iterate through MeterIDs and populate the new DataFrame with progress tracking
        meter_ids = df['MeterID'].unique()
        for meter_id in tqdm(meter_ids, desc="Processing MeterIDs"):
            meter_data = df[df['MeterID'] == meter_id]
            new_df[meter_id] = new_df.index.map(
                lambda dt: meter_data[meter_data['DateTime'] == dt]['kWh'].values[0] if len(meter_data[meter_data['DateTime'] == dt]['kWh']) > 0 else np.nan
            )

        # Reset the index of the new DataFrame
        new_df.reset_index(inplace=True)
        new_df.set_index('index', inplace=True)

        return new_df

    
    def applyDateTimeTransform(self, dataframe: DataFrame):
        # Use tqdm to track progress
        tqdm.pandas(desc="Converting")

        # Apply the conversion function to the 'code' column
        dataframe['DateTime'] = dataframe['codeDateTime'].progress_apply(self.code_to_datetime)

        return dataframe
    

if __name__ == "__main__":
    make_data = makeDataset("./data/raw", "./data/interim", "./data/processed")
    file_1 = make_data.loadRawData("File1.txt")
    transformed_file_1 = make_data.applyDateTimeTransform(file_1)
    grouped_file_1 = make_data.group_by_meter_id(transformed_file_1)
    make_data.saveInterimData("grouped_File1.csv", grouped_file_1)