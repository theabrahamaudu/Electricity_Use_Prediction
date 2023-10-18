import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame
from datetime import datetime, timedelta
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Dask
import dask.dataframe as dd
from dask import delayed, compute
from dask.diagnostics import ProgressBar

concat_data = pd.read_csv('./data/interim/concatenated_data.csv',
                          parse_dates=['index'], index_col='index')

# Load label data
label_data = pd.read_excel('./data/raw/CER_Electricity_Documentation/SME and Residential allocations.xlsx',
                           sheet_name='Sheet1',
                           usecols=['ID', 'Code', 'Residential - Tariff allocation', 'Residential - stimulus allocation', 'SME allocation']
                        )

# Filter out control Meters
control_meters = []
for i in range(len(label_data)):
    if label_data['Residential - Tariff allocation'][i] == 'E' or\
    label_data['Residential - stimulus allocation'][i] == 'E' or\
    label_data['SME allocation'][i] == 'C':
        control_meters.append(str(label_data['ID'][i]))

# Filter out control Meters from concatenated data
filtered_data = concat_data.drop(columns=control_meters)

cols_with_na = 0
cols = []
for i in filtered_data.columns:
    if filtered_data[i].isna().sum() > len(filtered_data)*0.20:
        cols_with_na += 1
        cols.append(i)

print(cols_with_na)
print("Perentage:",cols_with_na/len(filtered_data.columns)*100)
# print(cols)

def sumAllRows(dataframe: pd.DataFrame) -> pd.DataFrame:

    # Sum all rows along axis 1
    row_sums = dataframe.sum(axis=1)

    # Create a new DataFrame with only the row sums
    result_df = pd.DataFrame({'kWh': row_sums})
    
    return result_df

filtered_data_no_mv = filtered_data.drop(columns=cols)

# Fill missing values column by column
partitions = 4
filled_data1 = dd.from_pandas(filtered_data_no_mv.iloc[:, :500], npartitions=partitions)
filled_data2 = dd.from_pandas(filtered_data_no_mv.iloc[:, 500:1000], npartitions=partitions)
filled_data3 = dd.from_pandas(filtered_data_no_mv.iloc[:, 1000:1500], npartitions=partitions)
filled_data4 = dd.from_pandas(filtered_data_no_mv.iloc[:, 1500:2000], npartitions=partitions)
filled_data5 = dd.from_pandas(filtered_data_no_mv.iloc[:, 2000:2500], npartitions=partitions)
filled_data6 = dd.from_pandas(filtered_data_no_mv.iloc[:, 2500:3000], npartitions=partitions)
filled_data7 = dd.from_pandas(filtered_data_no_mv.iloc[:, 3000:3500], npartitions=partitions)
filled_data8 = dd.from_pandas(filtered_data_no_mv.iloc[:, 3500:4000], npartitions=partitions)
filled_data9 = dd.from_pandas(filtered_data_no_mv.iloc[:, 4000:4500], npartitions=partitions)
filled_data10 = dd.from_pandas(filtered_data_no_mv.iloc[:, 4500:], npartitions=partitions)
# filled_data11 = dd.from_pandas(filtered_data_no_mv.iloc[:, 5000:5500], npartitions=10)
# filled_data12 = dd.from_pandas(filtered_data_no_mv.iloc[:, 5500:], npartitions=10)

with ProgressBar():
    filled_data1 = filled_data1.ffill().compute(num_workers=12)
    filled_data2 = filled_data2.ffill().compute(num_workers=12)
    filled_data3 = filled_data3.ffill().compute(num_workers=12)
    filled_data4 = filled_data4.ffill().compute(num_workers=12)
    filled_data5 = filled_data5.ffill().compute(num_workers=12)
    filled_data6 = filled_data6.ffill().compute(num_workers=12)
    filled_data7 = filled_data7.ffill().compute(num_workers=12)
    filled_data8 = filled_data8.ffill().compute(num_workers=12)
    filled_data9 = filled_data9.ffill().compute(num_workers=12)
    filled_data10 = filled_data10.ffill().compute(num_workers=12)

# Check stats
summed_filled = [filled_data1, filled_data2, filled_data3, filled_data4, filled_data5,
                 filled_data6, filled_data7, filled_data8, filled_data9, filled_data10]
data_list = []
for i in summed_filled:
    summed_filled_data = sumAllRows(i)
    data_list.append(summed_filled_data)

# Concatenate all data
all_data_concat = pd.concat(data_list, axis=1, ignore_index=True)
all_data_concat = sumAllRows(all_data_concat)
print(all_data_concat.describe())