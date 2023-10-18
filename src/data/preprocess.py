    # def saveProcessedData(self, filename: str, dataframe: DataFrame):
    #     dataframe.to_csv(self.processed_path+"/"+filename, index=True, header=True)
    #     logger.info(f"Saved {filename} to {self.processed_path}")

        
    
    # def filterData(self, dataframe: DataFrame):
    #     # Load filter codes data
    #     label_data = pd.read_excel(f'{self.raw_path}/CER_Electricity_Documentation/SME and Residential allocations.xlsx',
    #                        sheet_name='Sheet1',
    #                        usecols=['ID', 'Code', 'Residential - Tariff allocation', 'Residential - stimulus allocation', 'SME allocation']
    #                     )
        
    #     # Get control meters
    #     control_meters = []
    #     for i in range(len(label_data)):
    #         if label_data['Residential - Tariff allocation'][i] == 'E' or\
    #         label_data['Residential - stimulus allocation'][i] == 'E' or\
    #         label_data['SME allocation'][i] == 'C':
    #             control_meters.append(str(label_data['ID'][i]))

    #     # Filter out control Meters from concatenated data
    #     filtered_data = dataframe.drop(columns=control_meters)
    #     self.saveInterimData("filtered_concatenated_data.csv", filtered_data)

      