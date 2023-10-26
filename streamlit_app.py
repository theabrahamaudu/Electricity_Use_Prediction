"""
Electricity Load Forecasting Streamlit Interface

This module provides a Streamlit web user interface for electricity load forecasting. 
The interface allows users to upload previous load consumption data, 
process the data on the server, request load predictions, and display the results.

Usage:
- Upload consumption history data.
- Process the uploaded data on the server.
- Request electricity load predictions.
- Display the results and performance metrics.

The interface connects to a backend API server for data processing and predictions.

Requirements:
- The `streamlit` library should be installed.
- A backend API server is expected to be running and accessible at the specified URL.

Example:
To use this module, run the script. The Streamlit web interface will start, allowing users to upload data and interact with the forecasting model.
"""

import os
import numpy as np
import json
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO
import zipfile
import requests

# Set server URL
server: str = 'http://127.0.0.1:8000' # Local


def run():
    """
    Streamlit configuration for Electricity Load Forecasting web user interface

    - Allows user to upload previous load consumption data to server
    - Allows user to toggle file processing on server
    - Sends request to backend API to make prediction
    - Sends request to backend for predicted data and then displays results
    """

    st.set_page_config(page_title="Load Forecasting",
                        page_icon=":chart_with_upwards_trend:")
    st.image("https://images.pexels.com/photos/923953/pexels-photo-923953.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=1")
    st.title("Electricity Load Forecasting")
    st.subheader("Built with AutoEncoder LSTM Model\n")

    st.sidebar.text("Steps:\n"
                    "1. Choose consumption history file\n"
                    "2. Upload file\n"
                    "3. Process file\n"
                    "4. Request prediction\n"
                    "5. Request for, and Display results")
    
    # File selection interface
    with st.spinner("Adding file to queue..."):
        file = st.file_uploader("Choose consumption history (CSV)", type=['csv'])


    if file is not None:
        # Create a session state object to hold the file name
        st.session_state['filename'] = file.name

        # File upload
        if st.button("Upload"):
            # Send file upload request
            with st.spinner("Uploading file..."):
                up_status = requests.post(server+"/upload", files={"file": file}, verify=False)
                # Parse request response
                if up_status.status_code == 200:
                    up_status = up_status.json()
                    st.session_state['filename'] = up_status['filename']
                    st.success(f"File uploaded successfully!")
                else:
                    st.error("File upload failed.")

        # File Processing
        if st.button("Process data"):
            # Send request to process uploaded file
            with st.spinner("Processing file..."):
                state = requests.post(server+"/process", json={"filename":st.session_state['filename']}, verify=False).json()
                # Parse request response
                if "successfully" in str(state["success_message"]):
                    st.info(state['success_message'])
                    st.session_state['filename'] = state['filename']
                else:
                    st.warning(state['success_message'])

        # Forecasting
        if st.button("Forecast"):
            # Send request to predict on uploaded file
            with st.spinner("Forecasting..."):
                pred = requests.post(server+"/predict", verify=False)
                # Parse request response
                if pred.status_code == 200:
                    pred = pred.json()
                    if pred['pred_time'] != "Invalid":
                        st.success(f"Total Inference Time: {round(pred['pred_time'], 4)} secs")
                        st.success(f"Time per Inference: {round(pred['single_pred_time'], 4)}")
                        st.session_state['filename'] = pred['filename']
                    else:
                        st.warning(f"Error making predictions on server")
                else:
                    st.warning(f"Error making prediction request: code {pred.status_code}")

        # Display results
        plot_range = st.number_input("Number of days to plot", min_value=1, max_value=10000, value=3, step=1)
        plot_range = plot_range * 48
        if st.button("Plot Forecast"):
            # Retrieve forecast     
            with st.spinner("Retrieving forecast data..."):
                forecast_data = requests.post(server+"/retrieve", verify=False)
                # Check if the request was successful (status code 200)
                if forecast_data.status_code == 200:
                    try:
                        # Access the Numpy file content from the response
                        npy_content = forecast_data.content
                        st.info("Forecast data retrived")
                    except: 
                        # Throw error if file has dictionary
                        response_dict = forecast_data.json()
                        st.warning(response_dict['response'])

            if npy_content:
                with st.spinner("Generating plots..."):
                    zip_file = zipfile.ZipFile(BytesIO(npy_content), 'r')

                    # Extract the npy files from the zip archive
                    pred_files = zip_file.namelist() 
                    
                    if len(pred_files) >= 3:
                        # Read the first npy file into a NumPy array (predictions data)
                        npy_data1 = zip_file.read(pred_files[0])
                        preds_arr = np.load(BytesIO(npy_data1))

                        # Read the second npy file into a NumPy array (processed data)
                        npy_data2 = zip_file.read(pred_files[1])
                        targets_arr = np.load(BytesIO(npy_data2))

                        # Read the third file into a dictionary from JSON
                        json_data = zip_file.read(pred_files[2])
                        json_dict = json.load(BytesIO(json_data))

                        if plot_range > len(targets_arr):
                            plot_range = len(targets_arr)
                        
                        # Plot the predictions and targets
                        fig = plt.figure(figsize=(8,5))
                        plt.plot(targets_arr[:plot_range], label='Actual')
                        plt.plot(preds_arr[:plot_range], label='Predicted')
                        plt.title('Power Consumption Forecast')
                        plt.xlabel('Timestamp')
                        plt.ylabel('kWh')
                        plt.legend()
                        st.plotly_chart(fig, use_container_width=True)


                        # Display the metrics
                        st.info(f"RMSE: {str(round(json_dict[0]['rmse'], 4))} kWh")
                        st.info(f"RMSE less than 10% of mean: {'True' if json_dict[0]['rmse_less_10'] == 1 else 'False'}")
                        st.info(f"NRMSE by mean: {str(round(json_dict[0]['nrmse_mean'], 4))}")
                        st.info(f"NRMSE by range: {str(round(json_dict[0]['nrmse_max_min'], 4))}")

if __name__ == "__main__":
    run()
                    