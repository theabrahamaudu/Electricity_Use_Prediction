import os
import time
import pandas as pd
from pandas import DataFrame
import numpy as np
import json
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import mpld3
from mpld3 import plugins
from io import StringIO, BytesIO
import zipfile
import requests


# specify temporary files folder
temp_folder = os.path.join(os.path.dirname(__file__), "temp")
os.makedirs(temp_folder, exist_ok=True)

server: str = 'http://127.0.0.1:8000' # Local


def run():
    """
    Streamlit configuration for Electricity Load Forecasting web user interface

    - Allows user to upload previous load consumption data to server
    - Allows user to toggle file processing on server
    - Sends request to backend API to make prediction and then displays results
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
                    "5. Display results")
    
    with st.spinner("Adding file to queue..."):
        file = st.file_uploader("Choose consumption history (CSV)", type=['csv'])


    if file is not None:
        # Create a session state object
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
            # Send request to process uploaded file
            with st.spinner("Forecasting..."):
                pred = requests.post(server+"/predict", verify=False)
                # Parse request response
                if pred.status_code == 200:
                    pred = pred.json()
                    if pred['pred_time'] != "Invalid":
                        st.info(f"{pred['pred_time']}, {pred['single_pred_time']}")
                        st.session_state['filename'] = pred['filename']


                    else:
                        st.warning(f"Error making predictions on server")
                else:
                    st.warning(f"Error making prediction request: code {pred.status_code}")

        # Display results
        if st.button("Plot Forecast"):
            # Retrieve forecast     
            with st.spinner("Retrieving forecast data..."):
                forecast_data = requests.post(server+"/retrieve", json={"filename":st.session_state['filename']}, verify=False)
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
            st.write("working")
            if len(npy_content) > 0:
                st.write("working too")
                with st.spinner("Generating plots..."):
                    st.write("working too bro")
                    zip_file = zipfile.ZipFile(BytesIO(npy_content))

                    # Extract the npy files from the zip archive
                    npy_files = [filename for filename in zip_file.namelist() if filename.endswith('.npy')]

                    if len(npy_files) >= 3:
                        # Read the first npy file into a NumPy array (predictions data)
                        npy_data1 = zip_file.read(npy_files[0])
                        preds_arr = np.load(BytesIO(npy_data1))

                        # Read the second npy file into a NumPy array (processed data)
                        npy_data2 = zip_file.read(npy_files[1])
                        targets_arr = np.load(BytesIO(npy_data2))

                        # Read the third file into a dictionary from JSON
                        json_data = zip_file.read(npy_files[2])
                        json_dict = json.loads(json_data)

                        # Plot the predictions and targets
                        fig = plt.figure(figsize=(20, 8))
                        plt.plot(preds_arr, label='Predictions')
                        plt.plot(targets_arr, label='Actual')
                        # plt.title('Power Consumption Forecast')
                        # plt.xlabel('Timestamp')
                        # plt.ylabel('kWh')
                        plt.legend()
                        fig_html = mpld3.fig_to_html(fig)
                        components.html(fig_html, height=600)

                        # Display the metrics
                        # st.write("RMSE:", json_dict['rmse'])
                        # st.write("RMSE less than 10%:", json_dict['rmse_less_10'])
                        # st.write("NRMSE mean:", json_dict['nrmse_mean'])
                        # st.write("NRMSE max-min:", json_dict['nrmse_max_min'])

                        # Define some CSS to control our custom labels
                        # css = """
                        # table
                        # {
                        # border-collapse: collapse;
                        # }
                        # th
                        # {
                        # color: #ffffff;
                        # background-color: #000000;
                        # }
                        # td
                        # {
                        # background-color: #cccccc;
                        # }
                        # table, th, td
                        # {
                        # font-family:Arial, Helvetica, sans-serif;
                        # border: 1px solid black;
                        # text-align: right;
                        # }
                        # """
                        # for axes in fig.axes:
                        #     for line in axes.get_lines():
                        #         # get the x and y coords
                        #         xy_data = line.get_xydata()
                        #         labels = []
                        #         for x, y in xy_data:
                        #             # Create a label for each point with the x and y coords
                        #             html_label = f'<table border="1" class="dataframe"> <thead> <tr style="text-align: right;"> </thead> <tbody> <tr> <th>x</th> <td>{x}</td> </tr> <tr> <th>y</th> <td>{y}</td> </tr> </tbody> </table>'
                        #             labels.append(html_label)
                        #         # Create the tooltip with the labels (x and y coords) and attach it to each line with the css specified
                        #         tooltip = plugins.PointHTMLTooltip(line, labels, css=css)
                        #         # Since this is a separate plugin, you have to connect it
                        #         plugins.connect(fig, tooltip)

if __name__ == "__main__":
    run()
                    