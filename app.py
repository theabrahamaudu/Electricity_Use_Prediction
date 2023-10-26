"""
Electricity Use Prediction API Module

This module defines a FastAPI-based API for electricity use prediction. It provides endpoints for testing the server,
uploading files, processing data, predicting electricity use, and retrieving predictions and metrics.

The API includes the following endpoints:
1. '/test': A test endpoint to verify that the server is up and running.
2. '/upload': An endpoint to upload a file to the server.
3. '/process': An endpoint to process the uploaded file and prepare it for prediction.
4. '/predict': An endpoint to make predictions based on the processed data.
5. '/retrieve': An endpoint to retrieve predictions, actual targets, and evaluation metrics.

Attributes:
    app (FastAPI): An instance of the FastAPI application.
    logger (Logger): A logger for logging API events.

Usage:
    To run the API, execute this module. The API will be accessible at 'http://127.0.0.1:8000'.
    You can use various API endpoints to interact with the model and retrieve predictions.

    Example:
    $ uvicorn app:app --host 127.0.0.1 --port 8000 --reload
"""

import os
import time
import zipfile
import numpy as np
import uvicorn
from fastapi import FastAPI, Response, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from src.data.preprocess import preprocessPipeline
from src.utils.backend_log_config import backend as logger
from src.utils.backend_utils import metricsUtils


# Initialize FastAPI
app = FastAPI(title='Electricity Use Prediction API',
              version='0.1.0',
              description="Used to demo the Electricity Use Prediction model's capabilities",
              )

# API test endpoint
@app.get('/test')
@app.get('/')
def test_page() -> JSONResponse:
    """
    Test Endpoint

    This endpoint is used to verify that the server is up and running. It provides a default response message to confirm the system's health.

    Returns:
        JSONResponse: A JSON response containing a message indicating the system's health.

    Example:
        To check if the API service is running, make a GET request to '/test' or '/'.
        The endpoint will respond with a JSON message confirming the system's health.

        JSON Response:
        {
            "message": "System is healthy"
        }
    """
    
    logger.info("API service tested")
    return JSONResponse(content={"message": "System is healthy"})

# File upload endpoint
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> JSONResponse:
    """
    Upload File Endpoint

    This endpoint allows clients to upload a file to the server. The uploaded file is read and saved on the server, and a JSON response is returned, containing the file path on the server and a success message for UI display.

    Args:
        file (UploadFile, optional): The file to be uploaded from the client machine. Defaults to File(...).

    Returns:
        JSONResponse: A JSON response containing two parameters:
        - "filename" (str): The path to the uploaded file on the server.
        - "success_message" (str): A status message indicating the success or failure of the upload operation.

    Example:
        To upload a file to the server, make a POST request to '/upload' with the file as part of the request.
        The endpoint will respond with a JSON message that includes the path to the uploaded file on the server and a success message.

        JSON Response (Success):
        {
            "filename": "path_to_uploaded_file_on_server",
            "success_message": "File uploaded successfully!"
        }

        JSON Response (Failure):
        {
            "filename": "",
            "success_message": "File upload failed: Error message"
        }
    """
    
    filename = file.filename
    
    try:
        if filename:
            # File was selected and submit button was clicked, save the file
            main_directory = os.getcwd()
            directory_path = os.path.join(main_directory, "temp/interim")

            # Clear all files in temp dir
            # Get the list of files in the directory
            file_list = os.listdir(directory_path)

            # Iterate over the files and delete them
            for file_name in file_list:
                if ".git" not in file_name:
                    file_path = os.path.join(directory_path, file_name)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        logger.info(f"Deleted file: {file_path}")

            # Define new file file path
            file_path = os.path.join(directory_path, filename)

            # Create the directory if it does not exist
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)

            # Write the uploaded file
            with open(file_path, "wb") as f:
                contents = await file.read()
                f.write(contents)
            logger.info(f"File '{filename}' uploaded and saved to: {file_path}")
            success_message = f"{filename} Uploaded successfully!"
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        success_message = f"File upload failed: {e}"
        file_path = ""

    return JSONResponse(content={"filename": file_path, "success_message": success_message})


@app.post("/process")
def process_file(data: dict) -> JSONResponse:
    """
    Process File Endpoint

    This endpoint processes an uploaded file to prepare it for prediction. 
    It reads the uploaded file, processes the data, and saves the processed data 
    as separate files (dataX.npy and dataY.npy). 
    The response includes the paths to the processed files and a success message.

    Args:
        data (dict): A dictionary containing the filename of the uploaded file.

    Returns:
        JSONResponse: A response containing the paths to the processed data files (dataX.npy and dataY.npy) and a success message.

    Example:
        To process an uploaded file, send a POST request to '/process' with the following data in the request body:

        {
            "filename": "uploaded_data.csv"
        }

        Successful Response:
        The response will contain the paths to the processed data files and a success message.

        {
            "filename": "./temp/processed/dataX.npy",
            "success_message": "File 'uploaded_data.csv' processed successfully!"
        }

        Failed Response:
        If processing fails, the response will contain an error message.

        {
            "filename": "",
            "success_message": "File processing failed: Error details..."
        }
    """

    logger.info(f"Processing file: {data['filename']}")
    try:
        pipeline = preprocessPipeline(raw_path="./temp/raw",
                                     interim_path="./temp/interim",
                                     processed_path="./temp/processed")
        dataX, dataY = pipeline.inferencePreprocess(data['filename'])
        file_path = "./temp/processed/dataX.npy"
        targets_path = "./temp/processed/dataY.npy"
        np.save(file_path, dataX)
        np.save(targets_path, dataY)
        logger.info(f"File '{data['filename']}' processed successfully!")
        success_message = f"File '{data['filename']}' processed successfully!"
        return JSONResponse(content={"filename": file_path,"success_message": success_message})
    except Exception as e:
        logger.error(f"File processing failed: {e}")
        success_message = f"File processing failed: {e}"
        file_path = ""
    return JSONResponse(content={"filename": file_path,"success_message": success_message})


@app.post("/predict")
def predict() -> JSONResponse:
    """
    Predict File Endpoint

    This endpoint performs predictions on a processed data file. The data is loaded from a file,
    and a pre-trained model is used to make predictions. 
    The predicted data is saved to a file, and a JSON response is returned, 
    containing the file path of the predictions, the total prediction time, and the time per prediction.

    Returns:
        JSONResponse: A JSON response containing the following parameters:
        - "filename" (str): The path to the file containing the predictions.
        - "pred_time" (float or str): The total time taken for all predictions in seconds or "Invalid" if an error occurred.
        - "single_pred_time" (float or str): The time taken for a single prediction in seconds or "Invalid" if an error occurred.

    Example:
        To make predictions on processed data, send a POST request to '/predict'.
        The endpoint will respond with a JSON message that includes the path to the file containing the predictions, 
        the total prediction time, and the time per prediction.

        JSON Response (Success):
        {
            "filename": "path_to_prediction_file",
            "pred_time": 1.234,  # Total prediction time in seconds
            "single_pred_time": 0.005  # Time per prediction in seconds
        }

        JSON Response (Failure):
        {
            "filename": "",
            "pred_time": "Invalid",
            "single_pred_time": "Invalid"
        }
    """

    logger.info(f"Predicting file: ./temp/processed/dataX.npy")
    try:
        logger.info(f"Loading file: ./temp/processed/dataX.npy")
        processed_data = np.load("./temp/processed/dataX.npy")
    except Exception as e:
        logger.error(f"File loading failed: {e}")
    

    try:
        logger.info(f"Predicting file: ./temp/processed/dataX.npy")
        # Load Model
        MODELS_DIR = './models'
        model = tf.keras.models.load_model(MODELS_DIR + '/model_0.2.4-fulldata.keras')
        start = time.perf_counter()
        prediction = model.predict(processed_data)
        pred_time = time.perf_counter() - start
        single_pred_time = pred_time/len(prediction)
        filename = "./temp/processed/prediction.npy"
        np.save(filename, prediction)
        logger.info(f"File ./temp/processed/dataX.npy predicted successfully!")
    except Exception as e:
        logger.error(f"File prediction failed: {e}")
        pred_time = "Invalid"
        single_pred_time = "Invalid"
        filename = ""

    return JSONResponse(content={"filename": filename, "pred_time": pred_time, "single_pred_time": single_pred_time})

@app.post("/retrieve", response_model=None)
def retrieve() -> JSONResponse | Response:
    """
    Retrieve File Endpoint

    This endpoint retrieves prediction data and associated metrics from the server. It loads the predictions, actual targets, and metrics data, creates a zip archive containing these files, and returns the archive for download. The response can be a ZIP file with the predictions, actual targets, and metrics data, or an error message if the files are not available.

    Returns:
        JSONResponse | Response: If successful, a ZIP file containing predictions, actual targets, and metrics data is returned for download. If there's an error, a JSON response with an error message is returned.

    Example:
        To retrieve prediction data, send a POST request to '/retrieve'.
        The endpoint will respond with a ZIP file containing the prediction data, actual targets, and metrics data.

        Successful ZIP Response:
        A ZIP file is returned for download, containing the prediction data, actual targets, and metrics data.

        Failed Response:
        {
            "response": "Error: File path not provided."
        }
    """

    logger.info(f"Retrieving file: ./temp/processed/dataX.npy and corresponding actual targets")

    preds = np.load("./temp/processed/prediction.npy")
    actual = np.load("./temp/processed/dataY.npy")
    metrics_utils = metricsUtils()
    rmse, rmse_less_10, nrmse_mean, nrmse_max_min = metrics_utils.stat_eval(actual, preds)
    metrics_utils.save_metrics(rmse, rmse_less_10, nrmse_mean, nrmse_max_min)


    preds_array_path = "./temp/processed/prediction_unscaled.npy"
    targets_array_path = "./temp/processed/dataY_unscaled.npy"
    metrics = "./temp/processed/metrics.json"
    file_paths = [preds_array_path, targets_array_path, metrics]

    if file_paths:
        try:
            # Create a zip archive containing the files
            zip_file_path = './temp/interim/files.zip'
            with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for file_path in file_paths:
                    file_name = os.path.basename(file_path)
                    zip_file.write(file_path, file_name)

            # Set the appropriate HTTP response headers
            with open(zip_file_path, 'rb') as zip_file:
                content = zip_file.read()

            response = Response(content=content, media_type='application/zip')
            response.headers['Content-Disposition'] = 'attachment; filename="files.zip"'

            return response

        except Exception as e:
            return JSONResponse(content={"response": f"Error: {e}"})

    return JSONResponse(content={"response": "Error: File path not provided."})



if __name__ == '__main__':
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)


    

