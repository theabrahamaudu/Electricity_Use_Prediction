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
    """Test endpoint to verify server up.

    Returns:
        JSONResponse: Default response message
    """    
    logger.info("API service tested")
    return JSONResponse(content={"message": "System is healthy"})

# File upload endpoint
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)) -> JSONResponse:
    """Endpoint to upload file to server from client side

    Read and save uploaded file on server.

    Return JSON response with parameters:
        "filename": Path to uploaded file on server\n
        "success_message": Status of upload operation for UI return

    Args:
        file (UploadFile, optional): File from client machine. Defaults to File(...).

    Returns:
        JSONResponse: File path on server and success message
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
    except Exception as e:
        logger.error(f"File processing failed: {e}")
        success_message = f"File processing failed: {e}"
        file_path = ""
    return JSONResponse(content={"filename": file_path,"success_message": success_message})


@app.post("/predict")
def predict() -> JSONResponse:
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


    

