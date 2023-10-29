Electricity Consumption Forecasting
==============================

Timeseries forecasting of electricity use with AutoEncoder LSTM

Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    │
    ├── .streamlit         <- Streamlit configuration
    │
    ├── data
    │   ├── external       <- Experimental and UI testing data.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- Experimental process documentation
    │
    ├── logs               <- Log files for pipeline and backend
    │
    ├── models             <- Trained and serialized models & model histories
    │
    ├── notebooks          <- Jupyter notebooks for experimental codes
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures of models and tests to be used in reporting 
    │   └── forecasts      <- Generated graphics and figures of forecast outside the dataset to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to generate and preprocess data
    │   │   └── make_dataset.py
    │   │   └── preprocess.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   └── train_model.py
    │   │
    │   └── utils          <- Scripts to create loggers and use preprocessing methods in server endpoints
    │
    ├── temp               <- Directory for backend temporary files
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

# Electricity Consumption Forecasting

This project is able to take previous electricity consumption data and make a forecast of future electricity consumption for the city.
It can be used by interacting with the web UI.

## Description

The forecasting infrastructure of this project was built using the ISSDA CER Electricity [dataset](https://www.ucd.ie/issda/data/commissionforenergyregulationcer/).

The dataset was then preprocessed to transform the timestamp codes to standard datetime format, group by the meter IDs, remove control meter IDs and remove data points with excessive missing values, as well as summing up the data naively by timestammp, and then scaling, to bring all the features within a narrow range of 0 and 1 for better performance.  

Finally, the data was restructured using a sliding window approach which allows the model to take the previous `n_steps` as input to predict the next `m_steps` into the future.

## Architechture
Below is the project architecture.

## Model Performance
Several AutoEncoder LSTM model architectures were experimented in order to develop the final model

Here are the stats for the best model which trained for 55 epochs (48,125 steps), using a data strucutre which allows the model consider 1440 previous temsteps to predict the next timestep.
~~~
===== Statistical Eval ===== 

Mean kWh: 3407.6062 
RMSE: 115.8286 -- RMSE < 10% Mean:  True 
NRMSE Mean:  0.03399119798831835 -- NRMSE Mean < 0.01:  False 
NRMSE Max Min:  0.02076489853254594 -- NRMSE Max Min < 0.01:  False
~~~
Where `NRMSE Mean` stands for NRMSE calculated using the mean of the data, and `NRMSE Max Min` for NRMSE calculated using the range of the data. 
All the models as well as scaler were saved to memory so as to enable the transformation of new data from end users before prediction on the data.

The app has a prediction API built with FastAPI and a frontend built with Streamlit

## Basic Workflow

- Perform EDA
- Preprocess raw data and save scaler
- Train models and save models
- Test models
- Setup API service
- Design frontend

## API Structure

The API service has five endpoints:

* '/test': A test endpoint to verify that the server is up and running.
* '/upload': An endpoint to upload a file to the server.
* '/process': An endpoint to process the uploaded file and prepare it for prediction.
* '/predict': An endpoint to make predictions based on the processed data.
* '/retrieve': An endpoint to retrieve predictions, actual targets, and evaluation metrics.

## Frontend Structure

The frontend is built with Streamlit.  

How it works:

- Allows user to upload previous load consumption data to server
- Allows user to toggle file processing on server
- Sends request to backend API to make prediction
- Sends request to backend for predicted data and then displays results

## Getting Started

### Dependencies
OS Level:
* Windows/MacOS/Ubuntu (Tested on Ubuntu)
* graphviz (standalone app required by Keras for plotting model)  

Python:
* ipykernel
* pandas
* dask
* tqdm
* tensorflow==2.14.0
* matplotlib
* scikit-learn
* attention
* openpyxl
* pydot
* fastapi
* uvicorn
* streamlit
* python-multipart
* plotly

### Installing

To test this project on your local machine:

* Create a new project
* Clone this repo by running:

    ```
    git clone https://github.com/theabrahamaudu/Electricity_Use_Prediction.git
    ```

* install the requirements by running:

    ```
    pip install -r requirements.txt
    ```
* Request and dowload the dataset from the ISSDA website using the link in the description section above

* Unzip the meter data `.txt` files (six in total) and copy them into the `data/raw` directory

* Open the `main.py` file and set all stage flags to true

* Run the data preprocessing, model training and evaluation pipeline:
    ```
    python main.py
    ```
    ###### N.B: This version of the model trained for ~35 hours on RTX 3060 6GB Mobile GPU. Alternatively, message me to get the dowload link for the final model to demo the project (paste it in the `models` dir). For this approach, set the `Model Training` stage flag to `False`

### Executing program
* Start the API service by running:

    ```
    python app.py
    ```

* Run the Streamlit frontend:

    ```
    streamlit run streamlit_app.py
    ```

* To test the app, use the `test_data.csv` in the `./data/external` directory as previous electricity consumption data

## Help

Feel free to reach out to me if you have any issues using the platform


## Version History

* See [commit change](https://github.com/theabrahamaudu/credit_card_default_predictor/commits/main)
* See [release history](https://github.com/theabrahamaudu/credit_card_default_predictor/releases)


