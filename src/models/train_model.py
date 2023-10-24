# Regular imports
import os
import glob
import random
import numpy as np
import json
import time
import pickle
import joblib
from numpy import ndarray
import matplotlib.pyplot as plt
from src.utils.pipeline_log_config import pipeline as logger

# Metrics
import math
from sklearn.metrics import mean_squared_error

# Tensorflow
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras import optimizers, Sequential, Model
from tensorflow.keras.utils import plot_model

class modelPipeline:
    def __init__(self,
                 processed_path: str = "./data/processed",
                 reports_path: str = "./reports",
                 models_path: str = "./models",
                 version: str= "0.2.4-fulldata"):
        # Model version
        self.version = version

        # Paths
        self.processed_path = processed_path
        self.reports_path = reports_path
        self.models_path = models_path

        # Seed for reproducibility
        logger.info("Setting seed for reproducibility")
        self.seed = 0
        self.seed_everything(self.seed)

        # Configure GPU
        logger.info("Attempting to configure GPU")
        self.configure_gpu()

        # Load processed data
        logger.info("Loading processed data")
        try:
            self.X_train = self.load_array('X_train.npy')
            self.y_train = self.load_array('y_train.npy')
            self.X_test = self.load_array('X_test.npy')
            self.y_test = self.load_array('y_test.npy')
        except Exception as e:
            logger.error(f"Failed to load processed data: {e}")
            raise e

        # Define Hyperparameters 
        self.serie_size =  self.X_train.shape[1]
        self.n_features =  self.X_train.shape[2]

        self.epochs = 100
        self.patience = int(np.sqrt(self.epochs))
        self.batch = 20
        self.lr = 0.00001

        # Load scaler
        self.scaler = joblib.load(f"{self.processed_path}/scaler.pkl")
    
    def seed_everything(self, seed):
        # Set seeds to make the experiment more reproducible.
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    
    def configure_gpu(self):
        try:
            gpus = tf.config.list_physical_devices('GPU')
            logger.info(f"Found {len(gpus)} GPU(s): {gpus}")
        except Exception as e:
            logger.warning("Error configuring GPU: \n", e)
            logger.info("Training will run on CPU")

        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info("Memory growth enabled for all Physical GPUs")

    def load_array(self, filename: str) -> ndarray:
        array = np.load(f'{self.processed_path}/{filename}')
        logger.info(f"Loaded {filename} from {self.processed_path}")
        return array
    

    def save_model_diagram(self, model: Model):
        plot_model(model,
           show_shapes=True, 
           show_layer_names=False,
           show_layer_activations=True,
           show_trainable=True, 
           to_file=f'{self.reports_path}/figures/architecture_{self.version}.png')
        logger.info(f"Saved model diagram to {self.reports_path} as architecture_{self.version}.png")   
    

    def save_partial_history(self, model: Model, overwrite=False):
        encoder_decoder_history = model.history.history
        loss = encoder_decoder_history['loss']
        val_loss = encoder_decoder_history['val_loss']
        epochs = len(loss)

        file_path = f"{self.models_path}/partials/partial_history_{self.version}.json"

        if not overwrite:
            # Read the existing JSON data from the file
            try:
                with open(file_path, "r") as file:
                    existing_data = json.load(file)
            except FileNotFoundError:
                existing_data = {'loss': [], 'val_loss': []}
        else:
            existing_data = {'loss': [], 'val_loss': []}

        existing_data['loss'].extend(loss)
        existing_data['val_loss'].extend(val_loss)

        # Write the combined data back to the JSON file
        with open(file_path, "w") as file:
            json.dump(existing_data, file, indent=4)
        logger.info(f"\n{epochs} epoch metrics saved to partial_history_{self.version}.json")


    def build_model(self):
        logger.info(f"Building model:: version: {self.version}")
        try:
            # Define the model
            encoder_decoder = Sequential()
            encoder_decoder.add(L.Bidirectional(L.LSTM(self.serie_size, activation='tanh', input_shape=(self.serie_size, self.n_features), return_sequences=True)))
            encoder_decoder.add(L.LSTM(256, activation='tanh', return_sequences=True))
            encoder_decoder.add(L.LSTM(128, activation='tanh', return_sequences=True))
            encoder_decoder.add(L.LSTM(64, activation='tanh', return_sequences=False))
            encoder_decoder.add(L.BatchNormalization())
            encoder_decoder.add(L.RepeatVector(self.serie_size))
            encoder_decoder.add(L.LSTM(self.serie_size, activation='tanh', return_sequences=True))
            encoder_decoder.add(L.BatchNormalization())
            encoder_decoder.add(L.LSTM(64, activation='tanh', return_sequences=True))
            encoder_decoder.add(L.LSTM(128, activation='tanh', return_sequences=True))
            encoder_decoder.add(L.LSTM(256, activation='tanh', return_sequences=True))
            encoder_decoder.add(L.Bidirectional(L.LSTM(128, activation='tanh', return_sequences=False)))
            encoder_decoder.add(L.Dropout(0.2))
            encoder_decoder.add(L.Dense(self.y_train.shape[1]))

            # Compile the model
            adam = optimizers.Adam(self.lr)
            encoder_decoder.compile(loss='mse', optimizer=adam)

            # Build the model
            encoder_decoder.build(input_shape=(None, self.serie_size, self.n_features))
            encoder_decoder.summary()
            logger.info(f"Built model:: version: {self.version}")

            return encoder_decoder
        except Exception as e:
            logger.error(f"Failed to build model: {e}")
            raise e
    

    def train_model(self, model: Model):
        # set conditions to stop training if the model begins to overfit
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                        patience=self.patience, 
                                                        mode='min',
                                                        restore_best_weights=True)
        # Setup backup and restore callback
        resume = tf.keras.callbacks.BackupAndRestore(backup_dir=f"{self.models_path}/backups")

        logger.info(f"Training model:: version: {self.version}")
        alt_start = time.perf_counter()
        try:
            start = time.perf_counter()
            encoder_decoder_history = model.fit(self.X_train, self.y_train, 
                                                        batch_size=self.batch, 
                                                        epochs=self.epochs, 
                                                        verbose=1,
                                                        callbacks=[early_stopping, resume],
                                                        validation_split=0.1)
            self.train_time = time.perf_counter() - start
            logger.info(f"Training completed in {self.train_time} seconds")
            return model, encoder_decoder_history, self.train_time
        except KeyboardInterrupt as e:
            self.alt_train_time = time.perf_counter() - alt_start
            self.save_partial_history(model)
            logger.warning(f"Model training interrupted: {e}")
            return model, None, self.alt_train_time
        

    def save_model(self, training_output: tuple):
        model, encoder_decoder_history, _ = training_output

        if encoder_decoder_history is not None:
            # Save the model
            model.save(f"{self.models_path}/model_{self.version}.keras")
            logger.info(f"Model saved to {self.models_path}/model_{self.version}.keras")
            # Save the model history
            with open(f"{self.models_path}/model_history_{self.version}.pkl", "wb") as f:
                pickle.dump(encoder_decoder_history.history, f)
            logger.info(f"Model history saved to {self.models_path}/model_history_{self.version}.pkl")
        else:
            # Save the partial model
            model.save(f"{self.models_path}/partials/model_{self.version}_partial-{len(model.history.history['loss'])}-epochs.keras")
            logger.info(f"Model saved to {self.models_path}/partials/model_{self.version}_partial-{len(model.history.history['loss'])}-epochs.keras")

    
    def load_model(self, history: bool = False) -> Model | tuple[Model, dict]:
        if history:
            try:
                with open(f"{self.models_path}/model_history_{self.version}.pkl", "rb") as f:
                    history = pickle.load(f)
                model = tf.keras.models.load_model(f"{self.models_path}/model_{self.version}.keras")
                logger.info(f"Model loaded from {self.models_path}/model_{self.version}.keras")
                return model, history
            except Exception as e:
                logger.error(f"Failed to load model with history: {e}")
                raise e
        else:
            try:
                try:
                    model = tf.keras.models.load_model(f"{self.models_path}/model_{self.version}.keras")
                    logger.info(f"Model loaded from {self.models_path}/model_{self.version}.keras")
                    return model
                except:
                    model_name = glob.glob(f"{self.models_path}/partials/model_{self.version}*.keras")[0]
                    model = tf.keras.models.load_model(model_name)
                    logger.info(f"Model loaded from {model_name}")
                    return model
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                raise e
        
    def plot_history(self, history: dict):
        plt.figure(figsize=(10, 10))
        plt.plot(history['loss'], label='loss')
        plt.plot(history['val_loss'], label='val_loss')
        plt.legend()
        plt.show()

    
    def unscale_data(self, y_test: np.ndarray, yhat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        logger.info("Unscaling test and prediction targets")
        try:
            return self.scaler.inverse_transform(y_test), self.scaler.inverse_transform(yhat)
        except Exception as e:
            logger.error(f"Failed to unscale data: {e}")
            raise e


    def stat_eval(self, y_test: np.ndarray, yhat: np.ndarray) -> tuple[float, int, float, float]:
        """
        Carries out statistical evaluation of model based on test data and 
        prediction. 
        
        Prints and returns a summary dictionary
        """
        # unscaling test and prediction targets
        y_test, yhat = self.unscale_data(y_test, yhat)

        # statistical evaluations
        print("="*5,"Statistical Eval","="*5,"\n")

        rmse = math.sqrt(mean_squared_error(y_pred=yhat, y_true=y_test))
        nrmse_mean = float(rmse/np.mean(y_test))
        nrmse_max_min = float(rmse/(np.max(y_test)-np.min(y_test)))

        print(f"Mean kWh: {round(np.mean(y_test), 4)} \nRMSE: {round(rmse, 4)}",
            f"-- RMSE < 10% Mean: ", round(rmse, 4)< round(np.mean(y_test), 4)/10,
            f"\nNRMSE Mean: ", nrmse_mean, "-- NRMSE Mean * 100: ", nrmse_mean*100,
            f"\nNRMSE Max Min: ", nrmse_max_min, "-- NRMSE Max Min * 100: ", nrmse_max_min*100)
        
        rmse_less_10 = 1 if round(rmse, 4)< round(np.mean(y_test), 4)/10 else 0
                        
        return rmse, rmse_less_10, nrmse_mean, nrmse_max_min
    

    def save_metrics(self, 
                 train_time: float, 
                 infer_time: float, 
                 rmse: float,
                 rmse_less_10: int,
                 nrmse_mean: float,
                 nrmse_max_min: float,
                 overwrite: bool = False
                 ):
        """
        Saves metrics to a JSON file.

        Parameters:
        version (str): Version identifier.
        train_time (float): Training time.
        infer_time (float): Inference time.
        rmse (float): Root Mean Squared Error.
        rmse_less_10 (bool): Boolean indicating if RMSE is less than 10.
        file_path (str): Path to the JSON file.
        overwrite (bool): If True, overwrite the existing file.

        Returns:
        None
        """
        new_metrics = {
            "version": self.version,
            "metrics": {
                "train_time": train_time,
                "infer_time": infer_time,
                "rmse": rmse,
                "rmse_less_10": rmse_less_10,
                "nrmse_mean": nrmse_mean,
                "nrmse_max_min": nrmse_max_min
            }
        }

        file_path = f"{self.reports_path}/metrics.json"

        if not overwrite:
            # Read the existing JSON data from the file
            try:
                with open(file_path, "r") as file:
                    existing_data = json.load(file)
            except FileNotFoundError:
                existing_data = []
        else:
            existing_data = []

        # Check if the new metrics already exist, and only append if not
        if new_metrics not in existing_data:
            existing_data.append(new_metrics)

            # Write the combined data back to the JSON file
            with open(file_path, "w") as file:
                json.dump(existing_data, file, indent=4)

            logger.info(f"Model version {self.version} metrics saved to file: {self.reports_path}/metrics.json")
        else:
            print(f"Model version {self.version} metrics already exist in file: {self.reports_path}/metrics.json")


    def evaluate_model(self, model: Model):
        logger.info("Evaluating model")
        try:
            start = time.perf_counter()
            yhat = model.predict(self.X_test)
            infer_time = (time.perf_counter() - start)/len(yhat)

            rmse, rmse_less_10, nrmse_mean, nrmse_max_min = self.stat_eval(self.y_test, yhat)

            try:
                train_time = self.train_time
            except AttributeError:
                logger.warning("`train_time` not detected. Attempting to use `alt_train_time`")
                try:
                    train_time = self.alt_train_time
                except AttributeError:
                    logger.warning("`alt_train_time` not detected. Training time set to 0.0")
                    train_time = 0.0

            self.save_metrics(train_time=train_time,
                            infer_time=infer_time,
                            rmse=rmse,
                            rmse_less_10=rmse_less_10,
                            nrmse_mean=nrmse_mean,
                            nrmse_max_min=nrmse_max_min)
            return yhat
        except Exception as e:
            logger.error(f"Failed to evaluate model: {e}")
            raise e
    

    def save_testplot(self, yhat: ndarray, datapoints: int = 712, display: bool = False):
        y_test, yhat = self.unscale_data(self.y_test, yhat)
        plt.figure(figsize=(20,8))
        plt.plot(y_test[:datapoints])
        plt.plot(yhat[:datapoints])
        plt.legend(['Actual', 'Predicted'])
        plt.xlabel('Time')
        plt.ylabel('kWh')
        plt.savefig(f'{self.reports_path}/figures/testplot_{self.version}.png')
        logger.info(f"Test plot saved to {self.reports_path}/figures/testplot_{self.version}.png")
        if display:
            plt.show()