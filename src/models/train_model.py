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
    """
    A class for building, training, and evaluating a deep learning model for time series forecasting.

    This class provides a set of methods and utilities for building, training, and evaluating 
    deep learning models for time series forecasting. It includes functionalities for loading processed data,
    configuring GPU settings, building and training models, and evaluating model performance.

    Args:
        processed_path (str, optional): The path to the processed data. Defaults to "./data/processed".
        reports_path (str, optional): The path to save reports and figures. Defaults to "./reports".
        models_path (str, optional): The path to save trained models. Defaults to "./models".
        version (str, optional): The version identifier for the model. Defaults to "0.2.4-fulldata".

    Attributes:
        version (str): The model version identifier.
        processed_path (str): The path to processed data.
        reports_path (str): The path to save reports and figures.
        models_path (str): The path to save trained models.

    Example:
        from train_model import modelPipeline

        # Initialize the model pipeline
        modeling = modelPipeline(
            processed_path="./data/processed",
            reports_path="./reports",
            models_path="./models",
            version= "0.2.4-fulldata"
        )

        # Build the model
        model = modeling.build_model()

        # Save the model diagram
        modeling.save_model_diagram(model)

        # Train the model
        training_output = modeling.train_model(model)

        # Save the model
        modeling.save_model(training_output)
    """
    def __init__(self,
                 processed_path: str = "./data/processed",
                 reports_path: str = "./reports",
                 models_path: str = "./models",
                 version: str= "0.2.4-fulldata"):
        """
        Initialize the modelPipeline class.

        Args:
            processed_path (str, optional): The path to the processed data. Defaults to "./data/processed".
            reports_path (str, optional): The path to save reports and figures. Defaults to "./reports".
            models_path (str, optional): The path to save trained models. Defaults to "./models".
            version (str, optional): The version identifier for the model. Defaults to "0.2.4-fulldata".

        Attributes:
            version (str): The model version identifier.
            processed_path (str): The path to processed data.
            reports_path (str): The path to save reports and figures.
            models_path (str): The path to save trained models.

        Raises:
            Exception: If there is an error loading processed data.

        Example:
            from train_model import modelPipeline

            # Initialize the model pipeline
            model = modelPipeline(processed_path="./data/processed", reports_path="./reports", models_path="./models")

            # Access model attributes and methods
            print(model.version)
            model.train_model()
        """
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
        """
        Set seeds to make the experiment more reproducible.

        Args:
            seed: The seed value for random number generators.

        Sets the random seed for Python's built-in random number generator (`random.seed`),
        NumPy (`np.random.seed`), and TensorFlow (`tf.random.set_seed`).
        It also sets environment variables to enforce determinism in TensorFlow operations and GPU memory allocation.

        Example:
            from train_model import modelPipeline

            # Initialize the model pipeline
            model = modelPipeline()

            # Set seeds for reproducibility
            model.seed_everything(seed=42)
        """
        # Set seeds to make the experiment more reproducible.
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

    
    def configure_gpu(self):
        """
        Configure GPU for TensorFlow-based model training.

        This method attempts to configure the GPU for TensorFlow-based model training.
        It checks the available GPUs, sets memory growth for each GPU to avoid memory allocation issues, and logs the configuration.

        Note that if the system does not have GPUs, the training will fall back to running on the CPU.

        Example:
            from train_model import modelPipeline

            # Initialize the model pipeline
            model = modelPipeline()

            # Configure GPU for model training
            model.configure_gpu()
        """
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
        """
        Load a NumPy array from a file.

        This method loads a NumPy array from a specified file and returns the loaded array.
        It also logs the filename and path from which the array is loaded.

        Parameters:
            filename (str): The name of the file to load the NumPy array from.

        Returns:
            ndarray: The loaded NumPy array.

        Example:
            from train_model import modelPipeline

            # Initialize the model pipeline
            model = modelPipeline()

            # Load a NumPy array from a file
            loaded_array = model.load_array('my_array.npy')
        """
        array = np.load(f'{self.processed_path}/{filename}')
        logger.info(f"Loaded {filename} from {self.processed_path}")
        return array
    

    def save_model_diagram(self, model: Model):
        """
        Save a diagram of the model architecture as an image.

        This method generates a diagram of the given TensorFlow Keras model's architecture and saves it as an image file.

        Parameters:
            model (Model): The TensorFlow Keras model for which the architecture diagram will be generated and saved.

        Example:
            from train_model import modelPipeline

            # Initialize the model pipeline
            model = modelPipeline()

            # Load a pre-trained model
            loaded_model = model.load_model()

            # Save the model's architecture as an image
            model.save_model_diagram(loaded_model)
        """
        plot_model(model,
           show_shapes=True, 
           show_layer_names=False,
           show_layer_activations=True,
           show_trainable=True, 
           to_file=f'{self.reports_path}/figures/architecture_{self.version}.png')
        logger.info(f"Saved model diagram to {self.reports_path} as architecture_{self.version}.png")   
    

    def save_partial_history(self, model: Model, overwrite=False):
        """
        Save training history metrics to a JSON file for a partially trained model.

        This method saves the training history metrics, including loss and validation loss, to a JSON file.
        It is typically used for saving training progress when training is interrupted and resumed.

        Parameters:
            model (Model): The TensorFlow Keras model whose training history will be saved.
            overwrite (bool): If True, overwrite the existing file; otherwise, append to the existing data.

        Example:
            from train_model import modelPipeline

            # Initialize the model pipeline
            model = modelPipeline()

            # Load a partially trained model
            loaded_model = model.load_model(history=True)

            # Save the partial training history
            model.save_partial_history(loaded_model, overwrite=False)
        """
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


    def build_model(self) -> Model:
        """
        Build and compile a deep learning model for time series forecasting.

        This method defines and compiles a deep learning model for time series forecasting using TensorFlow and Keras.

        Returns:
            Model: The compiled deep learning model.

        Example:
            from modelPipeline import modelPipeline

            # Initialize the model pipeline
            model = modelPipeline()

            # Build the model
            model = model.build_model()
        """
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
    

    def train_model(self, model: Model) -> tuple[Model, tf.keras.callbacks.History | None, float]:
        """
        Train a deep learning model for time series forecasting.

        This method trains the provided deep learning model using the training data and specified hyperparameters.

        Args:
            model (Model): The deep learning model to be trained.

        Returns:
            tuple[Model, tf.keras.callbacks.History | None, float]: A tuple containing the trained model,
            training history (if available), and the training time in seconds.

        Example:
            from modelPipeline import modelPipeline

            # Initialize the model pipeline
            model = modelPipeline()

            # Build the model
            model = model.build_model()

            # Train the model
            trained_model, history, training_time = model.train_model(model)
        """
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
        """
        Save a trained deep learning model and its training history to disk.

        This method saves the trained deep learning model and its training history (if available) to the specified directories.

        Args:
            training_output (tuple[Model, tf.keras.callbacks.History | None, float]): A tuple containing the trained model,
            training history (if available), and the training time in seconds.

        Example:
            from modelPipeline import modelPipeline

            # Initialize the model pipeline
            model = modelPipeline()

            # Build and train the model
            trained_model, history, training_time = model.train_model()

            # Save the trained model and history
            model.save_model((trained_model, history, training_time))
        """
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
        """
        Load a trained deep learning model and its training history from disk.

        This method loads a previously trained deep learning model and its training history (if available) from the specified directories.

        Args:
            history (bool, optional): If True, load the model history along with the model. Defaults to False.

        Returns:
            Model | tuple[Model, dict]: The loaded deep learning model. If history is True, a tuple containing the model
            and its training history dictionary is returned.

        Example:
            from modelPipeline import modelPipeline

            # Initialize the model pipeline
            model = modelPipeline()

            # Load a trained model without history
            loaded_model = model.load_model()

            # Load a trained model with history
            loaded_model, loaded_history = model.load_model(history=True)
        """
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

    @staticmethod    
    def plot_history(history: dict):
        """
        Plot the training and validation loss from a model's training history.

        This method is a static method that takes a dictionary containing training and validation loss values
        and plots them to visualize the training progress.

        Args:
            history (dict): A dictionary containing training and validation loss values.

        Returns:
            None

        Example:
            from modelPipeline import modelPipeline

            # Load the model history
            model = modelPipeline()
            loaded_history = model.load_model(history=True)

            # Plot the training and validation loss
            model.plot_history(loaded_history)
        """
        plt.figure(figsize=(10, 10))
        plt.plot(history['loss'], label='loss')
        plt.plot(history['val_loss'], label='val_loss')
        plt.legend()
        plt.show()

    
    def unscale_data(self, y_test: np.ndarray, yhat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Unscale the scaled test and prediction data.

        This method takes scaled test data (y_test) and scaled prediction data (yhat), and it un-scales both
        using the scaler that was used during data preprocessing. This is necessary to get the predictions back
        to the original scale.

        Args:
            y_test (np.ndarray): Scaled test data.
            yhat (np.ndarray): Scaled prediction data.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing un-scaled test data and un-scaled prediction data.

        Example:
            from modelPipeline import modelPipeline

            # Create a model instance
            model = modelPipeline()

            # Load a trained model
            trained_model = model.load_model()

            # Generate predictions
            yhat = trained_model.predict(model.X_test)

            # Unscale the predictions
            unscaled_y_test, unscaled_yhat = model.unscale_data(model.y_test, yhat)
        """
        logger.info("Unscaling test and prediction targets")
        try:
            return self.scaler.inverse_transform(y_test), self.scaler.inverse_transform(yhat)
        except Exception as e:
            logger.error(f"Failed to unscale data: {e}")
            raise e


    def stat_eval(self, y_test: np.ndarray, yhat: np.ndarray) -> tuple[float, int, float, float]:
        """
        Perform statistical evaluation of model predictions.

        This method calculates various statistical metrics to evaluate the model's performance based on
        test data and predictions. It calculates Root Mean Squared Error (RMSE), Normalized RMSE (NRMSE) with 
        respect to mean and range, and checks if RMSE is less than 10% of the mean value.

        Args:
            y_test (np.ndarray): Test data.
            yhat (np.ndarray): Model predictions.

        Returns:
            tuple[float, int, float, float]: A tuple containing RMSE, an indicator of whether RMSE is 
            less than 10% of the mean value, NRMSE with respect to mean, and NRMSE with respect to the 
            range of test data.

        Example:
            from modelPipeline import modelPipeline

            # Create a model instance
            model = modelPipeline()

            # Load a trained model
            trained_model = model.load_model()

            # Generate predictions
            yhat = trained_model.predict(model.X_test)

            # Evaluate the model's predictions
            rmse, rmse_less_10, nrmse_mean, nrmse_max_min = model.stat_eval(model.y_test, yhat)

            # Print the evaluation results
            print("RMSE:", rmse)
            print("Is RMSE less than 10% of the mean:", rmse_less_10)
            print("NRMSE with respect to mean:", nrmse_mean)
            print("NRMSE with respect to range:", nrmse_max_min)
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
        Save model performance metrics to a JSON file.

        This method saves various performance metrics, including training and inference times, RMSE, 
        and NRMSE, to a JSON file. Metrics are organized by model version, and you can choose whether
        to overwrite existing metrics or append the new metrics to an existing JSON file.

        Args:
            train_time (float): Training time in seconds.
            infer_time (float): Inference time per data point in seconds.
            rmse (float): Root Mean Squared Error (RMSE) of the model's predictions.
            rmse_less_10 (int): Indicator of whether RMSE is less than 10% of the mean value.
            nrmse_mean (float): Normalized RMSE (NRMSE) with respect to the mean of the test data.
            nrmse_max_min (float): Normalized RMSE (NRMSE) with respect to the range of the test data.
            overwrite (bool, optional): If True, overwrite the existing JSON file. 
                If False, append the new metrics. Defaults to False.

        Example:
            from modelPipeline import modelPipeline

            # Create a model instance
            model = modelPipeline()

            # Load a trained model
            trained_model = model.load_model()

            # Generate predictions
            yhat = trained_model.predict(model.X_test)

            # Evaluate the model's predictions
            rmse, rmse_less_10, nrmse_mean, nrmse_max_min = model.stat_eval(model.y_test, yhat)

            # Save performance metrics to a JSON file
            model.save_metrics(train_time=120.0, infer_time=0.005, rmse=5.0, 
                            rmse_less_10=1, nrmse_mean=0.1, nrmse_max_min=0.2)
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
        """
        Evaluate the model's performance on the test data and save metrics.

        This method takes a trained model, generates predictions on the test data, 
        and computes various performance metrics, including RMSE (Root Mean Squared Error),
        NRMSE (Normalized RMSE), training time, and inference time. The evaluation results 
        are then saved to a JSON file.

        Args:
            model (tf.keras.Model): A trained Keras model for making predictions.

        Returns:
            np.ndarray: Model predictions on the test data.

        Example:
            from modelPipeline import modelPipeline

            # Create a model instance
            model = modelPipeline()

            # Load a trained model
            trained_model = model.load_model()

            # Evaluate the model's performance and save metrics
            yhat = model.evaluate_model(trained_model)
        """
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
        """
        Generate and save a test plot comparing actual and predicted data.

        This method takes predicted data (`yhat`), unscales it, and generates a plot
        that compares the actual and predicted values for a specified number of datapoints.
        The plot is saved as an image file and can be displayed if the `display` parameter is set to True.

        Args:
            yhat (np.ndarray): Predicted data from the model.
            datapoints (int, optional): Number of datapoints to include in the plot. Defaults to 712.
            display (bool, optional): Whether to display the plot interactively. Defaults to False.

        Example:
            from modelPipeline import modelPipeline

            # Create a model instance
            model = modelPipeline()

            # Load a trained model
            trained_model = model.load_model()

            # Generate and save a test plot
            yhat = model.evaluate_model(trained_model)
            model.save_testplot(yhat, datapoints=100, display=True)
        """
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