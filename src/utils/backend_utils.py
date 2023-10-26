import numpy as np
import json
import joblib
import math
from sklearn.metrics import mean_squared_error
from src.utils.backend_log_config import backend as logger

class metricsUtils:
    def __init__(self, scaler_path: str = "./temp/processed/scaler.pkl"):
        """
        Initialize a metricsUtils instance.

        Args:
            scaler_path (str, optional): Path to the scaler file. Defaults to "./temp/processed/scaler.pkl".

        Example:
            # Create a metricsUtils instance
            metrics = metricsUtils()
        """
        self.scaler = joblib.load(scaler_path)
    

    def unscale_data(self, y_test: np.ndarray, yhat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Unscale test and prediction data.

        This method takes scaled test and prediction data, unscales it, and saves the unscaled data to files.

        Args:
            y_test (np.ndarray): Scaled test data.
            yhat (np.ndarray): Scaled prediction data.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing unscaled test and prediction data.

        Example:
            # Unscaled test and prediction data
            y_test_unscaled, yhat_unscaled = metrics.unscale_data(y_test, yhat)
        """
        logger.info("Unscaling test and prediction targets")
        try:
            y_test_unscaled = self.scaler.inverse_transform(y_test)
            yhat_unscaled = self.scaler.inverse_transform(yhat)
            np.save("./temp/processed/dataY_unscaled.npy", y_test_unscaled)
            np.save("./temp/processed/prediction_unscaled.npy", yhat_unscaled)
            logger.info("Unscaled data saved to ./temp/processed/dataY_unscaled.npy and ./temp/processed/prediction_unscaled.npy")
            return y_test_unscaled, yhat_unscaled
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
            # Evaluate the model's predictions
            rmse, rmse_less_10, nrmse_mean, nrmse_max_min = metrics.stat_eval(y_test, yhat)

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
                    rmse: float,
                    rmse_less_10: int,
                    nrmse_mean: float,
                    nrmse_max_min: float,
                    overwrite: bool = True
                    ):
        """
        Save metrics to a JSON file.

        This method saves the calculated metrics to a JSON file, and you can choose to overwrite the file or append the data.

        Args:
            rmse (float): Root Mean Squared Error.
            rmse_less_10 (int): Indicator of whether RMSE is less than 10% of the mean value (0 or 1).
            nrmse_mean (float): Normalized RMSE with respect to mean.
            nrmse_max_min (float): Normalized RMSE with respect to the range of test data.
            overwrite (bool, optional): If True, overwrite the existing JSON file. Defaults to True.

        Example:
            # Save metrics to a JSON file
            metrics.save_metrics(rmse, rmse_less_10, nrmse_mean, nrmse_max_min, overwrite=True)
        """
        new_metrics = {
                "rmse": rmse,
                "rmse_less_10": rmse_less_10,
                "nrmse_mean": nrmse_mean,
                "nrmse_max_min": nrmse_max_min
            }
        
        joblib.dump(new_metrics, "./temp/processed/metrics.pkl")
        file_path = f"./temp/processed/metrics.json"

        if not overwrite:
            # Read the existing JSON data from the file
            try:
                with open(file_path, "r") as file:
                    existing_data = json.load(file)
            except FileNotFoundError:
                existing_data = []
        else:
            existing_data = []

        # Append the new data
        existing_data.append(new_metrics)

        # Write the combined data back to the JSON file
        with open(file_path, "w") as file:
            json.dump(existing_data, file, indent=4)

        logger.info(f"Forecast metrics saved to file: ./temp/processed/metrics.json")
    