from abc import ABC, abstractmethod
import numpy as np

class MLModel(ABC):
    """Abstract base class for machine learning models.

    Defines abstract methods for training the model, setting hyperparameters, and making predictions.

    Methods:
        train: Abstract method for training the model.
        set_hyperparameters: Abstract method for setting hyperparameters.
        predict: Abstract method for making predictions.
    """

    @abstractmethod
    def train(self, training_set: np.ndarray, validation_set: np.ndarray, metrics_list: list, verbose: bool) -> dict:
        """Train the model.

        Args:
            training_set (np.ndarray): Training data.
            validation_set (np.ndarray): Validation data.
            verbose (bool): Verbosity mode.

        Returns:
            dict: Dictionary containing training/validation performance metrics.
        """
        pass

    @abstractmethod
    def set_hyperparameters(self, hyper_param: dict):
        """Set hyperparameters for the model.

        Args:
            hyper_param (dict): Dictionary containing hyperparameters.
        """
        pass

    @abstractmethod
    def predict(self, patterns: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model.

        Args:
            patterns (np.ndarray): Input data for prediction.

        Returns:
            np.ndarray: Predicted values.
        """
        pass