from abc import ABC, abstractmethod
import numpy as np
import math

class ActivationFunction(ABC):
    """Abstract base class for activation functions.

    Defines abstract methods for computing the activation function and its derivative.

    Methods:
        compute: Abstract method for computing the activation function.
        derivate: Abstract method for computing the derivative of the activation function.
    """
    @abstractmethod
    def compute(self, x):
        """Abstract method for computing the activation function."""
        pass
    
    @abstractmethod
    def derivate(self, x):
        """Abstract method for computing the derivative of the activation function."""
        pass

class SigmoidFunction(ActivationFunction):
    """Sigmoid activation function.

    Sigmoid function maps any real-valued number to the range between 0 and 1. It's commonly used as an activation function
    in neural networks for binary classification problems.

    Attributes:
        slope (float): Slope parameter for controlling the steepness of the sigmoid curve.
        vectorized_fun (function): Vectorized sigmoid function for applying element-wise on arrays.

    Methods:
        __init__: Initializes the sigmoid function with an optional slope parameter.
        compute: Computes the sigmoid activation function for a given input or array of inputs.
        derivate: Computes the derivative of the sigmoid activation function for a given input or array of inputs.
    """

    def __init__(self, slope=2):
        """Initialize the sigmoid function with an optional slope parameter.

        Args:
            slope (float): Slope parameter for controlling the steepness of the sigmoid curve. Default is 2.
        """
        self.slope = slope
        # Vectorize sigmoid function for array operations
        self.vectorized_fun = np.vectorize(lambda in_x: 1/(1 + math.exp(-(in_x * self.slope))))

    def compute(self, x: float | np.ndarray):
        """Compute the sigmoid activation function for a given input or array of inputs.

        Args:
            x (float or np.ndarray): Input value(s) to compute the activation function.

        Returns:
            float or np.ndarray: Computed activation value(s).
        """
        if isinstance(x, float):
            return 1/(1 + math.exp(-(x * self.slope)))
        else:
            return self.vectorized_fun(x)

    def derivate(self, x: float | np.ndarray):
        """Compute the derivative of the sigmoid activation function for a given input or array of inputs.

        Args:
            x (float or np.ndarray): Input value(s) to compute the derivative.

        Returns:
            float or np.ndarray: Computed derivative value(s).
        """
        tmp = self.compute(x)
        return self.slope * tmp * (1 - tmp)

class IdentityFunction(ActivationFunction):
    """Identity activation function.

    Identity function returns the input value itself. It's commonly used in the output layer of a neural network for
    regression problems.

    Methods:
        compute: Computes the identity activation function for a given input or array of inputs.
        derivate: Computes the derivative of the identity activation function (which is constant 1).

    """

    def compute(self, x: float | np.ndarray):
        """Compute the identity activation function for a given input or array of inputs.

        Args:
            x (float or np.ndarray): Input value(s) to compute the activation function.

        Returns:
            float or np.ndarray: Computed activation value(s).
        """
        if isinstance(x, float):
            return x
        else:
            return x.copy()

    def derivate(self, x: float | np.ndarray):
        """Compute the derivative of the identity activation function.

        For the identity function, the derivative is constant and equal to 1.

        Args:
            x (float or np.ndarray): Input value(s) to compute the derivative.

        Returns:
            float or np.ndarray: Computed derivative value(s).
        """
        if isinstance(x, float):
            return 1
        else:
            return np.ones(x.shape)

class HyperbolicTangentFunction(ActivationFunction):
    """Hyperbolic Tangent (tanh) activation function.

    Hyperbolic tangent function maps any real-valued number to the range between -1 and 1. It's commonly used as an
    activation function in neural networks.

    Attributes:
        slope (float): Slope parameter for controlling the steepness of the tanh curve.
        vectorized_fun (function): Vectorized tanh function for applying element-wise on arrays.

    Methods:
        __init__: Initializes the tanh function with an optional slope parameter.
        compute: Computes the tanh activation function for a given input or array of inputs.
        derivate: Computes the derivative of the tanh activation function for a given input or array of inputs.
    """

    def __init__(self, slope=2):
        """Initialize the tanh function with an optional slope parameter.

        Args:
            slope (float): Slope parameter for controlling the steepness of the tanh curve. Default is 2.
        """
        self.slope = slope
        # Vectorize tanh function for array operations
        self.vectorized_fun = np.vectorize(lambda in_x: (math.exp(2 * in_x * self.slope) - 1) / (math.exp(2 * in_x * self.slope) + 1))

    def compute(self, x: float | np.ndarray):
        """Compute the tanh activation function for a given input or array of inputs.

        Args:
            x (float or np.ndarray): Input value(s) to compute the activation function.

        Returns:
            float or np.ndarray: Computed activation value(s).
        """
        if isinstance(x, float):
            return (math.exp(2 * x * self.slope) - 1) / (math.exp(2 * x * self.slope) + 1)
        else:
            return self.vectorized_fun(x)

    def derivate(self, x: float | np.ndarray):
        """Compute the derivative of the tanh activation function for a given input or array of inputs.

        Args:
            x (float or np.ndarray): Input value(s) to compute the derivative.

        Returns:
            float or np.ndarray: Computed derivative value(s).
        """
        tmp = self.compute(x)
        return (1 - self.slope * tmp**2)
    
class RectifiedLinearUnit(ActivationFunction):
    """Rectified Linear Unit (ReLU) activation function.

    It returns the input value if it is positive, otherwise returns zero.

    Attributes:
        vectorized_fun (function): Vectorized ReLU function for applying element-wise on arrays.
        vectorized_derivate (function): Vectorized derivative of the ReLU function for computing gradients.

    Methods:
        __init__: Initializes the ReLU activation function.
        compute: Computes the ReLU activation function for a given input or array of inputs.
        derivate: Computes the derivative of the ReLU activation function for a given input or array of inputs.
    """

    def __init__(self):
        """Initialize the ReLU activation function."""
        # Vectorize ReLU function and its derivative for array operations
        self.vectorized_fun = np.vectorize(lambda in_x: in_x if in_x > 0 else 0.0)
        self.vectorized_derivate = np.vectorize(lambda in_x: 1.0 if in_x > 0 else 0.0)
        
    def compute(self, x: float | np.ndarray):
        """Compute the ReLU activation function for a given input or array of inputs.

        Args:
            x (float or np.ndarray): Input value(s) to compute the activation function.

        Returns:
            float or np.ndarray: Computed activation value(s).
        """
        if isinstance(x, float):
            return x if x > 0 else 0.0
        else:
            return self.vectorized_fun(x)
    
    def derivate(self, x: float | np.ndarray):
        """Compute the derivative of the ReLU activation function for a given input or array of inputs.

        Args:
            x (float or np.ndarray): Input value(s) to compute the derivative.

        Returns:
            float or np.ndarray: Computed derivative value(s).
        """
        if isinstance(x, float):
            return 1.0 if x > 0 else 0.0
        else:
            return self.vectorized_derivate(x)
        
class ExponentialLinearUnit(ActivationFunction):
    """Exponential Linear Unit (ELU) activation function.

    ELU is a type of activation function commonly used in neural networks. It is similar to the Rectified Linear Unit (ReLU)
    but has a smooth gradient for negative values, which can help with the vanishing gradient problem during training.

    Attributes:
        slope (float): Slope parameter to control the shape of the function.
        vectorized_fun (function): Vectorized ELU function for applying element-wise on arrays.
        vectorized_derivate (function): Vectorized derivative of the ELU function for computing gradients.

    Methods:
        __init__: Initializes the ELU with a given slope parameter.
        compute: Computes the ELU activation function for a given input or array of inputs.
        derivate: Computes the derivative of the ELU activation function for a given input or array of inputs.
    """
    
    def __init__(self, slope = 1):
        """Initialize the ELU activation function with a given slope parameter.

        Args:
            slope (float): Slope parameter to control the shape of the function.
        """
        self.slope = slope
        # Vectorize ELU function and its derivative for array operations
        self.vectorized_fun = np.vectorize(lambda in_x: in_x if in_x >= 0 else self.slope*(math.exp(in_x)-1))
        self.vectorized_derivate = np.vectorize(lambda in_x: 1 if in_x > 0 else self.slope*np.exp(in_x))
        
    def compute(self, x):
        """Compute the ELU activation function for a given input or array of inputs.

        Args:
            x (float or array_like): Input value(s) to compute the activation function.

        Returns:
            float or ndarray: Computed activation value(s).
        """
        if isinstance(x, float):
            return x if x >= 0 else self.slope*(math.exp(x)-1)
        else:
            return self.vectorized_fun(x)
        
    def derivate(self, x):
        """Compute the derivative of the ELU activation function for a given input or array of inputs.

        Args:
            x (float or array_like): Input value(s) to compute the derivative.

        Returns:
            float or ndarray: Computed derivative value(s).
        """
        if isinstance(x, float):
            return 1 if x > 0 else self.slope*np.exp(x)
        else:
            return self.vectorized_derivate(x)