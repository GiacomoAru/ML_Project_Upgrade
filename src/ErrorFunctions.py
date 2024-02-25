import numpy as np
import math

'''
The collections of all implemented error functions
'''

def mean_euclidean_error(outputs:np.ndarray, targets:np.ndarray):
    '''
    Calculates the Mean Euclidean Error for a given learning set of patterns
    
    Parameters
    ----------
    outputs: np.ndarray
        the predicted NN's outputs
    targets: np.ndarray
        the targhet values

    Returns
    -------
    return: float
        the Mean Euclidean Error value    
    '''

    sum = 0
    for diff in (outputs-targets):
        sum += math.sqrt(np.sum(diff**2))

    return sum/len(outputs)

def mean_squared_error(outputs:np.ndarray, targets:np.ndarray):
    '''
    Calculates the Mean Squared Error for a given learning set of patterns
    
    Parameters
    ----------
    outputs: np.ndarray
        the predicted NN's outputs
    targets: np.ndarray
        the targhet values

    Returns
    -------
    return: float
        the Mean Squared Error value
    '''

    error = np.mean(np.sum((targets-outputs)**2, axis=1))
    if np.isnan(error):
        print("targets:", targets)
        print("outs:", outputs)

    return error

def accuracy(targets: np.ndarray, outputs: np.ndarray, threshold: float = 0.5) -> float:
    '''
    Calculates the accuracy for a given learning set of patterns, computed as:
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
    
    Parameters
    ----------
    outputs: np.ndarray
        the predicted NN's outputs, boolean values or the probability of the positive class
    targets: np.ndarray
        the targhet values, trictly boolean values

    Returns
    -------
    return: float
        the accuracy value
    '''
    # Check if the arrays have the same length
    if len(targets) != len(outputs):
        raise ValueError("Arrays must have the same length.")
    
    # Convert probabilities to binary labels using the threshold
    outputs = (outputs >= threshold).astype(int)
    
    # Compute the number of correctly predicted labels
    correct_predictions = np.sum(targets == outputs)
    
    # Compute the accuracy
    accuracy = correct_predictions / len(targets)
    
    return accuracy