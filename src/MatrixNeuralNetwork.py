import MLModel
import numpy as np
import copy

class MatrixNeuralNetwork(MLModel):
    
    default_hyperparameters = []
    default_values = {}

    def __init__(self, structure:list[int], n_input:int, seed:int = None):
        '''
        
        structure: [10, sigmoid, ['normal', {loc:0, scale:0.5}]] ---> only output layer, sigmoid act fun, weight init range and probability
            distribution of the choice
            
        structure: [[5, relu, ['uniform', {low:-1, high:1}]],
                     [15, relu, ['uniform', {low:-1, high:1}]],
                     [10, identity, ['uniform', {low:-1, high:1}]]] --> 2 hidden layer and 1 output
        '''
        
        random_generator = np.random.default_rng(seed)
        
        self.layers = []
        prec_input = n_input
        for elem in structure:
            weight = getattr(random_generator, elem[2][0])(**elem[2][1], size=prec_input*elem[0])
            layer = np.array([])
        
        for layer in structure:
        return
    
    def set_hyperparameters(self, hyper_param: dict):

        pass
    
    def predict(self, patterns: np.ndarray) -> np.ndarray:

        pass

    def __predict_for_training(self, patterns: np.ndarray) -> np.ndarray:
        
        return
    
    def train(self, training_set: np.ndarray, validation_set: np.ndarray, metrics_list: list, verbose: bool) -> dict:

        pass

    


    