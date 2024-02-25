from MLModel import *
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
            
            unit_number = elem[0]
            distribution_str = elem[2][0]
            distribution_args = elem[2][1]
            activation_fun = copy.deepcopy(elem[1])
            
            weight = getattr(random_generator, distribution_str)(**distribution_args, size=prec_input*unit_number)
            layer = np.array([weight[i:i + unit_number] for i in range(0, len(weight), unit_number)])
            prec_input = unit_number
            
            self.layers.append([layer, activation_fun])

        return
    
    def __str__(self):
        ret = ''
        for i, elem in enumerate(self.layers):
            ret += 'layer_' + str(i) + '\n'
            ret +=  str(elem[1]) + ' : ' + str(elem[0].shape) + '\n'
            ret += str(elem[0]) + '\n' + '-'*100 + '\n'
            
        return ret    
        
        
        
    def set_hyperparameters(self, hyper_param: dict):

        pass
    
    def predict(self, patterns: np.ndarray) -> np.ndarray:
        
        tmp_values = patterns
        for weights, act_fun in self.layers:
            # weight
            tmp_values = np.matmul(tmp_values, weights)
            # activation fun
            tmp_values = act_fun.compute(tmp_values)
        
        return tmp_values

    def __predict_for_training(self, patterns: np.ndarray) -> np.ndarray:
        
        return
    
    def train(self, training_set: np.ndarray, validation_set: np.ndarray, metrics_list: list, verbose: bool) -> dict:

        pass

    


    