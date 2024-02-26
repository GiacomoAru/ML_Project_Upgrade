from MLModel import *
import numpy as np
import copy
import datetime 
from ErrorFunctions import *

class MatrixNeuralNetwork(MLModel):
    
    default_hyperparameters_values = {
            'batch_size': 64, 
            'min_epochs': 0,
            'max_epochs': 512, 
            
            'retraining_error_target': -1.0, # off
            'retraining_error_tol': 0.1,
            'patience': 5, 
            'error_increase_tolerance': 0.0001, 
        
            'lambda_tikhonov': 0.0, # off
            
            'learning_rate': 0.01,
            'learning_rate_adjust':True,
            'lr_decay_tau': 0, # off
            'lr_decay_multiplier': 0.0, # off
            'alpha_momentum': 0.0, # off
            'nesterov_momentum': False,

            'adamax': False,
            'adamax_learning_rate': 0.01,
            'exp_decay_rate_1': 0.9,
            'exp_decay_rate_2': 0.999,
            
            'collect_data': True, 
            'collect_batch_data': False}

    def __init__(self, structure:list[int], n_input:int, seed:int = None):
        '''
        
        structure: [10, sigmoid, ['normal', {loc:0, scale:0.5}]] ---> only output layer, sigmoid act fun, weight init range and probability
            distribution of the choice
            
        structure: [[5, relu, ['uniform', {low:-1, high:1}]],
                     [15, relu, ['uniform', {low:-1, high:1}]],
                     [10, identity, ['uniform', {low:-1, high:1}]]] --> 2 hidden layer and 1 output
        '''
        
        # inizializate the layers and the weights
        random_generator = np.random.default_rng(seed)
        self.input_size = n_input
        self.output_size = structure[-1][0]
        self.layers = []
        prec_input = n_input
        for elem in structure:
            
            unit_number = elem[0]
            distribution_str = elem[2][0]
            distribution_args = elem[2][1]
            activation_fun = copy.deepcopy(elem[1])
            
            # + 1 for the bias weight in every neuron
            weight = np.array(getattr(random_generator, distribution_str)(**distribution_args, size = (prec_input + 1)*unit_number))
            layer = np.reshape(weight, (prec_input + 1, unit_number))
            prec_input = unit_number
            
            self.layers.append([layer, activation_fun])

        # set default hyperparameters for training
        self.set_hyperparameters()
        return
    
    def __str__(self):
        ret = ''
        for i, elem in enumerate(self.layers):
            ret += 'layer_' + str(i) + '\n'
            ret +=  str(elem[1]) + ' : ' + str(elem[0].shape) + '\n'
            ret += str(elem[0]) + '\n' + '-'*100 + '\n'
            
        return ret    
        
    def set_hyperparameters(self,
              batch_size:int = 64, # batch mode
              min_epochs: int = 0,
              max_epochs:int = 512, 
              
              retraining_error_target: float = -1.0, # off
              retraining_error_tol:float = 0.1,
              patience: int = -1, 
              error_increase_tolerance:float = 0.0001, 
            
              lambda_tikhonov:float = 0.0, # off
              
              learning_rate:float = 0.01,
              learning_rate_adjust:bool = True,
              lr_decay_tau:int = 0, # off
              lr_decay_multiplier:float = 0.0, # off
              alpha_momentum:float = 0.0, # off
              nesterov_momentum:bool = False,

              adamax:bool = False,
              adamax_learning_rate:float = 0.01,
              exp_decay_rate_1:float = 0.9,
              exp_decay_rate_2:float = 0.999,
              
              collect_data:bool=True,
              collect_prediction_data:bool=False,
              collect_weights_data:bool=False, 
              collect_batch_data:bool=False):
        
        self.batch_size = batch_size
        self.min_epochs = min_epochs
        self.max_epochs = max_epochs
        self.retraining_error_target = retraining_error_target
        self.retraining_error_tol = retraining_error_tol
        self.patience = patience
        self.error_increase_tolerance = error_increase_tolerance
        self.lambda_tikhonov = lambda_tikhonov
        self.learning_rate = learning_rate
        self.learning_rate_adjust = learning_rate_adjust
        self.lr_decay_tau = lr_decay_tau
        self.lr_decay_multiplier = lr_decay_multiplier
        self.alpha_momentum = alpha_momentum
        self.nesterov_momentum = nesterov_momentum
        self.adamax = adamax
        self.adamax_learning_rate = adamax_learning_rate
        self.exp_decay_rate_1 = exp_decay_rate_1
        self.exp_decay_rate_2 = exp_decay_rate_2
        self.collect_data = collect_data
        self.collect_prediction_data = collect_prediction_data
        self.collect_weights_data = collect_weights_data
        self.collect_batch_data = collect_batch_data
        
        if self.learning_rate_adjust:
            self.true_learning_rate = self.learning_rate/self.batch_size
        self.eta_tau = self.learning_rate*self.lr_decay_multiplier
    
    def predict(self, patterns: np.ndarray) -> np.ndarray:
        
        tmp_values = patterns
        for weights, act_fun in self.layers:
            # adding bias ones
            tmp_values_bias = np.ones((tmp_values.shape[0],tmp_values.shape[1]+1))
            tmp_values_bias[:,:-1] = tmp_values
            # weight
            tmp_values = np.matmul(tmp_values_bias, weights)
            # activation fun
            tmp_values = act_fun.compute(tmp_values)
        
        return tmp_values

    def predict_for_training(self, patterns: np.ndarray) -> np.ndarray:
        
        self.layers_output = []
        self.layers_net = []
        
        tmp_values = patterns
        for weights, act_fun in self.layers:
            # adding bias ones
            tmp_values_bias = np.ones((tmp_values.shape[0],tmp_values.shape[1]+1))
            tmp_values_bias[:,:-1] = tmp_values
            
            # weight
            tmp_values = np.matmul(tmp_values_bias, weights)
            self.layers_net.append(tmp_values) # net
            # activation fun
            tmp_values = act_fun.compute(tmp_values)
            self.layers_output.append(tmp_values) # o
        return tmp_values
           
    def backpropagate(self, target: np.ndarray):
        
        self.layers_delta_error = []
        
        # output layer
        a = self.layers[-1][1].derivate(self.layers_net[-1])
        b = (target - self.layers_output[-1])
        self.layers_delta_error.append(np.multiply(a,b))
        
        for i in range(1, len(self.layers)):
            a = self.layers[-(i+1)][1].derivate(self.layers_net[-(i+1)])
            
            b = np.empty(a.shape)
            for row_index in range(a.shape[0]):
                c = np.broadcast_to(self.layers_delta_error[-1][row_index],((self.layers[-i][0].shape[0] - 1, self.layers[-i][0].shape[1])))
                
                c = np.multiply(c, self.layers[-i][0][:-1,:])
                b[row_index] = np.sum(c, axis=1)
            self.layers_delta_error.append(b)
            
        self.layers_delta_error = reversed(self.layers_delta_error)
        return
    
    def __weights_update(self):
        
        return
    
    def train(self, training_set: np.ndarray, validation_set: np.ndarray = None, additional_data_set: np.ndarray=None, metrics_list: list = [], verbose: bool = True) -> dict:
        
        # simple check to adjust bad input values
        if self.batch_size > training_set_length: self.batch_size = training_set_length
        if self.patience <= 0:
            exhausting_patience = 1
        else:
            exhausting_patience = self.patience
            
        # initializing every variables with the correct value
        self.epochs = 0
        last_error_increase_percentage = -1
        training_set_length = len(training_set)
        last_val_error = np.inf
        
        
        
        # variables used in retrain to stop at the right training error
        NOT_retraining_stop = True
        retraining_error_target = self.retraining_error_target + self.retraining_error_target*self.retraining_error_tol
        
        # initializing the dict where collect stats of the training
        hyperparameters = type(self).default_hyperparameters_values.copy()
        for key in hyperparameters:
            hyperparameters[key] = getattr(self, key)
            
        stats = {
            # -- input stats --
            'hyperparameters':hyperparameters,

            # -- retraining stats --
            'tr_error_when_to_stop' : 0,
                        
            # -- training stats --
            # epoch stats
            'epochs':0
        }
            
        if self.collect_data: 
            # take training time for the batch
            stats['total_train_time'] = datetime.datetime.now() - datetime.datetime.now() # zero initialized datetime
            stats['mean_epoch_train_time'] = 0 # to compute mean
            
            if self.collect_weights_data: 
                stats['layers_weights'] = {}
                if self.collect_batch_data:
                    stats['layers_weights_batch'] = {}
                
                for i in range(len(self.layers)):
                    # epoch stats
                    stats['layers_weights']['layer_' + str(i)] = []
                    if self.collect_batch_data:
                        # batch stats
                        stats['layers_weights_batch']['layer_' + str(i)] = []
            
            if self.collect_prediction_data:
                stats['tr_pred'] = []
                stats['val_pred'] = []
                stats['add_pred'] = []
                if self.collect_batch_data:
                    stats['tr_pred_batch'] = []
                    stats['val_pred_batch'] = []
                    stats['add_pred_batch'] = []
            
            # initializing lists to collect data
            stats['additional_metrics'] = [mes.__name__ for mes in metrics_list]
            for mes in metrics_list:
            # epoch stats
                stats['tr_' + mes.__name__] = []
                if not(validation_set is None):
                    stats['val_' + mes.__name__] = []
                if not(additional_data_set is None):
                    stats['add_' + mes.__name__] = []
                
            if self.collect_batch_data:
                for mes in metrics_list:
                    # batch stats
                    stats['tr_batch_' + mes.__name__] = []
                    if not(validation_set is None):
                        stats['val_batch_' + mes.__name__] = []
                    if not(additional_data_set is None):
                        stats['add_batch_' + mes.__name__] = []

        start_time = datetime.datetime.now()
        if verbose: print('starting values: ', hyperparameters)
           
        # start training cycle
        batch_index = 0
        while (self.epochs < self.max_epochs) and (exhausting_patience > 0) and NOT_retraining_stop:
            # batch iteration
            
            if batch_index + self.batch_size > training_set_length:
                batch = np.append(training_set[batch_index:], 
                                  training_set[: batch_index + self.batch_size], axis=0)
                
            else:
                batch = training_set[batch_index: batch_index + self.batch_size]
            batch_index += self.batch_size
            
            self.__predict_for_training(batch[:self.input_size])
            self.__backpropagate(batch[self.input_size:])
            self.__weights_update()
            
            # stats for every batch
            if self.collect_data and self.collect_batch_data:
                # computing errors
                for mes in metrics_list:
                    stats['tr_batch_' + mes.__name__].append(mes(self.predict(training_set[:,:self.input_size]), training_set[:,self.input_size:]))
                    if not(validation_set is None):
                        stats['val_batch_' + mes.__name__].append(mes(self.predict(validation_set[:,:self.input_size]), validation_set[:,self.input_size:]))
                    if not(additional_data_set is None):
                        stats['add_batch_' + mes.__name__].append(mes(self.predict(additional_data_set[:,:self.input_size]), additional_data_set[:,self.input_size:]))
                
                # storing unit's weights
                if self.collect_weights_data:
                    for i, layer in enumerate(self.layers):
                        # epoch stats
                        stats['layers_weights_batch']['layer_' + str(i)].append(layer[0].copy())
                        
                if self.collect_prediction_data:
                    stats['tr_pred_batch'].append(self.predict(training_set[:,:self.input_size]))
                    if not(validation_set is None):
                        stats['val_pred_batch'].append(self.predict_array(validation_set[:,:self.input_size]))
                    if not(additional_data_set is None):
                        stats['add_pred_batch'].append(self.predict(additional_data_set[:,:self.input_size]))

            # after every batch is checked if an epoch is passed
            if batch_index >= training_set_length:
                # end of the epoch
                self.epochs += 1
                batch_index = batch_index%training_set_length

                # patience related computation, useful to check if to stop the training
                if not(validation_set is None) or self.retraining_error_target > 0:
                    tr_err = mean_squared_error(self.predict(training_set[:,:self.input_size]), training_set[:,self.input_size:])  
                    if tr_err < retraining_error_target:
                        NOT_retraining_stop = False
                        
                if not(validation_set is None):
                    # the training error is computed
                    val_err = mean_squared_error(self.predict(validation_set[:,:self.input_size]), validation_set[:,self.input_size:])          
                    
                    last_error_increase_percentage = (val_err - last_val_error)/last_val_error
                    if last_error_increase_percentage < 0:
                        stats['tr_error_when_to_stop'] = tr_err
                    last_val_error = val_err
                    
                    if self.patience > 0:
                        if self.epochs >= self.min_epochs and last_error_increase_percentage >= self.error_increase_tolerance:
                            exhausting_patience -= 1
                        else:
                            exhausting_patience = self.patience
                
                
                # stats for every epoch
                if self.collect_data:
                    # take training time for the epoch
                    end_time = datetime.datetime.now()   
                    stats['total_train_time'] += (end_time-start_time)

                    # computing every error and printing some information if verbose is True
                    if verbose: metrics_to_print = ''
                    
                    pred_tr = self.predict(training_set[:,:self.input_size])
                    pred_val = self.predict(validation_set[:,:self.input_size])
                    pred_add = self.predict(additional_data_set[:,:self.input_size])
                    
                    if self.collect_prediction_data:
                        stats['tr_pred'].append(pred_tr)
                        if not(validation_set is None):
                            stats['val_pred'].append(pred_val)
                        if not(additional_data_set is None):
                            stats['add_pred'].append(pred_add)
                    
                    if self.collect_weights_data:
                        for i, layer in enumerate(self.layers):
                            # epoch stats
                            stats['layers_weights']['layer_' + str(i)].append(layer[0].copy())
                            
                    for mes in metrics_list:     
                        tr_err = mes(pred_tr, training_set[:,self.input_size:])
                        stats['tr_' + mes.__name__].append(tr_err)
                        if not(validation_set is None):
                            val_err = mes(pred_val, validation_set[:,self.input_size:])
                            stats['val_' + mes.__name__].append(val_err)
                        if not(additional_data_set is None):
                            add_err = mes(pred_add, additional_data_set[:,self.input_size:])
                            stats['val_' + mes.__name__].append(add_err)
                            
                        if verbose: 
                            metrics_to_print += ' | ' +mes.__name__ + ': tr=' + str(tr_err) + ' val=' + str(val_err)

                    if verbose: print('[' + str(self.epochs) + '/' + str(self.max_epochs) + '] tr time:', end_time-start_time, metrics_to_print)
                    # take training time for the batch
                    start_time = datetime.datetime.now()
            
        # final stats gathering
        stats['epochs'] = self.epochs
        if self.collect_data:
            stats['mean_epoch_train_time'] = stats['total_train_time']/stats['epochs']
        return stats

    


    