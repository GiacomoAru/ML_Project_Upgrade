import numpy as np
import pandas as pd
import random
import pathlib

# -- create data structures --
def create_dataset(n_items, n_input, input_range, output_functions, seed):
    '''
    Create a dummy dataset for early training tests given the inputs
    '''
    random.seed(seed)

    n_output = len(output_functions)
    x = np.ndarray((n_items, n_input + n_output))

    for i in range(n_items):
        for l in range(n_input):
            x[i,l] = random.uniform(input_range[0], input_range[1])

        for l, fun in enumerate(output_functions):
            
            x[i, n_input + l] = fun(x[i][:n_input])
            #print(x[i][:n_input], fun(x[i][:n_input]), x[i, l])

    return pd.DataFrame(x, columns = ['input_' + str(i + 1) for i in range(n_input)] + ['output_' + str(i + 1) for i in range(n_output)])


def monk_to_csv():
    '''
    used to convert the monk datasets in csv
    '''
    for j in range(1,4):
        datas_tr = {'input_1':[],
                'input_2':[],
                'input_3':[],
                'input_4':[],
                'input_5':[],
                'input_6':[],
                'output_1':[]}
        datas_ts = {'input_1':[],
                'input_2':[],
                'input_3':[],
                'input_4':[],
                'input_5':[],
                'input_6':[],
                'output_1':[]}
        
        for line in open(pathlib.Path(__file__).parent.parent.joinpath('data\\monks\\monks-' + str(j) + '.test')):
            line_divided = line.split(' ')[1:]
            datas_ts['output_1'].append(line_divided[0])
            for i in range(1,7):
                datas_ts['input_' + str(i)].append(line_divided[i])

        for line in open(pathlib.Path(__file__).parent.parent.joinpath('data\\monks\\monks-' + str(j) + '.train')):
            line_divided = line.split(' ')[1:]
            datas_tr['output_1'].append(line_divided[0])
            for i in range(1,7):
                datas_tr['input_' + str(i)].append(line_divided[i])

        df = pd.DataFrame(datas_tr)
        df.to_csv(pathlib.Path(__file__).parent.parent.joinpath('data\\monks_csv\\monks_tr_' + str(j) + '.csv'))
        df = pd.DataFrame(datas_ts)
        df.to_csv(pathlib.Path(__file__).parent.parent.joinpath('data\\monks_csv\\monks_ts_' + str(j) + '.csv'))

