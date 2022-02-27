import time
import os

def time_keeper(logger, start):
    end = time.time()
    elapsed = end - start
    logger.info('The total time is: {}'.format(time_converter(elapsed)))

def time_converter(elapsed):
    hour = int(elapsed / 3600)
    left = elapsed % 3600
    minute = int(left / 60)
    seconds = left % 60
    return '{} h {} m {} s'.format(hour, minute, seconds)

'''
Get number of trainable parameters
'''
def count_parameters(model, verbose=False):
    if not verbose:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total Trainable Params: {total_params}")
        return total_params
    from prettytable import PrettyTable
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def create_folder(path, model_name=None):
    if model_name:
        if os.path.exists(path+model_name):
            return
        else:
            os.mkdir(path+model_name)
    else:
        if not os.path.exists(path):
            os.makedirs(path)