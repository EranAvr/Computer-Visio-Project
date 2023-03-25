import os
import torch

# My module-
import citrus_diseases_module as citrus


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device used: {device}\n')

if __name__ == '__main__':
    CONFIG = {'TRAIN': True,
              'TEST': True,
              'LOAD_PREV': False}


    #   Data Loaders:
    train_loader = citrus.train_loader
    test_loader = citrus.test_loader
    another_loader = citrus.foreign_loader

    #   Model setup:
    torch.cuda.empty_cache()
    my_model = citrus.MODEL
    optimizer = citrus.OPTIMIZER

    if CONFIG['LOAD_PREV']:
        state_dict_name = f'model_state_dict{666}.pth'
        if os.path.exists(state_dict_name):
            print(f'Loading model state dict: {state_dict_name}')
            sd_path = os.path.join(os.getcwd(), state_dict_name)
            my_model.load_state_dict(torch.load(sd_path))
        else:
            print(f'Could not find file: {state_dict_name}')
            exit(-1)
    if CONFIG['TRAIN']:
        print('Training model.')
        citrus.train(my_model, optimizer, data_loader=train_loader)
    if CONFIG['TEST']:
        print('Testing model.')
        citrus.test(my_model, data_loader=test_loader)
