import json
import os

import matplotlib.pyplot as plt

for nn_dir in os.listdir('../models/'):
    for nn_file in os.listdir(nn_dir):
        if nn_file[-12:] == 'history.json':
            with open(nn_dir + '/' + nn_file) as json_fp:
                full_history = json.load(json_fp)
                title = nn_file.replace('model_', '').replace('_history.json', '')
                plt.plot(full_history['accuracy'])
                plt.plot(full_history['val_accuracy'])
                plt.legend(('train', 'test'))
                plt.title(title + ' accuracy history')
                plt.show()

                plt.plot(full_history['loss'])
                plt.plot(full_history['val_loss'])
                plt.legend(('train', 'test'))
                plt.title(title + ' loss history')
                plt.show()
