import matplotlib.pyplot as plt
import numpy as np


def plot_history(history, save_path, save=False, show=True):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [1000$]')
    plt.plot(history.epoch, np.array(history.history['loss']),
             label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['acc']),
             label='Accuracy')
    plt.legend()

    if show:
        plt.show()
    if save:
        plt.savefig(save_path, dpi=200)


def plot_history_separte(history, save_path_acc, save_path_loss, acc='acc', loss='loss', save=False, show=True):
    plot_graphic(x=history.epoch, y=np.array(history.history[loss]), x_label='epochs', y_label='loss',
                 title=loss + ' history', save_path=save_path_loss, save=save, show=show)
    plot_graphic(x=history.epoch, y=np.array(history.history[acc]), x_label='epochs', y_label='accuracy',
                 title=acc + ' history', save_path=save_path_acc, save=save, show=show)


def plot_history_separate_from_dict(history_dict, save_path_acc, save_path_loss, acc='acc', loss='loss',
                                    save=False, show=True):
    epochs = []
    for i in range(0, history_dict['acc'].__len__()): epochs.append(i)

    plot_graphic(x=epochs, y=np.array(history_dict[loss]), x_label='epochs', y_label='loss',
                 title=loss + ' history', save_path=save_path_loss, save=save, show=show)
    plot_graphic(x=epochs, y=np.array(history_dict[acc]), x_label='epochs', y_label='accuracy',
                 title=acc + ' history', save_path=save_path_acc, save=save, show=show)


def plot_graphic(x, y, x_label, y_label, title, save_path=None, save=False, show=False, close_plt=True):
    plt.plot(x, y)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)

    if save:
        if save_path == None:
            raise ValueError('save_path == None with save == True')
        plt.savefig(save_path, dpi=200)

    if show:
        plt.show()

    if close_plt:
        plt.close()
