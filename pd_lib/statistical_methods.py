import json
import numpy as np
from pd_lib import data_maker as dmk


class SimpleStatistical():
    def __init__(self, x, classes):
        self.classes = classes
        for key in classes.keys():
            for i, chanel in enumerate(['R', 'G', 'B']):
                self.classes[key]['%s_std' % chanel] = classes[key]['x'][:, :, :, i].std()
                self.classes[key]['%s_mean' % chanel] = classes[key]['x'][:, :, :, i].mean()

                self.classes[key]['%s_stds' % chanel] = \
                    list(map(lambda x_ex: x_ex.std(), classes[key]['x'][:, :, :, i]))
                self.classes[key]['%s_means' % chanel] = list(
                    map(lambda x_ex: x_ex.mean(), classes[key]['x'][:, :, :, i]))

                self.classes[key]['%s_std_min' % chanel] = min(self.classes[key]['%s_stds' % chanel])
                self.classes[key]['%s_std_max' % chanel] = max(self.classes[key]['%s_stds' % chanel])
                self.classes[key]['%s_mean_min' % chanel] = min(self.classes[key]['%s_means' % chanel])
                self.classes[key]['%s_mean_max' % chanel] = max(self.classes[key]['%s_means' % chanel])

            del classes[key]['x']

    def predict(self, x):
        diff_min = 999999999999

        answers = []
        for x_ex in x:
            for key in self.classes.keys():
                R_mean_diff = abs(x_ex[:, :, 0].mean() - self.classes[key]['R_mean'])
                G_mean_diff = abs(x_ex[:, :, 1].mean() - self.classes[key]['G_mean'])
                B_mean_diff = abs(x_ex[:, :, 2].mean() - self.classes[key]['B_mean'])
                R_std_diff = abs(x_ex[:, :, 0].std() - self.classes[key]['R_std'])
                G_std_diff = abs(x_ex[:, :, 1].std() - self.classes[key]['G_std'])
                B_std_diff = abs(x_ex[:, :, 2].std() - self.classes[key]['B_std'])
                cur_diff = (R_std_diff + R_mean_diff + +G_std_diff + G_mean_diff + +B_std_diff + B_mean_diff) / 6
                if cur_diff < diff_min:
                    diff_min = cur_diff
                    class_name = key
            answers.append(self.classes[class_name]['value'])
        return answers


############################################################
# --------------------- test --------------------------------
############################################################
def test():
    with open('config_fit_CNN.json') as config_fp:
        config_dict = json.load(config_fp)

    classes, img_shape, x_train, y_train = \
        dmk.json_big_load(config_dict['data']['train_json'])

    for key in classes.keys():
        classes[key]['x'] = np.empty(0, dtype='uint8')

    for x, y in zip(x_train, y_train):
        for key in classes.keys():
            if (classes[key]['value'] == y).all():
                classes[key]['x'] = np.append(classes[key]['x'], x)
                break

    for key in classes.keys():
        classes[key]['x'].shape = (classes[key]['num'], *x_train.shape[1:4])

    model = SimpleStatistical(
        x=x_train,
        classes=classes
    )

    right_ans = 0
    for y, y_answer in zip(y_train, model.predict(x_train)):
        if (y == y_answer).all():
            right_ans += 1

    print('acc = %.2f' % (right_ans / x_train.shape[0]))


if __name__ == '__main__':
    test()
