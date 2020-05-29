import numpy as np
from pd_lib.keras_addition_ import get_full_model


class ClassifierInterface():
    def predict(self, x_arr):
        raise Exception('Method predict need to be overrided')


###########################################################################
# ---------------------- realizations -------------------------------------
###########################################################################
class PlantDetector():
    def __init__(self, green_threshold):
        self.green_threshold = green_threshold

    def get_plant_indexes(self, x_arr):
        plant_indexes = []
        for i, x in enumerate(x_arr):
            if self.get_green_content(x) > self.green_threshold:
                plant_indexes.append(i)

        return plant_indexes

    def get_green_content(self, x):
        count_green = 0
        x_reshaped = x.reshape((x.shape[0] * x.shape[1], x.shape[2]))
        for pixel in x_reshaped:
            pixel = list(map(int, pixel.tolist()))
            if sum(pixel) != 0:
                green_pixel = 100 * (pixel[1] / sum(pixel))
                blue_pixel = 100 * (pixel[2] / sum(pixel))
                red_pixel = 100 * (pixel[0] / sum(pixel))
                if green_pixel > red_pixel and green_pixel > blue_pixel and green_pixel > 35:
                    count_green += 1
        green_percent = round((count_green / (x_reshaped.shape[0])), 2)
        return green_percent


class CNNClassifier(ClassifierInterface):
    def __init__(self, json_path, h5_path, classes, green_threshold):
        self.model = get_full_model(json_path=json_path, h5_path=h5_path)
        self.classes = classes
        self.plant_detector = PlantDetector(green_threshold=green_threshold)

    def predict(self, x_arr):
        plant_indexes = self.plant_detector.get_plant_indexes(x_arr)

        nn_answers = self.model.predict(x_arr[plant_indexes])

        res = np.zeros(shape=(len(x_arr), *nn_answers.shape[1:]))

        nn_iter_answers = iter(nn_answers)
        for i in range(len(res)):
            if i in plant_indexes:
                res[i] = nn_iter_answers.__next__()
            else:
                res[i] = -1

        return res


###############################################################################
# ------------------------ get ------------------------------------------------
###############################################################################
def get_classifier_by_name(name, args):
    if name.lower() == 'cnn':
        return CNNClassifier(**args)
    else:
        raise Exception('Undefined cluster name = %s' % name)
