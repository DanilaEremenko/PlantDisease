import json
import unittest
from pd_main_part.classifiers import PlantDetector
from pd_lib.data_maker import json_big_load


class PlantDetectorTest(unittest.TestCase):
    def setUp(self):
        with open('../config_full_system.json', encoding='utf-8') as config_fp:
            green_threshold = json.load(config_fp)['classifier']['args']['green_threshold']
            self.plant_detector = PlantDetector(green_threshold=green_threshold)

    def test_accuracy_100(self):
        classes, _, x_data, y_data = json_big_load('../Datasets/ds_tests/ds_plant_detector/ds_plant_detector.json')
        plant_indexes = self.plant_detector.get_plant_indexes(x_arr=x_data)
        right_ans = 0
        for i in plant_indexes:
            if list(y_data[i]) == [0, 1]:
                right_ans += 1

        if right_ans == classes['PLANT']['num']:
            print('right = plant_num = %d' % (right_ans))
            assert True
        else:
            print('right = %d, plant_num = %d' % (right_ans, classes['PLANT']['num']))
            assert False
