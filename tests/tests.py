import json
import unittest

from keras_preprocessing.image import ImageDataGenerator

from pd_lib.conv_network import get_model_by_name
from pd_main_part.classifiers import PlantDetector, get_classifier_by_name
from pd_lib.data_maker import json_big_load
import pandas as pd
import os

os.chdir(os.getcwd() + "/..")


class PlantDetectorTest(unittest.TestCase):
    def setUp(self):
        with open('config_full_system.json', encoding='utf-8') as config_fp:
            green_threshold = json.load(config_fp)['classifier']['args']['green_threshold']
            self.plant_detector = PlantDetector(green_threshold=green_threshold)

    def test_accuracy_100(self):
        classes, _, x_data, y_data = json_big_load('Datasets/ds_tests/ds_plant_detector/ds_plant_detector.json')
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


class ClassifierTest(unittest.TestCase):
    def setUp(self):
        with open('config_full_system.json', encoding='utf-8') as config_fp:
            config_dict = json.load(config_fp)
            classifier_name = config_dict['classifier']['name']
            classifier_args = config_dict['classifier']['args']
            self.classifier = get_classifier_by_name(classifier_name, classifier_args)

        with open('Datasets/ds_classifier/ds_7500/big_ds.json', encoding='utf-8') as train_json_fp:
            test_samples = json.load(train_json_fp)

            test_samples['df'] = pd.DataFrame(test_samples['dataframe'])
            test_samples['df'] = test_samples['df'].replace(['фитофтороз', 'здоровый куст'],
                                                            ['афитофтороз', 'яздоровый куст'])
            del test_samples['dataframe']
            batch_size = 16
            self.steps = len(test_samples['df']) / batch_size

            self.test_generator = ImageDataGenerator(
                #############################
                # effects needed only to fit
                #############################
            ) \
                .flow_from_dataframe(
                shuffle=False,
                dataframe=test_samples['df'],
                x_col='id',
                y_col='label',
                target_size=(256, 256),
                color_mode='rgb',
                batch_size=batch_size
            )

    def test_accuracy(self):
        test_loss, test_acc = self.classifier.model.evaluate_generator(
            generator=self.test_generator,
            steps=self.steps
        )

        print('test acc = %.2f' % test_acc)
        assert test_acc > 0.95

    def test_models_time_comparison(self):
        from matplotlib import pyplot as plt
        import time
        models_names = ('Xception', 'DenseNet121', 'MobileNetV2')
        for model_name in models_names:
            model, _ = get_model_by_name(model_name, input_shape=(256, 256, 3), output_shape=4)
            time_list = []
            steps_list = (1_000, 5_000, 10_000)
            for samples_size in steps_list:
                start_time = time.time()
                results = model.predict_generator(
                    generator=self.test_generator,
                    steps=samples_size
                )
                time_list.append(time.time() - start_time)
            plt.plot(steps_list, time_list)
        plt.xlabel('size')
        plt.ylabel('time')
        plt.title('models comparison')
        plt.legend(models_names, loc='upper_right')
        plt.show()
