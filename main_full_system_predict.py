import numpy as np
import json
import os

from pd_main_part.classifiers import get_classifier_by_name
from pd_main_part.preprocessors import get_preprocessor_by_name


def parse_args_for_train():
    # -------------------- initialize arguments ----------------------------------
    import argparse
    parser = argparse.ArgumentParser(description="Some description")

    parser.add_argument("-i", "--input_file", type=str, help="input file with input images for NN")

    parser.add_argument("-o", "--output_file", type=str, help="output file with NN answers")

    parser.add_argument("-c", "--config_file", type=str, help="config file with NN description")

    parser.add_argument("-n", "--number_of_samples", type=int, help="number of samples")
    # -------------------- parsing arguments ----------------------------------
    args = parser.parse_args()

    input_file = args.input_file

    output_file = args.output_file

    config_file = args.config_file

    n_samples = args.number_of_samples

    for arg in (input_file, output_file, config_file, n_samples):
        if arg is None:
            raise Exception('arg is None')

    return input_file, output_file, config_file, n_samples


def main():
    with open(os.path.abspath('config_full_system.json')) as config_fp:
        config_dict = json.load(config_fp)

        classifier = get_classifier_by_name(
            config_dict['classifier']['name'],
            config_dict['classifier']['args'])

        input_file, output_file, config_file, n_samples = parse_args_for_train()

        shape = (n_samples, 256, 256, 3)

        x_data = np.memmap(input_file, shape=shape, offset=128)

        if config_dict['preprocessor']['use']:
            preprocess_function = get_preprocessor_by_name(
                config_dict['preprocessor']['name'],
                config_dict['preprocessor']['args']).preprocess()
            x_data = np.array(map(lambda x: preprocess_function(x), x_data))

        y_data = classifier.predict(x_data)

        np.save(output_file, y_data)


if __name__ == '__main__':
    main()
