import argparse
from .addition import get_full_model


def parse_args_for_train():
    parser = argparse.ArgumentParser(description="Some description")

    parser.add_argument("-j", "--json_list", type=str, action="append",
                        help="json with train data")

    parser.add_argument("-w", "--weights_path", type=str, help="file with weigths of NN")

    parser.add_argument("-s", "--structure_path", type=str, help="file with structure of NN")

    args = parser.parse_args()

    weights_path = args.weights_path

    structure_path = args.structure_path

    model = get_full_model(json_path=structure_path, h5_path=weights_path, verbose=True)

    json_list = args.json_list

    if json_list is None:
        raise Exception("Nor one json file passed")

    return model, json_list
