import argparse


def parse_json_list_from_cmd():
    parser = argparse.ArgumentParser(description="Some description")

    parser.add_argument("-j", "--json_list", type=str, action="append",
                        help="json with train data")

    args = parser.parse_args()

    json_list = args.json_list

    if json_list is None:
        raise Exception("Nor one json file passed")

    return json_list
