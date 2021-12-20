import json


def read_constants(path='./constants/dataset_constants.json'):
    constants = json.load(open(path, 'r'))
    return constants
