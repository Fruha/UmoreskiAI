import os
import json

def save_in_json(data: dict, path: str, indent: int = 4) -> None:
    """
    Save data in Json object
    Arguments:
        data: dict
        path: path to save file
        indent = 4: count of spaces for indent
    """
    with open(path, 'w', encoding='utf-8') as outfile:
        json.dump(data, outfile, indent=4, ensure_ascii=False)


def load_from_json(path: str) -> dict:
    """
    Load data from json file
    Arguments:
        path: path to file
    Return:
        data from file\n
        None if file don't exists
    """
    if os.path.exists(path):
        with open(path, encoding='utf-8') as json_file:
            data = json.load(json_file)
            return data
    else:
        return {}