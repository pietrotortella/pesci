import os
import copy
import json


def clean_jsons(jsons_filepaths, to_prepend, save_filepath=None):
    """
    cleans many json files to work on a machine.
    It takes a list of json files, opens them, looks for
    the 'filename' keys and replaces the initial part of
    the value with new_dir

    :param jsons_filepaths: a list of strings
        where each string is the filepath of a json file
    :param to_prepend: a string
        the path to prepend to all
        (e.g. the path of the train folder, that contains all
        the folders ALB, BET, DOL, ecc.)
    :param save_filepath: None or a string
        if None, nothing happens, if string, the path
        where to save the clean informations

    :return:
        - new_json:
        a list of dictionaries
        which contains all the infos from the json files,
        cleaned to work on the local machine

        - bad_jsons: a sublist of filepaths, with the elements it
         was not able to process
    """
    new_json = []

    bad_jsons = []

    for jfilepath in jsons_filepaths:
        try:
            with open(jfilepath, 'r') as inf:
                now_json = json.load(inf, encoding='utf-8')
        except ValueError:
            bad_jsons.append(jfilepath)

        for now_dict in now_json:
            new_dict = copy.deepcopy(now_dict)

            now_filename = now_dict['filename']

            split_filepath = now_filename.split('/')
            last_part = '/'.join(split_filepath[-2:])

            new_path = os.path.join(to_prepend, last_part)
            new_dict['filename'] = new_path

            new_json.append(new_dict)

    if save_filepath is not None:
        with open(save_filepath, 'w') as out_file:
            json.dump(new_json, out_file, indent=4)

    return new_json, bad_jsons


if __name__ == '__main__':
    from config.folders import DATA_PATH, ANNOTATIONS_PATH

    json_names = os.listdir(ANNOTATIONS_PATH)

#    jsons_filepaths = [os.path.join(ANNOTATIONS_PATH, name) for name in json_names[-2:]]
    jsons_filepaths = [os.path.join(ANNOTATIONS_PATH, name)
                       for name in json_names if 'tortella' in name]

#    print(jsons_filepaths)

    to_prepend = os.path.join(DATA_PATH, 'train')

    newname = 'cleaned_tort.json'

    jj, bads = clean_jsons(jsons_filepaths, to_prepend, save_filepath=os.path.join(ANNOTATIONS_PATH, newname))

    if bads:
        print 'wasnt able to process the files ', bads

    print 'Final entry sample', jj[0]


