import json
import os,sys


def setting_annotations_filename_paths(jsonInput,path,jsonoutput):
    """
    This function changes each filename path of a given json file, preserving the basename and the last directory name
    and substituting all the precedent path with a given one.
    :param jsonInput: json input file
    :param path: new path to be attached to the /lastdirname/filename.jpg part
    :param jsonoutput: updated json
    :return:
    """
    # Extracting data from input json file
    with open(jsonInput) as data_file:
        data = json.load(data_file)

    for dict_it in data:
        # Extracting path parts
        lasttwo = os.path.join(os.path.basename(os.path.dirname(data[0]['filename'])),\
                           os.path.basename(data[0]['filename']))
        newpath = os.path.join(path, lasttwo)
        # Updating path
        dict_it['filename']=newpath

    # Write on output json
    with open(jsonoutput,'w') as f:
        json.dump(data,f,indent=4)

#-----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    jsoninput = sys.argv[1]
    path = sys.argv[2]
    jsonoutput = sys.argv[3]
    setting_annotations_filename_paths(jsoninput,path,jsonoutput)

