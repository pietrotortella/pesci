##IMPORTING REQUIRED PACKAGES
import json
from scipy import misc
import sys
import matplotlib.pyplot as plt


def annotations(j_path,mypath):

##THIS FUNCTION TAKE IN INPUT j_path (the json file absolute path) and my_path (the absolute path of folder that contains images)
##IT RETURNS IN OUTPUT A LIST OF DICTIONARIES that eachone contains all annotations of one image



#read the json file from path

    jsn_file = open(j_path)

    data = json.load(jsn_file)


    def rect(image,annot,label):
        ''' returns the subimage of @image correspoding to the rectangle annotation @annot'''
        #if it works on "fish" it calculates as well absolute coordinates of window that contains the fish
        #these coordinates are useful tu calculate the relative coordinates of point annotations in the fish window
        x = int(annot["x"])
        h = int(annot["height"])
        y = int(annot["y"])
        w = int(annot["width"])
        abs_cord = [y,x]

        return (abs_cord,image[y:y + h, x:x + w]) if label == "fish" else image[y:y + h, x:x + w]

    def pnt(annot,abs_coordinates):
        ''' returns the RELATIVES coordinates of the point annotation @annot'''
        x = int(annot["x"])-abs_coordinates[1]
        y = int(annot["y"])-abs_coordinates[0]
        return [y,x]

##filenames is a list that contains the absolute path of each image
    filenames = [data[i]["filename"] if "/home" in data[i]["filename"]  else mypath + data[i]["filename"][-14:] for i in
                     range(len(data))]  # list of paths of all images contained in the json file

    output_list = []

##this cycle puts all notations in the list that the function need to returns
    for k in range(len(data)):
            d = {}
            relat_coord = []
            fish_dict = filter(lambda select: select['class'] == 'fish', data[k]["annotations"])[0]
            img = misc.imread(filenames[k])
            coord, rect_img = rect(img, fish_dict, "fish")
            relat_coord.append(coord)
            d.update({"fish": rect_img})

            for i in range(len(data[k]["annotations"])):

                v = data[k]["annotations"][i]["class"]


                if  v == 'non_fish':
                    app = data[k]["annotations"][i]
                    rect_img = rect(img, app, "non_fish")
                    d.update({"non_fish": rect_img})

                elif v =='head':
                    app = data[k]["annotations"][i]
                    pc = pnt(app,relat_coord[0])
                    d.update({"head": pc})

                elif v =='tail':
                    app = data[k]["annotations"][i]
                    pc = pnt(app, relat_coord[0])
                    d.update({"tail": pc})
                elif v =='up_fin':  # default, could also just omit condition or 'if True'
                    app = data[k]["annotations"][i]
                    pc = pnt(app, relat_coord[0])
                    d.update({"up_fin": pc})
                elif v=='low_fin':
                    app = data[k]["annotations"][i]
                    pc = pnt(app, relat_coord[0])
                    d.update({"low_fin": pc})

            output_list.append(d)

    return output_list







if __name__ == '__main__':
    json_path = sys.argv[1]
    mypath = sys.argv[2]

    listt =annotations(json_path, mypath)

    # folder containing the json file
    # json_path = "/home/marco/Desktop/Kaggle_challenge/train/annotations/check.json"


    # path of the folder containing all the folders of fish classes
    #mypath = "/home/marco/Desktop/Kaggle_challenge/train/LAG"



###--TEST FUNCTION--PRINT SUBIMAGES "FISH"(WITH SCATTER ANNOTATIONS) "NON FISH" FROM THE LIST RETURNED FROM FUNCTION--

    # for i in range(len(listt)) :
    #     plt.figure((i*2))
    #     plt.imshow(listt[i]["fish"])
    #     plt.scatter(listt[i]["head"][1],listt[i]["head"][0])
    #     plt.scatter(listt[i]["tail"][1], listt[i]["tail"][0])
    #     plt.scatter(listt[i]["up_fin"][1], listt[i]["up_fin"][0])
    #     plt.scatter(listt[i]["low_fin"][1], listt[i]["low_fin"][0])
    #     plt.figure((i*2)+1)
    #     plt.imshow(listt[i]["non_fish"])
    #
    # #print listt
    # plt.show()



































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































