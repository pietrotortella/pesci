import os, sys
from scipy import misc
from os import listdir
from os.path import isfile, join

# This function can be used by the terminal: python mirror.py '/path/of/images'
# It will create a subfolder with mirrored images

# The following is the function that performs the triple mirroring and save outputs into a folder_destination
def mirrorImg(image_file,folder_destination):
    image = misc.imread(image_file)
    im_mir2=image[:,::-1,:]#np.fliplr(image)
    im_mir3=im_mir2[::-1,:,:]#np.flipud(im_mir2)
    im_mir4=im_mir3[:,::-1,:]#np.fliplr(im_mir3)
    misc.imsave(join(folder_destination, os.path.basename(image_file)[0:-4]+'_mir2.jpg'),im_mir2)
    misc.imsave(join(folder_destination, os.path.basename(image_file)[0:-4]+'_mir3.jpg'),im_mir3)
    misc.imsave(join(folder_destination, os.path.basename(image_file)[0:-4]+'_mir4.jpg'),im_mir4)
    print 'image ' + image_file + ' has been mirrored (three times)\n'

if __name__ == '__main__':
    mypath = sys.argv[1]
    ### The current path is set to "mypath" as the current location and create, if it does not already exist, the "mirrored" subfolder
    os.chdir(mypath)
    try:
        os.mkdir('mirrored')
    except:
        pass
    # List all the files within mypath (we are assuming just .jpg files are contained)
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    # The function is applied to all the images within "mypath" and saved in the "mirrored" subfolder
    for f in onlyfiles:
        mirrorImg(f, './mirrored')
