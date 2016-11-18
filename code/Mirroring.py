import os
from scipy import misc
from os import listdir
from os.path import isfile, join

### Set "mypath" as the folder containing the images
mypath = '/home/terminale0/Downloads/train/train/ALB'

### The current path is set to "mypath" as the current location and the "mirrored" subfolder is created if it does not already exist
os.chdir(mypath)
try:
    os.mkdir('mirrored')
except:
    pass

# List all the files within mypath (we are assuming just .jpg files are contained)
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

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

# The function is applied to all the images within "mypath" and saved in the "mirrored" subfolder
for f in onlyfiles:
    mirrorImg(f,'./mirrored')

### Uncomment this code to view images
# f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col', sharey='row')
# ax1.imshow(image)
# ax1.set_title('Original')
# ax2.imshow(im_mir2)
# ax2.set_title('First mirroring')
# ax3.imshow(im_mir3)
# ax3.set_title('Second mirroring')
# ax4.imshow(im_mir4)
# ax4.set_title('Third mirroring')
