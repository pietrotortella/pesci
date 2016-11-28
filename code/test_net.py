from recognition_conv_net import get_transformed_ims
from scipy import misc
import matplotlib.pyplot as plt
from create_database import samples



im = misc.imread('/home/terminale2/Desktop/tiger.jpg')

im = im.mean(axis=2)

imu = get_transformed_ims(im)
plt.figure()
plt.imshow(imu)
plt.show()