import numpy as np
import tensorflow as tf

from create_database import samples
from recognition_conv_net import iterate_networks

filepath = '/home/terminale2/Documents/ALL_small.json'

# drop = np.arange(0.4,0.75,0.05)
# reg = np.array([10 ** (-n/2) for n in range(2,11)])

drop = np.arange(0.5,0.7,2)
reg = np.array([0.01])

conf_matrices = []
accuracies_test = []
accuracies_train = []

for i in range(len(drop)):
    for j in range(len(reg)):
        conf_matrix, acc_test,acc_train = iterate_networks(filepath=filepath,dropout=drop[i],regular_factor=np.float32(reg[j]), niter=600000)
        conf_matrices.append((drop[i],reg[j],conf_matrix))
        accuracies_test.append((drop[i],reg[j],np.array(acc_test)))
        accuracies_train.append((drop[i],reg[j],np.array(acc_train)))

        filename = 'conf_matrix_' + str(drop[i]) + '-' + str(reg[j]) + '.npy'
        np.save(filename, conf_matrix)

        filename = 'accuracy_test_' + str(drop[i]) + '-' + str(reg[j]) + '.npy'
        np.save(filename, acc_test)

        filename = 'accuracy_train_' + str(drop[i]) + '-' + str(reg[j]) + '.npy'
        np.save(filename, acc_train)




print conf_matrices, accuracies_test, accuracies_train