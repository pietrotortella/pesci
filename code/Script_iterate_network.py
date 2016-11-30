import numpy as np
import tensorflow as tf
import json

from create_database import samples
from recognition_conv_net import iterate_networks

filepath = '/home/terminale2/Documents/small_test_database.json'

#filepath = '/home/terminale2/Documents/ALL_small.json'

# drop = np.arange(0.4,0.75,0.05)
# reg = np.array([10 ** (-n/2) for n in range(2,11)])

drop = np.array([0.75])#np.arange(0.5, 0.7, 2)
reg = np.array([0.])

conf_matrices = []
accuracies_test = []
accuracies_train = []

for i in range(len(drop)):
    for j in range(len(reg)):
        (
            conf_matrix, acc_test, acc_train,
            train_history, test_history
        ) = iterate_networks(filepath=filepath,
                             dropout=drop[i],
                             regular_factor=np.float32(reg[j]),
                             niter=2000000)
        conf_matrices.append((drop[i],reg[j],conf_matrix))
        accuracies_test.append((drop[i],reg[j],np.array(acc_test)))
        accuracies_train.append((drop[i],reg[j],np.array(acc_train)))

        filename = 'NN_results/conf_matrix_' + str(drop[i]) + '-' + str(reg[j]) + '.npy'
        np.save(filename, conf_matrix)

        filename = 'NN_results/accuracy_test_' + str(drop[i]) + '-' + str(reg[j]) + '.npy'
        np.save(filename, acc_test)

        filename = 'NN_results/accuracy_train_' + str(drop[i]) + '-' + str(reg[j]) + '.npy'
        np.save(filename, acc_train)

        # filename = 'NN_results/accuracy_test_history' + str(drop[i]) + '-' + str(reg[j]) + '.json'
        # with open(filename, 'w') as outf:
        #     json.dump(test_history, outf)
        #
        # filename = 'NN_results/accuracy_train_history' + str(drop[i]) + '-' + str(reg[j]) + '.json'
        # with open(filename, 'w') as outf:
        #     json.dump(train_history, outf)

print conf_matrices, accuracies_test, accuracies_train