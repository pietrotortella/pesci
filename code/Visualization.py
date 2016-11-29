import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

accuracytest_folder = 'accuracy/test'
accuracytrain_folder = 'accuracy/train'

name_test = os.listdir(accuracytest_folder)
name_train = os.listdir(accuracytrain_folder)

test_results = pd.DataFrame()

for name in name_test:
    filepath = os.path.join(accuracytest_folder, name)

    try:
        dropout = float(name[14:18])
        reg = float(name[19:-4])
        len = 3
    except ValueError:
        dropout = float(name[14:17])
        reg = float(name[18:-4])
        len = 2




    this_accuracy = np.load(filepath)

    test_results.ix[reg, dropout] = this_accuracy


train_results = pd.DataFrame()

for name in name_train:
    filepath = os.path.join(accuracytrain_folder, name)

    try:
        dropout = float(name[15:19])
        reg = float(name[20:-4])
    except ValueError:
        dropout = float(name[15:18])
        reg = float(name[19:-4])

    this_accuracy = np.load(filepath)

    train_results.ix[reg, dropout] = this_accuracy

train_results = train_results.set_index(np.abs(train_results.index)[np.abs(train_results.index).argsort()])

print train_results.index
#train_results = train_results.ix[sorted(train_results.index)]

test_results = test_results.set_index(np.abs(test_results.index)[np.abs(test_results.index).argsort()])
print train_results

print test_results