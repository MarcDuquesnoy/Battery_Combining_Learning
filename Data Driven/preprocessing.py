from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


# Loading of the dataset. Please note you need to adapt the path on your own.

path_on_your_own = './data/Electrode_Structures_Dataset.csv'
Data = pd.read_csv(path_on_your_own, sep=",")
Data["tliq"] = np.sqrt(Data['tliq'])
Data["tsol"] = np.sqrt(Data['tsol'])
Data['Gap'] = 185.53059*np.exp(-Data['Gap']/38.122) - 6.51726

# splitting the dataset for all output variables

Train, Test = train_test_split(Data, train_size=0.8)

for out in ["tliq", 'tsol', 'CC/AM', 'CC/CBD', 'Active_Surface']:

    x = Train[[out, 'Gap', 'AM', 'CBD', 'porous']]
    y = Test[[out, 'Gap', 'AM', 'CBD', 'porous']]

    x.to_csv('./data/path_you_need_for_training_'+out+'.dat', sep=' ')
    y.to_csv('./data/path_you_need_for_testing_'+out+'.dat', sep=' ')