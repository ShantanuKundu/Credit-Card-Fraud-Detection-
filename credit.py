#Importing modules
import numpy as np
import pandas as pd
import SplitData as sd
import DTTrain as dt
import LRTrain as lr

if __name__=='__main__':
    #Reading dataset: creditcard.csv
    dataset= pd.read_csv(r'ur_local_creditcard.csv_location')
    #print('Dataset Information:\n',dataset.info())
    print('Dataset Read')

    #Empty Datframes
    X=pd.DataFrame()
    Y=pd.DataFrame()
    X_train=pd.DataFrame()
    X_test=pd.DataFrame()
    Y_train=pd.DataFrame()
    Y_test=pd.DataFrame()

    #Calling the split function
    X,Y,X_train,X_test,Y_train,Y_test=sd.dataset_split(dataset)
    print('Training Data\n',X_train.head())
    print('\nTesting Data\n',X_test.head())
    print('\nTraining Data\n',Y_train.head())
    print('\nTesting Data\n',Y_test.head())


    print('\nTraining Data:',X_train.shape,Y_train.shape,)
    print('\nTesting Data:',X_test.shape,Y_test.shape)

    print('\nDataset Split\n')
    #User input for later checking of the algorithm
    input_data = np.array([-4.241638012,	-4.479747105,	1.500400421,	0.761600727, 2.534547639,	-1.369927622,	
                       -1.280461667,	0.110400214,	1.668234589,	-0.709295843,	2.403673606,	-1.404736145,
                       2.92813025,	1.370847564,	1.612285848, -0.856012518,	0.930796158,	1.439836836,	3.498868668,
                       -0.240451453,	-0.168899916,	1.174981173,	3.510390343,	0.692636159,	1.139299014,	0.949596639,
                       0.516918474,	-0.343522116,	30])

    #Calling the respective algorithm to determine the model
    choice = int(input('Algorithms used:\n1. Decision Tree\n2. Logistic Regression\nEnter choice of Algorithm:\n'))
    while 1:
        if choice == 1:
            dt.model_dt(X_train,Y_train,X_test,Y_test,input_data)
            choice=int(input('To Exit Press 3 Or To Continue Press Respective Algorithm Number:'))
        elif choice == 2:
            lr.model_lr(X_train,Y_train,X_test,Y_test,input_data)
            choice=int(input('To Exit Press 3 Or To Continue Press Respective Algorithm Number:'))
        elif  choice==3:
            break
        else:
            print('Invalid Choice')
    
