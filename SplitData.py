from sklearn.model_selection import train_test_split  
#Function to distinguish the Required columns from the dataset
def dataset_split(dataset):
    #INDEPENDENT DATA
    to_drop=int(input('Enter the no. of columns to drop:'))
    to_drop_col=[]
    for i in range(to_drop):
      to_drop_col.append(str(input('Enter the column to drop:')))
    X = dataset.drop(to_drop_col, axis=1)  
    #DEPENDENT DATA
    dependent=str(input('Enter the Dependent Data:'))
    Y = dataset[dependent]  
    print(X.head(),'\n',Y.head())

    # Checking the distribution of Class Variable
    dt=dataset[dependent].value_counts()
    print('Frequency of 0:',dt[0],'\nFrequency of 1:',dt[1])

    #Determining the Training and Testing dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

    return X,Y,X_train,X_test,Y_train,Y_test




if __name__=='__main__':
    dataset_split(dataset)
