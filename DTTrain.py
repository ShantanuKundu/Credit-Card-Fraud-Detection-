from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np


def model_dt(X_train,Y_train,X_test,Y_test, input_data):
        #Implementing Decision Tree over the Testing Dataset
        classifier = DecisionTreeClassifier()
        print('\nTraining the model with Decision Tree Algorithm....\n')
        classifier.fit(X_train.values, Y_train.values)  

        #Prediction over the  Testing Data
        test_prediction = classifier.predict(X_test.values)

        #Checking the Accuracy of the model
        test_data_accuracy = accuracy_score(test_prediction, Y_test)
        print('\nPERFORMANCE OF THE MODEL\n')
        print('\nAccuracy Score:\n',test_data_accuracy)
        print('\nConfusion Matrix:\n',confusion_matrix(Y_test, test_prediction))  
        print('\nClassification Report:\n',classification_report(Y_test, test_prediction))

        print('\nChecking the model from a sample data\n')
        input_data=input_data.reshape(1,-1)
        pred = classifier.predict(input_data)
        for i in pred:
                if (i== 1):
                        print('Class Value:',i,'\nFraud Detected')
                else:
                        print('Class Value:',i,'\nNo Fraud Detected')


if __name__=='__main__':
    model_dt(X_train,Y_train,X_test,Y_test, input_data)
    
