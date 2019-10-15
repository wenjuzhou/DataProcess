import sys
sys.path.append('../')

import pandas as pd
import numpy as np
import csv
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import scale,minmax_scale,StandardScaler

from common import PMNF_exp, read_data

class EPMNF_model(object):
    
    def __init__(self,train_path,test_path,pred_path):
        self.train_path = train_path
        self.test_path = test_path
        self.pred_path = pred_path
        self.lasso_model = LassoCV(alphas=[float(i)*0.05 for i in range(1,100)],cv=10,n_alphas=10,max_iter=10000000,normalize=False,random_state=0)
    
    #get X_train,y_train,X_test,y_test, and EPMNF expansion
    def preprocess_data(self):
        train_data = read_data(self.train_path)
        test_data = read_data(self.test_path)
        len_train = len(train_data)
        len_test = len(test_data)
        train_data = np.asarray(train_data)
        test_data = np.asarray(test_data)
        #print(train_data.shape,test_data.shape)


        X_train,y_train = train_data[:,:-1],train_data[:,-1]
        X_test,y_test = test_data[:,:-1],test_data[:,-1]
        #print(X_train.shape,y_train.shape,X_test.shape,y_test.shape)

        X_all = np.append(X_train,X_test,axis=0)
        X_all_EPMNF = []
        for row in X_all:
            line = []
            for p in row:
                line = line + PMNF_exp(p)
            X_all_EPMNF.append(line)
        X_all_EPMNF = np.asarray(X_all_EPMNF)
        #print(X_all_EPMNF.shape)
        
        scaler = StandardScaler()
        scaler.fit(X_all_EPMNF)
        X_all_EPMNF = scaler.transform(X_all_EPMNF)

        X_train_EPMNF = X_all_EPMNF[:len_train,:]
        X_test_EPMNF = X_all_EPMNF[len_train:,:]
        print(X_train_EPMNF.shape,X_test_EPMNF.shape)

        return train_data,test_data,X_train_EPMNF,X_test_EPMNF,y_train,y_test

    def train(self):

        train_data,test_data,X_train_EPMNF,X_test_EPMNF,y_train,y_test = self.preprocess_data()
        self.lasso_model.fit(X_train_EPMNF,y_train)
        y_pred = self.lasso_model.predict(X_test_EPMNF)

        with open(self.pred_path,"w",newline='') as f:
            csv_writer = csv.writer(f)
            for i in range(len(test_data)):
                row = np.append(test_data[i],y_pred[i])
                csv_writer.writerow(row)
        #print(pred_data)
        print("The alpha is : {}".format(self.lasso_model.alpha_))
        print("The train R^2 is : {}".format(self.lasso_model.score(X_train_EPMNF,y_train)))
        print("The test R^2 is : {}".format(self.lasso_model.score(X_test_EPMNF,y_test)))
        print("number of no-zero coefs is : {}".format(np.count_nonzero(self.lasso_model.coef_)))

def main():
    train_path = '../Data/{}/train.csv'.format(sys.argv[1])
    test_path = '../Data/{}/test.csv'.format(sys.argv[1])
    pred_path = '../Data/{}/pred_EPMNF.csv'.format(sys.argv[1])
    EPMNF_model_instance = EPMNF_model(train_path,test_path,pred_path)
    EPMNF_model_instance.train()
    #EPMNF_model_instance.preprocess_data()
if __name__ == "__main__":
    main()


