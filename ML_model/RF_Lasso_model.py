import sys
sys.path.append('../')



import pandas as pd
import numpy as np
import csv
from sklearn.linear_model import LassoCV, MultiTaskLassoCV
from sklearn.preprocessing import scale,minmax_scale,StandardScaler
from sklearn.ensemble import ExtraTreesRegressor


from common import PMNF_exp,read_data

class RF_model(object):
    def __init__(self,train_path,test_path,pred_path):
        self.train_path = train_path
        self.test_path = test_path
        self.pred_path = pred_path
        self.RF_model = ExtraTreesRegressor(bootstrap=False,n_estimators=1000,max_depth=100)
        
    # preprocess data
    # 1. split trainset to X,y
    # 2. split testset to X_RF,y_RF and X_Lasso,y_Lasso
    def preprocess_data(self):

        # read data
        data_train = np.asarray(read_data(self.train_path))
        data_test = np.asarray(read_data(self.test_path))
        #print(data_train.shape,data_test.shape)
        
        # 1. split trainset to X,y
        self.X_RF_train = data_train[:,:-1]
        self.y_RF_train = data_train[:,-1]

        # 2. split testset to X_RF,y_RF and X_Lasso,y_Lasso
        #test data split by number of process(128) 
        split_at = data_test[:,-2].searchsorted([129])
        test_data_split = np.split(data_test,split_at)
        #print(test_data_split)

        #test data split to X,y of Random Forest and Lasso
        self.X_RF_test = test_data_split[0][:,:-1]
        self.y_RF_test = test_data_split[0][:,-1]
        self.X_lasso_test = test_data_split[1][:,:-1]
        self.y_lasso_test = test_data_split[1][:,-1]

        print(self.X_RF_test.shape,self.y_RF_test.shape,self.X_lasso_test.shape,self.y_lasso_test.shape)
    
    # train random forest model
    def RF_train(self):
        self.RF_model.fit(self.X_RF_train,self.y_RF_train)
    
    # predict runtime with random forest model 
    def RF_pred(self):
        y_RF_pred = self.RF_model.predict(self.X_RF_test)
        return y_RF_pred

    # control flow of overall preprocess,train,predict
    def run(self):
        self.preprocess_data()
        self.RF_train()
        y_RF_pred = self.RF_pred()
        MAPE_RF = np.mean(abs(y_RF_pred-self.y_RF_test)/self.y_RF_test)
        print("Random Forest MAPE is : {}".format(MAPE_RF))

        with open(self.pred_path,"w",newline='') as f:
            csv_writer = csv.writer(f)
            for i in range(len(self.y_RF_test)):
                #print(self.X_RF_test[i])
                #print(self.y_RF_test[i])
                #print(y_RF_pred[i])
                #print(i)
                row = np.append(np.append(self.X_RF_test[i],self.y_RF_test[i]),y_RF_pred[i])
                csv_writer.writerow(row)
        


class Lasso_model(object):

    def __init__(self,train_path,test_path,pred_path):
        self.train_path = train_path
        self.test_path = test_path
        self.pred_path = pred_path
        self.split_train_len = 8
        self.split_test_len = 32
        self.preprocess_data()
         # length of every train group of parameters
         # length of every test group of parameters

    def split_data(self,data,split_len):
        # sort data
        data = data[data[:,0].argsort(kind='mergesort')]
        data = data[data[:,1].argsort(kind='mergesort')]
        data = data[data[:,2].argsort(kind='mergesort')]
        # split data
        data = np.split(data,len(data)/split_len)
        return data
    
    def preprocess_data(self):
        # read data
        data_train = np.asarray(read_data(self.train_path))
        data_test = np.asarray(read_data(self.test_path))
        print(data_train.shape,data_test.shape)

        # sort and split train data group by number of process
        self.data_train_split = self.split_data(data_train,self.split_train_len)
        #print(self.data_train_split[0])

        # sort and split test data group by number of process
        self.data_test_split = self.split_data(data_test,self.split_test_len)
        #print(self.data_test_split[0])
    
    def get_train_test(self):
        # train and test PMNF expansion
        X_train_PMNF = np.asarray([PMNF_exp(row[-3]) for row in  self.data_train_split[0]])
        X_test_PMNF = np.asarray([PMNF_exp(row[-2]) for row in self.data_test_split[0][self.split_train_len:]])
        X_all_PMNF = np.append(X_train_PMNF,X_test_PMNF,axis=0)
        # standard scale feature
        scaler = StandardScaler()
        scaler.fit(X_all_PMNF)
        X_all_PMNF = scaler.transform(X_all_PMNF)
        X_train_PMNF = X_all_PMNF[:len(X_train_PMNF),:]
        X_test_PMNF = X_all_PMNF[len(X_train_PMNF):,:]


        y_trains = [group[:,-1] for group in self.data_train_split]
        y_tests = [group[self.split_train_len:,-1] for group in self.data_test_split]

        #print("X_train is :\n{}".format(X_train_PMNF))
        #print("X_test is :\n{}".format(X_test_PMNF))
        #print("X_all is :\n{}".format(X_all_PMNF.shape))
        #print(type(y_trains[0]),type(y_tests[0]))

        return X_train_PMNF,X_test_PMNF,y_trains,y_tests

# normal lasso model
class NormaLasso_model(Lasso_model):
    def __init__(self,train_path,test_path,pred_path):
        super().__init__(train_path,test_path,pred_path)
        self.normaLasso_model = LassoCV(alphas=[float(i)*0.05 for i in range(1,100)],cv=8,max_iter=1000000)

    def train(self,X_train,y_train):
        self.normaLasso_model.fit(X_train,y_train)
    
    def pred(self,X_pred):
        y_pred = self.normaLasso_model.predict(X_pred)
        return y_pred
    
    def run(self):
        
        X_train_PMNF,X_test_PMNF,y_trains,y_tests = super().get_train_test()

        with open(self.pred_path,"w",newline='') as f:
            csv_writer = csv.writer(f)
            for i in range(len(y_trains)):
                #print(y_trains[i])
                for row in self.data_train_split[i]:
                    csv_writer.writerow(row)
                self.train(X_train_PMNF,y_trains[i])
                y_pred = self.pred(X_test_PMNF)
                #print(y_pred)
                #print(self.data_test_split[i][self.split_train_len:,:].shape)
                rows = np.hstack((self.data_test_split[i][self.split_train_len:,:],y_pred.reshape(len(y_pred),1)))
                
                for row in rows:
                    csv_writer.writerow(row)
                

        
        

class MultiLasso_model(Lasso_model):
    def __init__(self,train_path,test_path,pred_path):
        super().__init__(train_path,test_path,pred_path)
        self.multiLasso_model = MultiTaskLassoCV(alphas=[float(i)*0.05 for i in range(1,100)],cv=8,max_iter=1000000)
   
    def train(self,X_train,Y_train):
        self.multiLasso_model.fit(X_train,Y_train)

    def pred(self,X_test):
        return self.multiLasso_model.predict(X_test)

    def run(self):
        X_train_PMNF,X_test_PMNF,y_trains,y_tests = super().get_train_test()
        self.train(X_train_PMNF,np.asarray(y_trains).T)
        y_preds = self.pred(X_test_PMNF).T

        print(y_preds.shape,np.asarray(y_tests).shape)

        with open(self.pred_path,"w",newline='') as f:
            csv_writer = csv.writer(f)
            for i in range(len(y_trains)):
                for row in self.data_train_split[i]:
                    csv_writer.writerow(row)
                
                group = self.data_test_split[i][self.split_train_len:,:]
                for j in range(len(group)):
                    row = np.append(group[j,:],y_preds[i][j])
                    csv_writer.writerow(row)







def main():
    folder = "../Data/{}".format(sys.argv[1])
    train_path = "{}/train.csv".format(folder)
    test_path = "{}/test.csv".format(folder)
    pred_RF_path = "{}/pred_RF.csv".format(folder)
    pred_normaLasso_path = "{}/pred_normaLasso.csv".format(folder)
    pred_multiLasso_path = "{}/pred_multiLasso.csv".format(folder)
    
    # random forest phase
    RF_model_instance = RF_model(train_path,test_path,pred_RF_path)
    RF_model_instance.run()

    # Lasso phase
    
    # normal Lasso 
    NormaLasso_model_instance = NormaLasso_model(pred_RF_path,test_path,pred_normaLasso_path)
    NormaLasso_model_instance.run()

    # multitask Lasso
    MultiLasso_model_instance = MultiLasso_model(pred_RF_path,test_path,pred_multiLasso_path)
    MultiLasso_model_instance.run()
    

if __name__ == "__main__":
    main()
        
    