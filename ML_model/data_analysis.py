import sys
sys.path.append('../')


import csv
import sys
import numpy as np
import matplotlib.pyplot as plt

from common import read_data


class analyze_data(object):
    def __init__(self,method_path_dict,plt_folder):
        self.method_path_dict = method_path_dict
        self.get_data()
        self.plt_folder = plt_folder
    
    def get_data(self):
        self.method_data_dict = {}
        for method in self.method_path_dict:
            self.method_data_dict[method] = np.asarray(read_data(self.method_path_dict[method]))




    def get_MAPE(self,method,data):
        num_train = 8
        num_group = 100

        # sort data by number of process
        data = data[data[:,-3].argsort(kind='mergesort')]
        inter_exec_time = data[:num_train*num_group,-2]
        inter_pred_time = data[:num_train*num_group,-1]
        extra_exec_time = data[num_train*num_group:,-2]
        extra_pred_time = data[num_train*num_group:,-1]

        # MAPE
        MAPE_inter = np.mean(abs(inter_pred_time-inter_exec_time)/inter_exec_time)
        MAPE_extra = np.mean(abs(extra_pred_time-extra_exec_time)/extra_exec_time)

        print("{} inter(16-128) MAPE is: {}\n{} extra(128-512) MAPE is: {}".format(method,MAPE_inter,method,MAPE_extra))
        return MAPE_inter,MAPE_extra

    def extact(self):

        len_group = 32
        nprocs = np.asarray([i for i in range(len_group)])
        extract_data = {}
        for method in self.method_data_dict:
            data = self.method_data_dict[method]
            data = data[data[:,1].argsort(kind='mergesort')]
            data = data[data[:,2].argsort(kind='mergesort')]
            data = data[data[:,3].argsort(kind='mergesort')]
            nprocs = data[:len_group,-3]
            split_len = len(data)/len_group
            extract_data["excute"] = np.asarray(np.split(data[:,-2],split_len))
            extract_data[method] = np.asarray(np.split(data[:,-1],split_len))
        return nprocs,extract_data
    
    def draw_plt(self):
        n_group = 100
        n_rownum = 2
        n_colnum = 5

        nprocs,extract_data = self.extact()

        for i in range(int(n_group/(n_rownum*n_colnum))):
            plt_path = "{}/{}.jpg".format(self.plt_folder,i)
            image,axes = plt.subplots(n_rownum,n_colnum)
            image.set_size_inches(50,20,)
            for row in range(n_rownum):
                for col in range(n_colnum):
                    methods = []
                    for method in extract_data:
                        axes[row,col].plot(nprocs,extract_data[method][i*n_rownum*n_colnum+row*n_rownum+col,:],label=method,marker='o')
                        methods = methods + [method]
                    methods = tuple(methods)
                    axes[row,col].legend(methods,loc='upper right')
                    
                    #plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,wspace=0.35)
            image.savefig(plt_path,dpi=100)

    def run(self):
        for method in self.method_data_dict:
            self.get_MAPE(method,self.method_data_dict[method])
        self.draw_plt()

def main():

    # file path
    data_folder = "../Data/{}".format(sys.argv[1])
    EPMNF_path = "{}/pred_EPMNF.csv".format(data_folder)
    normaLasso_path = "{}/pred_normaLasso.csv".format(data_folder)
    multiLasso_path = "{}/pred_multiLasso.csv".format(data_folder)
    plt_folder = "../Result/{}".format(sys.argv[1])

    method_path_dict = {
        "EPMNF" : EPMNF_path,
        "normaLasso" : normaLasso_path,
        "multiLasso" : multiLasso_path,
    }

    # analyze data
    analyze_data_instance = analyze_data(method_path_dict,plt_folder)
    analyze_data_instance.run()

if __name__ == "__main__":
    main()

