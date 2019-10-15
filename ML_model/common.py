import sys
sys.path.append('../')


import pandas as pd
import numpy as np
import math


import csv
import os
import math



# read data from csv
def read_data(path):
    data = []
    with open(path,"r") as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            row = [float(i) for i in row]
            data.append(row)
        return data

# expand parameter to PMNF 
def PMNF_exp(p):
    p = float(p)
    pow_half_p = math.sqrt(p)
    pow_1half_p = pow_half_p*p
    pow_2_p = p*p
    log_p = math.log2(p)


    p1 = pow_2_p*log_p
    p2 = pow_2_p
    p3 = pow_2_p/log_p

    p4 = pow_1half_p*log_p
    p5 = pow_1half_p
    p6 = pow_1half_p/log_p

    p7 = p*log_p
    p8 = p
    p9 = p/log_p

    p10 = pow_half_p*log_p
    p11 = pow_half_p
    p12 = pow_half_p/log_p

    p13 = log_p
    p14 = log_p*math.sqrt(log_p)
    p15 = log_p*log_p

    p16 = 1/p1
    p17 = 1/p2
    p18 = 1/p3
    p19 = 1/p4
    p20 = 1/p5
    p21 = 1/p6
    p22 = 1/p7
    p23 = 1/p8
    p24 = 1/p9
    p25 = 1/p10
    p26 = 1/p11
    p27 = 1/p12
    p28 = 1/p13
    p29 = 1/p14
    p30 = 1/p15

    exp_p = [p1,p2,p3,p4,p5,p6,p7,p8,p9,p10,p11,p12,p13,p14,p15,p16,p17,p18,p19,p20,p21,p22,p23,p24,p25,p26,p27,p28,p29,p30]
    #exp_p = [p2,p3,p5,p7,p8,p9,p11,p13,p17,p18,p20,p22,p23,p24,p26,p28]

    return exp_p

def log_y(y):
    return np.asarray([math.log2(yi) for yi in y])

def exp_y(y):
    return np.asarray([math.pow(2,yi) for yi in y])


def main():
    p=128
    print(PMNF_exp(p))

if __name__ == "__main__":
    main()