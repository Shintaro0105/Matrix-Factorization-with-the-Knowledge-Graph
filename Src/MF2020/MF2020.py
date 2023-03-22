import numpy as np
import time
import random
from math import sqrt,fabs,log
import sys
from tqdm import tqdm
import time
import pandas as pd
from recommenders.datasets.python_splitters import python_random_split
from recommenders.evaluation.python_evaluation import map_at_k, ndcg_at_k, precision_at_k, recall_at_k
from recommenders.models.cornac.cornac_utils import predict_ranking
from recommenders.utils.timer import Timer
from recommenders.utils.constants import SEED

class mf2020:
    def __init__(self, unum, inum, num_negatives, ratedim, trainfile, testfile, steps, delta, beta_e, beta_h):
        self.unum = unum
        self.inum = inum
        self.num_negatives = num_negatives
        self.ratedim = ratedim
        self.steps = steps
        self.delta = delta
        self.beta_e = beta_e
        self.beta_h = beta_h

        self.R, self.T, self.ba, self.train_matrix = self.load_rating(trainfile, testfile)
        print ('Load rating finished.')
        print ('train size : ', len(self.R))
        print ('test size : ', len(self.T))

        self.initialize();
        self.recommend();

    def load_rating(self, trainfile, testfile):
        R_train = []
        R_test = []
        matrix = np.zeros([self.unum,self.inum])
        ba = 0.0
        n = 0
        user_test_dict = dict()
        with open(trainfile) as infile:
            for line in infile.readlines():
                user, item, rating = line.strip().split('\t')
                R_train.append([int(user)-1, int(item)-1, 1])
                matrix[int(user)-1, int(item)-1]=1
                ba += int(rating)
                n += 1
        ba = ba / n
        ba = 0
        with open(testfile) as infile:
            for line in infile.readlines():
                user, item, rating = line.strip().split('\t')
                R_test.append([int(user)-1, int(item)-1, 1])

        return R_train, R_test, ba, matrix

    def initialize(self):
        self.U = np.random.normal(0,0.1,(self.unum, self.ratedim))
        self.V = np.random.normal(0,0.1,(self.inum, self.ratedim))
        self.a = np.zeros(self.unum)
        self.b = np.zeros(self.inum)
        self.mu = 0.0
    
    def sigmod(self, x):
        return 1 / (1 + np.exp(-x))
    
    def get_rating(self, i, j):
        return self.sigmod(self.mu + self.a[i] + self.b[j] + self.U[i, :].dot(self.V[j, :]))

    def maermse(self):
        m = 0.0
        mae = 0.0
        rmse = 0.0
        n = 0
        for t in self.T:
            n += 1
            i = t[0]
            j = t[1]
            r = t[2]
            r_p = self.get_rating(i, j)

            #if r_p > 5: r_p = 5
            #if r_p < 1: r_p = 1
            m = fabs(r_p - r)
            mae += m
            rmse += m * m
        mae = mae * 1.0 / n
        rmse = sqrt(rmse * 1.0 / n)
        return mae, rmse

    def recommend(self):
        mae = []
        rmse = []
        starttime = time.time()
        perror = 99999
        cerror = 9999

        for step in range(steps):
            total_error = 0.0
            train_start_time = time.time()
            train_set = []
            n = 0.0
            mktrainset = 0.0
            training = 0.0
            
            t0=time.time()
            
            for t in self.R:
                i = t[0]
                j = t[1]
                rij = t[2]
                train_set.append([int(i),int(j),1])
                
                cnt = 0
            
                while cnt < self.num_negatives:
                    k = np.random.randint(self.inum)
                    #if self.train_matrix[i,j] == 0:
                    train_set.append([int(i),int(k),0])
                    cnt+=1
                    
            np.random.shuffle(train_set)
            
            mktrainset = time.time()-t0
            
            t0=time.time()
            
            for t in train_set:
                n += 1
                i = t[0]
                j = t[1]
                rij = t[2]

                rij_t = self.get_rating(i, j)
                eij = rij - rij_t
                total_error += (-rij*np.log(rij_t)-(1-rij)*np.log(1-rij_t))
                
                mu_g = -eij + self.beta_e * self.mu
                
                self.mu -= delta * mu_g
                
                a_g = -eij + self.beta_e * self.a[i]
                b_g = -eij + self.beta_h * self.b[j]
                
                self.a[i] -= delta * a_g
                self.b[j] -= delta * b_g
                
                U_g = -eij * self.V[j, :] + self.beta_e * self.U[i, :]
                V_g = -eij * self.U[i, :] + self.beta_h * self.V[j, :]

                self.U[i, :] -= delta * U_g
                self.V[j, :] -= delta * V_g
                
            training = time.time()-t0
            
            perror = cerror
            cerror = total_error / n

            #self.delta = 0.93 * self.delta

            if(abs(perror - cerror) < 0.0001):
                break
            train_end_time = time.time()
            print ('step ', step, 'crror : ', cerror,'train time : ', (train_end_time - train_start_time))
            #print('Make Trainset : ',mktrainset, 'Training : ',training)
            #print ('train time : ', (train_end_time - train_start_time))
            #MAE, RMSE = self.maermse()
            #mae.append(MAE)
            #rmse.append(RMSE)
            #if step % 5 == 0:
            #print ('step, MAE, RMSE ', step, MAE, RMSE)
        test_time = time.time()
        print ('time: ', test_time - starttime)
        #print ('MAE: ', min(mae), ' RMSE: ', min(rmse))

if __name__ == '__main__':
    unum = 4010
    inum = 9788
    num_negatives = 8
    ratedim = 128 #int(sys.argv[1])
    train_rate = 0.8

    trainfile = '../data/um_' + str(train_rate) +'.train'
    testfile = '../data/um_' + str(train_rate) + '.test'
    steps = 500
    delta = 0.002
    beta_e = 0.005
    beta_h = 0.005
    print('train_rate: ', train_rate)
    print('ratedim: ', ratedim,'num_negatives:  ', num_negatives)
    print('max_steps: ', steps)
    print('delta: ', delta, 'beta_e: ', beta_e, 'beta_h: ', beta_h)

    MYP=mf2020(unum, inum, num_negatives, ratedim, trainfile, testfile, steps, delta, beta_e, beta_h)
    A=MYP.a.reshape((unum,1))
    B=MYP.b.reshape((inum,1))
    _A=np.ones([inum,1])
    _B=np.ones([unum,1])
    MU=np.ones([unum,inum])
    pred_rating=MYP.sigmod( MYP.mu*MU + A.dot(_A.T) + _B.dot(B.T) + MYP.U.dot(MYP.V.T))
    np.savetxt('pred_rating.txt',pred_rating)