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

class mymodel_spl_bias:
    def __init__(self, unum, inum, num_negatives, ratedim, userdim, itemdim, user_metapaths,item_metapaths, trainfile, testfile, steps, delta, beta_e, beta_h, beta_p, beta_w, beta_b, reg_u, reg_v):
        self.unum = unum
        self.inum = inum
        self.num_negatives = num_negatives
        self.ratedim = ratedim
        self.userdim = userdim
        self.itemdim = itemdim
        self.steps = steps
        self.delta = delta
        self.beta_e = beta_e
        self.beta_h = beta_h
        self.beta_p = beta_p
        self.beta_w = beta_w
        self.beta_b = beta_b
        self.reg_u = reg_u
        self.reg_v = reg_v

        self.user_metapathnum = len(user_metapaths)
        self.item_metapathnum = len(item_metapaths)

        self.X, self.user_metapathdims = self.load_embedding(user_metapaths, unum)
        print ('Load user embeddings finished.')

        self.Y, self.item_metapathdims = self.load_embedding(item_metapaths, inum)
        print ('Load user embeddings finished.')

        self.R, self.T, self.ba, self.train_matrix = self.load_rating(trainfile, testfile)
        print ('Load rating finished.')
        print ('train size : ', len(self.R))
        print ('test size : ', len(self.T))

        self.initialize();
        self.recommend();

    def load_embedding(self, metapaths, num):
        X = {}
        for i in range(num):
            X[i] = {}
        metapathdims = []
    
        ctn = 0
        for metapath in metapaths:
            sourcefile = '../data/embeddings/' + metapath
            #print sourcefile
            with open(sourcefile) as infile:
                
                k = int(infile.readline().strip().split(' ')[1])
                metapathdims.append(k)
                for i in range(num):
                    X[i][ctn] = np.zeros(k)

                n = 0
                for line in infile.readlines():
                    n += 1
                    arr = line.strip().split(' ')
                    i = int(arr[0]) - 1
                    for j in range(k):
                        X[i][ctn][j] = float(arr[j + 1])
                print ('metapath ', metapath, 'numbers ', n)
            ctn += 1
        return X, metapathdims

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
        self.E = np.random.randn(self.unum, self.itemdim) * 0.1
        self.H = np.random.randn(self.inum, self.userdim) * 0.1
        self.U = np.random.normal(0,0.1,(self.unum, self.ratedim))
        self.V = np.random.normal(0,0.1,(self.inum, self.ratedim))
        self.a = np.zeros(self.unum)
        self.b = np.zeros(self.inum)
        self.mu = 0.0

        self.pu = np.ones((self.unum, self.user_metapathnum)) * 1.0 / self.user_metapathnum
        self.pv = np.ones((self.inum, self.item_metapathnum)) * 1.0 / self.item_metapathnum


        self.Wu = {}
        self.bu = {}
        for k in range(self.user_metapathnum):
            self.Wu[k] = np.random.randn(self.userdim, self.user_metapathdims[k]) * 0.1
            self.bu[k] = np.random.randn(self.userdim) * 0.1

        self.Wv = {}
        self.bv = {}
        for k in range(self.item_metapathnum):
            self.Wv[k] = np.random.randn(self.itemdim, self.item_metapathdims[k]) * 0.1
            self.bv[k] = np.random.randn(self.itemdim) * 0.1
    
    def sigmod(self, x):
        return 1 / (1 + np.exp(-x))

    def cal_u(self, i):
        ui = np.zeros(self.userdim)
        for k in range(self.user_metapathnum):
            ui += self.pu[i][k] * self.sigmod((self.Wu[k].dot(self.X[i][k]) + self.bu[k]))
        return self.sigmod(ui)

    def cal_v(self, j):
        vj = np.zeros(self.itemdim)
        for k in range(self.item_metapathnum):
            vj += self.pv[j][k] * self.sigmod((self.Wv[k].dot(self.Y[j][k]) + self.bv[k]))
        return self.sigmod(vj)
    
    def get_rating(self, i, j):
        ui = self.cal_u(i)
        vj = self.cal_v(j)
        return self.sigmod(self.mu + self.a[i] + self.b[j] + self.U[i, :].dot(self.V[j, :]) + self.reg_u * ui.dot(self.H[j, :]) + self.reg_v * self.E[i, :].dot(vj))

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
            
            for t in self.R:
                i = t[0]
                j = t[1]
                rij = t[2]
                train_set.append([int(i),int(j),1])
                
                cnt = 0
            
                while cnt < self.num_negatives:
                    k = np.random.randint(self.inum)
                    #if self.train_matrix[i,k] == 0:
                    train_set.append([int(i),int(k),0])
                    cnt+=1
                    
            np.random.shuffle(train_set)
            
            for t in train_set:
                n+=1
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

                ui = self.cal_u(i)
                for k in range(self.user_metapathnum):
                    x_t = self.sigmod(self.Wu[k].dot(self.X[i][k]) + self.bu[k])
                    
                    pu_g = self.reg_u * -eij * (ui * (1-ui) * self.H[j, :]).dot(x_t) + self.beta_p * self.pu[i][k]
                    
                    Wu_g = self.reg_u * -eij * self.pu[i][k] * np.array([ui * (1-ui) * x_t * (1-x_t) * self.H[j, :]]).T.dot(np.array([self.X[i][k]])) + self.beta_w * self.Wu[k]
                    bu_g = self.reg_u * -eij * ui * (1-ui) * self.pu[i][k] * self.H[j, :] * x_t * (1-x_t) + self.beta_b * self.bu[k]
                    #print pu_g
                    self.pu[i][k] -= 0.1 * self.delta * pu_g
                    self.Wu[k] -= 0.1 * self.delta * Wu_g
                    self.bu[k] -= 0.1 * self.delta * bu_g

                H_g = self.reg_u * -eij * ui + self.beta_h * self.H[j, :]
                self.H[j, :] -= self.delta * H_g

                vj = self.cal_v(j)
                for k in range(self.item_metapathnum):
                    y_t = self.sigmod(self.Wv[k].dot(self.Y[j][k]) + self.bv[k])
                    pv_g = self.reg_v * -eij * (vj * (1-vj) * self.E[i, :]).dot(y_t) + self.beta_p * self.pv[j][k]
                    Wv_g = self.reg_v * -eij  * self.pv[j][k] * np.array([vj * (1-vj) * y_t * (1 - y_t) * self.E[i, :]]).T.dot(np.array([self.Y[j][k]])) + self.beta_w * self.Wv[k]
                    bv_g = self.reg_v * -eij * vj * (1-vj) * self.pv[j][k] * self.E[i, :] * y_t * (1 - y_t) + self.beta_b * self.bv[k]

                    self.pv[j][k] -= 0.1 * self.delta * pv_g
                    self.Wv[k] -= 0.1 * self.delta * Wv_g
                    self.bv[k] -= 0.1 * self.delta * bv_g

                E_g = self.reg_v * -eij * vj + 0.01 * self.E[i, :]
                self.E[i, :] -= self.delta * E_g

            perror = cerror
            cerror = total_error / n

            self.delta = 0.93 * self.delta

            if(abs(perror - cerror) < 0.0001):
                break
            train_end_time = time.time()
            print ('step ', step, 'crror : ', cerror,'train time : ', (train_end_time - train_start_time))
            #MAE, RMSE = self.maermse()
            #mae.append(MAE)
            #rmse.append(RMSE)
            #if step % 5 == 0:
            #print ('step, MAE, RMSE ', step, MAE, RMSE)
            #test_time = time.time()
            #print ('time: ', test_time - train_end_time)
        self.endstep=step
        self.endloss=cerror
        test_time = time.time()
        print ('time: ', test_time - starttime)
        #print ('MAE: ', min(mae), ' RMSE: ', min(rmse))
        
if __name__ == '__main__':
    
    unum = 4010
    inum = 9788
    num_negatives = 10
    ratedim = 128 #int(sys.argv[1])
    userdim = 30
    itemdim = 30
    train_rate = 0.8
    
    user_metapaths = ['umu', 'umamu', 'umdmu', 'umtmu']
    item_metapaths = ['mam', 'mdm', 'mtm', 'mum']

    for i in range(len(user_metapaths)):
        user_metapaths[i] += '_' + str(train_rate) + '.embedding'
    for i in range(len(item_metapaths)):
        item_metapaths[i] += '_' + str(train_rate) + '.embedding'

#user_metapaths = ['ubu_' + str(train_rate) +'.embedding', 'ubcibu_''.embedding', 'ubcabu_0.8.embedding']
    
#item_metapaths = ['bub_0.8.embedding', 'bcib_0.8.embedding', 'bcab_0.8.embedding']
    trainfile = '../data/um_' + str(train_rate) +'.train'
    testfile = '../data/um_' + str(train_rate) + '.test'
    steps = 150
    delta = 0.01
    beta_e = 0.005
    beta_h = 0.005
    beta_p = 2
    beta_w = 0.1
    beta_b = 0.1
    reg_u = 0.1
    reg_v = 0.1
    print('train_rate: ', train_rate)
    print('ratedim: ', ratedim, ' userdim: ', userdim, ' itemdim: ', itemdim)
    print('max_steps: ', steps)
    print('delta: ', delta, 'beta_e: ', beta_e, 'beta_h: ', beta_h, 'beta_p: ', beta_p, 'beta_w: ', beta_w, 'beta_b', beta_b, 'reg_u', reg_u, 'reg_v', reg_v)
    MYP=mymodel_spl_bias(unum, inum, num_negatives, ratedim, userdim, itemdim, user_metapaths, item_metapaths, trainfile, testfile, steps, delta, beta_e, beta_h, beta_p, beta_w, beta_b, reg_u, reg_v)
    pred_rating=np.zeros([unum,inum])
    eu=np.zeros([unum,userdim])
    for i in range(unum):
        ui = MYP.cal_u(i)
        eu[i,:]=ui.T

    ev=np.zeros([inum,itemdim])
    for i in range(inum):
        vi = MYP.cal_v(i)
        ev[i,:]=vi.T
                
    A=MYP.a.reshape((unum,1))
    B=MYP.b.reshape((inum,1))
    _A=np.ones([inum,1])
    _B=np.ones([unum,1])
    MU=np.ones([unum,inum])

    pred_rating=MYP.sigmod(MYP.mu*MU + A.dot(_A.T) + _B.dot(B.T) + MYP.U.dot(MYP.V.T) + MYP.reg_u * eu.dot(MYP.H.T) + MYP.reg_v * MYP.E.dot(ev.T))
    np.savetxt('dim128-30-30neg10.txt',pred_rating)