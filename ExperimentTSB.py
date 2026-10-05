import time

import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.metrics
from prts import ts_recall, ts_precision
import os
from TSB_UAD_code.feature import Window
from Techniques import distanceKR, dynamic
from TSB_UAD_code.slidingWindows import find_length
from sklearn.preprocessing import MinMaxScaler

import tsfel
# Data Preprocessing
class experiment:

    def __init__(self,slide=100,window=200,k=15,r=0.5,filepath = './data/YAHOO/Yahoo_A1real_1_data.out',slidingWindow=None,onlystatic=False,features=False,debug=False,normalize=False):

        self.debug=debug
        self.features=features
        df = pd.read_csv(filepath, header=None).dropna().to_numpy()
        max_length=len(df)+1
        self.name = filepath.split('/')[-1]
        #max_length = 10000
        self.normalize=normalize
        self.data = df[:max_length,0].astype(float)
        self.label = df[:max_length,1]
        self.initialslide=slidingWindow
        if slidingWindow is None:
            self.slidingWindow = find_length(self.data)
        else:
            self.slidingWindow = slidingWindow
        #print(f" sequence size: {slidingWindow}")

        self.X_data = Window(window = self.slidingWindow).convert(self.data).to_numpy()
        if features:
            self.X_data = self.gfeaturedfunormalize(self.X_data,self.name,slidingWindow)
        if normalize==True:
            self.X_data=(self.X_data-self.X_data.min())/(self.X_data.max()-self.X_data.min())
        # make labels like Thodoris
        templabel=self.label.copy()

        for i in range(len(self.label)):
            if self.label[i]==1:
                for q in range(max(0,i-self.slidingWindow+1),i):
                    templabel[q]=1
                for q in range(i,min(len(self.label),i+self.slidingWindow)):
                    templabel[q]=1
        self.label=templabel

        self.r=r
        self.k=k

        self.staticf1 = None
        self.dynamicf1 = None
        self.static_to_write=self.runstatic(slide=slide, window=window)
        if onlystatic==False:
            self.dyn_to_write=self.rundyn(slide=slide, window=window)
        if debug:
            plt.legend()
            plt.show()



    def calculate_f1_score(self,precision, recall):
        """
        Calculate F1 score given precision and recall.

        Parameters:
        - precision (float): Precision value between 0 and 1.
        - recall (float): Recall value between 0 and 1.

        Returns:
        - float: F1 score.
        """
        if precision + recall == 0:
            return 0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)
            return f1_score
    def gfeaturedfunormalize(self,dfvalues,name,lentgth):
        folder_path="./data/YAHOO/features/"
        try:
            if not os.path.exists(folder_path):
                # If not, create it
                os.makedirs(folder_path)
                print(f"Folder '{folder_path}' created.")
            fname=name.split(".")[0]+f"s{lentgth}."+name.split(".")[1]
            file_path = os.path.join(folder_path, fname)

            if os.path.exists(file_path):
                df=pd.read_csv(file_path, header=None)
                return df.values
            else:
                colnames = ["max","min","median", "PtP", "Var", "std", "RMS", "skewness", "kurtosis"]
                values = dfvalues
                alldata = []
                for row in values:
                    features = []
                    features.append(tsfel.feature_extraction.features.calc_median(row))
                    features.append(tsfel.feature_extraction.features.pk_pk_distance(row))
                    features.append(tsfel.feature_extraction.features.calc_var(row))
                    features.append(tsfel.feature_extraction.features.skewness(row))
                    alldata.append(features)
                dfnew = pd.DataFrame(alldata)
                dfnew.to_csv(file_path,index=False,header=False)
                return dfnew.values
        except Exception as e:
            print(e)
            exit(-1)

    def runstatic(self,slide=100,window=200):
        modelName='staticKR'
        clf = distanceKR.static_kr(k=self.k, R=self.r, slide=slide, window=window, window_norm=False, policy="or")
        #modelName='dyn'
        #clf = dynamic.dynamic_kr(slide=100,window=200,window_norm=False,policy="or")

        x = self.X_data
        starttime=time.time()
        score= clf.fit(x)
        endtime=time.time()
        self.statictime=endtime-starttime

        scoreinit=score
        # Post processing
        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
        score = np.array([score[0]]*math.ceil((self.slidingWindow-1)/2) + list(score) + [score[-1]]*((self.slidingWindow-1)//2))





        scoreAlarm = np.array([int(sc > 0) for sc in score])
        if len(set(scoreAlarm)) == 1:
            if scoreAlarm[0] == 1:
                recall = 1
                recallrange=0
                precision = len([lb for lb in self.label if lb > 0]) / len(self.label)
            else:
                recall = 0
                precision = 0
                recallrange=0
        else:
            recallrange = ts_recall(self.label, scoreAlarm, alpha=0, cardinality="one", bias="flat")
            recall = ts_recall(self.label, scoreAlarm, alpha=1, cardinality="one", bias="flat")
            precision = ts_precision(self.label, scoreAlarm, alpha=0, cardinality="one", bias="flat")

        self.staticf1 = self.calculate_f1_score(precision,recall)
        f1RR= self.calculate_f1_score(precision,recallrange)
        if self.debug:
            plt.plot(scoreinit,label="static")
            plt.plot(self.label,label="labels")


        return f"static,{self.k},{self.r},{window},{slide},{self.initialslide},{self.slidingWindow},{recall},{recallrange},{precision},{self.staticf1},{f1RR},{self.features},{self.name},{self.normalize}\n"
    def rundyn(self,slide=100,window=200):
        modelName='dyn'
        clf = dynamic.dynamic_kr(slide=slide, window=window, window_norm=False, policy="or")

        x = self.X_data
        starttime = time.time()
        score = clf.fit(x)
        endtime = time.time()
        self.dyntime = endtime - starttime
        if self.debug:
            plt.plot(score,label="dyn")
        # Post processing
        score = MinMaxScaler(feature_range=(0,1)).fit_transform(score.reshape(-1,1)).ravel()
        score = np.array([score[0]]*math.ceil((self.slidingWindow-1)/2) + list(score) + [score[-1]]*((self.slidingWindow-1)//2))


        scoreAlarm=np.array([int(sc>0) for sc in score])

        f1 = sklearn.metrics.f1_score(self.label, scoreAlarm)
        if len(set(scoreAlarm))==1:
            if scoreAlarm[0]==1:
                recall=1
                recallrange=1
                precision=len([lb for lb in self.label if lb>0])/len(self.label)
            else:
                recall=0
                recallrange=1
                precision=0
        else:
            recall = ts_recall(self.label, scoreAlarm, alpha=1, cardinality="one", bias="flat")
            recallrange = ts_recall(self.label, scoreAlarm, alpha=0, cardinality="one", bias="flat")

            precision=ts_precision(self.label, scoreAlarm, alpha=0, cardinality="one", bias="flat")
        #self.staticf1 = f1
        self.dynamicf1 = self.calculate_f1_score(precision, recall)
        f1RR = self.calculate_f1_score(precision, recallrange)

        return f"dyn,{0},{0},{window},{slide},{self.initialslide},{self.slidingWindow},{recall},{recallrange},{precision},{self.dynamicf1},{f1RR},{self.features},{self.name},{self.normalize}\n"
