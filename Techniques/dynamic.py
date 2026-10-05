import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from numpy import ndarray as NDArray
import pandas as pd




class dynamic_kr():

    def __init__(self,window_norm=False,slide=100,window=200,policy="or"):
        #self.z = list(d / 2 for d in range(4, 10))
        self.z = list(d / 2 for d in range(4, 20))
        self.metric= "euclidean"
        self.k=[5,6,7,8,9,10,13,17,21,30,40]
        #self.k = [5,8, 13, 21, 30, 40]
        self.window_norm=window_norm
        self.slide=slide
        self.window=window
        self.policy=policy

    def _calc_dist(self,query : NDArray, pts: NDArray):
        return cdist(query, pts, metric=self.metric)

    def search(self,query: NDArray,points:NDArray, k: int):
        dists = self._calc_dist(query,points)

        I = (
            np.argsort(dists, axis=1)
            if k > 1
            else np.expand_dims(np.argmin(dists, axis=1), axis=1)
        )
        D = np.take_along_axis(np.array(dists), I, axis=1)
        return D, I

    def _dynamic_rk(self, query : pd.DataFrame, pts: pd.DataFrame):
        ks = [k for k in self.k if k < pts.shape[0] - 1]

        if len(ks) == 0:
            return -1, -1.0, -1.0

        D, I = self.search(pts,query, max(ks) + 1)

        max_res, k_sel, r_sel = -1.0, -1, -1.0
        for k in ks:
            kdists = D[:, k]
            kdistmin=kdists.min()
            kdistmax=kdists.max()
            kdists=(kdists-kdistmin)/(kdistmax-kdistmin)


            m = kdists.mean()
            s = kdists.std()
            rs = [m + z_i * s for z_i in self.z]

            for r in rs:
                inliers_dists = kdists[kdists <= r]  # keeping the distances of inliers
                n_outliers = (kdists > r).sum()  # counting outliers
                if r <0:
                    continue
                if n_outliers == 0:
                    break

                if len(inliers_dists) <= 1:
                    continue

                dmean = m - inliers_dists.mean()
                dstd = s - inliers_dists.std()
                res = (dmean / m + dstd / s) / ((n_outliers))  # objective function

                if res > max_res:
                    minoutlier = min(kdists[kdists > r]) * (kdistmax - kdistmin) + kdistmin
                    maxinlier = max(kdists[kdists <= r]) * (kdistmax - kdistmin) + kdistmin

                    max_res, k_sel, r_sel = res, k, (minoutlier+maxinlier)/2

        return k_sel, r_sel, max_res


    def collect_scores(self,query_df):
        pts = query_df
        if self.window_norm:
            pts=(pts-pts.min())/(pts.max()-pts.min())
        curr_k, r, res = self._dynamic_rk(pts, pts)
        #print(f"chosen k:{curr_k} r:{r} with res:{res}")
        # k = curr_k + 1  # ΔΕΝ ΧΡΕΙΑΖΕΤΑΙ ΑΥΤΟ

        D, _ = self.search(pts,pts, curr_k)
        score=[]
        for d in D[:, curr_k]:
            if d<0 or r<0:
                score.append(0)
            elif d>r and (d-r)/d>0.05:
                #score.append((d-r)/d)
                score.append(1)
            else:
                score.append(0)
        return score
    

    def combinescores(self,final_score,currentdf,ids):
        scores = self.collect_scores(currentdf)
        for sc, ind in zip(scores, ids):
            if ind in final_score.keys():
                if self.policy=="or":
                    final_score[ind] = max(final_score[ind], sc)
                elif self.policy=="and":
                    final_score[ind] = min(final_score[ind], sc)
                elif self.policy=="first":
                    final_score[ind] = final_score[ind]
                elif self.policy=="last":
                    final_score[ind] = sc
                else:
                    final_score[ind] = sc
            else:
                final_score[ind] = sc
        return final_score
    

    def fit(self,df):
        final_score={}
        pos=0
        for pos in range(self.window,len(df),self.slide):
            currentdf=df[max(pos-self.window,0):pos]
            ids=list(range(max(pos-self.window,0),pos))
            final_score=self.combinescores(final_score, currentdf,ids)
        if pos< len(df):
            pos=len(df)
            currentdf = df[max(pos - self.window, 0):pos]
            ids = list(range(max(pos - self.window, 0), pos))
            final_score = self.combinescores(final_score, currentdf, ids)
        scores_to_return=[]
        for ind in range(0, len(df)):
            scores_to_return.append(final_score[ind])
        return np.array(scores_to_return)