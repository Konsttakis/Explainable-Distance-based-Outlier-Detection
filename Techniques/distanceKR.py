import numpy as np
from scipy.spatial.distance import cdist
from numpy import ndarray as NDArray
import pandas as pd


class static_kr():
    def __init__(self, k=5,R=1,slide=100,window=200,window_norm=False,policy="or"):
        self.k=k
        self.R=R
        self.window_norm=window_norm
        self.policy = policy
        self.slide = slide
        self.window = window
        self.metric="euclidean"
    def _calc_dist(self,query : NDArray, pts: NDArray):
        return cdist(query, pts, metric=self.metric)
    def collect_scores(self,query_df):
        pts = query_df
        if self.window_norm:
            pts=(pts-pts.min())/(pts.max()-pts.min())

        D, _ = self.search(pts, pts)
        score = []
        for d in D[:, self.k - 1]:
            if d < 0 or self.R < 0:
                score.append(0)
            elif d > self.R:
                score.append(1)
            else:
                score.append(0)
        return score

    def search(self, query: NDArray, points: NDArray):
        dists = self._calc_dist(query, points)

        I = (
            np.argsort(dists, axis=1)
            if self.k > 1
            else np.expand_dims(np.argmin(dists, axis=1), axis=1)
        )
        D = np.take_along_axis(np.array(dists), I, axis=1)
        return D, I

    def combinescores(self, final_score, currentdf, ids):
        scores = self.collect_scores(currentdf)
        for sc, ind in zip(scores, ids):
            if ind in final_score.keys():
                if self.policy == "or":
                    final_score[ind] = max(final_score[ind], sc)
                elif self.policy == "and":
                    final_score[ind] = min(final_score[ind], sc)
                elif self.policy == "first":
                    final_score[ind] = final_score[ind]
                elif self.policy == "last":
                    final_score[ind] = sc
                else:
                    final_score[ind] = sc
            else:
                final_score[ind] = sc
        return final_score

    def fit(self, df):
        final_score = {}
        pos=0
        for pos in range(self.window, len(df), self.slide):
            currentdf = df[max(pos - self.window, 0):pos]
            ids = [kati for kati in range(max(pos - self.window, 0), pos)]
            final_score = self.combinescores(final_score, currentdf, ids)
        if pos < len(df):
            pos = len(df)
            currentdf = df[max(pos - self.window, 0):pos]
            ids = [kati for kati in range(max(pos - self.window, 0), pos)]
            final_score = self.combinescores(final_score, currentdf, ids)
        scores_to_return = []
        for ind in range(0, len(df)):
            scores_to_return.append(final_score[ind])
        return np.array(scores_to_return)
