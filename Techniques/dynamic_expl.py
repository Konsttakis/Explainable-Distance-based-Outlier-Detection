import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from numpy import ndarray as NDArray
import pandas as pd


class dynamic_kr_expl():

    def __init__(self,window_norm=False,slide=100,window=200,policy="or"):
        self.z = list(d / 2 for d in range(4, 20))
        self.metric= "euclidean"
        self.k=[5,6,7,8,9,10,13,17,21,30,40]
        self.window_norm=window_norm
        self.slide=slide
        self.window=window
        self.policy=policy
        
        # Final explanation label of every point of the stream
        self.outlier_labels = []
        
        # Per-window history, used by the figure scripts
        self.k_history = []
        self.r_history = []
        self.density_profiles = []
        self.label_telemetry = []
        # Stream indices of the window currently being scored
        self._current_ids = None

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
        return D, I, dists

    def _dynamic_rk(self, query : pd.DataFrame, pts: pd.DataFrame):
        ks = [k for k in self.k if k < pts.shape[0] - 1]

        if len(ks) == 0:
            return -1, -1.0, -1.0, None, None

        D, I, full_dists = self.search(pts,query, max(ks) + 1)

        max_res, k_sel, r_sel = -1.0, -1, -1.0
        for k in ks:
            kdists = D[:, k]
            kdistmin=kdists.min()
            kdistmax=kdists.max()
            if kdistmax == kdistmin:
                kdists = np.zeros_like(kdists)
            else:
                kdists=(kdists-kdistmin)/(kdistmax-kdistmin)

            m = kdists.mean()
            s = kdists.std()
            rs = [m + z_i * s for z_i in self.z]

            for r in rs:
                inliers_dists = kdists[kdists <= r]  
                n_outliers = (kdists > r).sum()  
                if r <0:
                    continue
                if n_outliers == 0:
                    break

                if len(inliers_dists) <= 1:
                    continue

                dmean = m - inliers_dists.mean()
                dstd = s - inliers_dists.std()
                res = (dmean / m + dstd / s) / ((n_outliers))  

                if res > max_res:
                    minoutlier = min(kdists[kdists > r]) * (kdistmax - kdistmin) + kdistmin
                    maxinlier = max(kdists[kdists <= r]) * (kdistmax - kdistmin) + kdistmin

                    max_res, k_sel, r_sel = res, k, (minoutlier+maxinlier)/2

        return k_sel, r_sel, max_res, D, full_dists

    # NN-based labelling (Algorithm 1, Table 3 of the reference paper)
    def _nn_based_labeling(self, NN):
        nn1, nn2, nn3, nn4, nn5 = NN
        
        if nn5 == 0: 
            return "Isolated"
        if nn5 != 0 and nn4 == 0: 
            return "Near cluster"
        if nn5 != 0 and nn4 != 0 and nn3 == 0: 
            return "Near cluster"
        if 0 < nn5 < 2.5 and nn4 != 0 and 0 < nn3 < 1: 
            return "Sparse microcluster"
        if 0 < nn5 < 2.5 and nn4 != 0 and nn3 != 0 and nn2 == 0: 
            return "Sparse microcluster"
        if nn5 >= 2.5 and nn4 != 0 and 0 < nn3 < 1: 
            return "Sparse cluster"
        if nn5 >= 2.5 and nn4 != 0 and nn3 != 0 and nn2 == 0: 
            return "Sparse cluster"
        if nn5 != 0 and 0 < nn4 <= 2.5 and nn3 >= 1 and nn2 != 0: 
            return "Dense microcluster"
        if nn5 != 0 and nn4 > 2.5 and nn3 >= 1 and nn2 != 0: 
            return "Dense cluster"
        
        return "Unclassified Outlier"

    # Status-based labelling (Algorithm 2, Table 4)
    def _status_based_labeling(self, Status):
        outlier_pos = {}
        for row_idx, row in enumerate(Status):
            pos = 7 
            for col_idx, is_outlier in enumerate(row):
                if is_outlier:
                    pos = col_idx + 1
                    break
            outlier_pos[row_idx + 1] = pos

        op = outlier_pos
        
        if op[5] == 1: return "Isolated"
        if op[5] != 1 and op[4] == 1: return "Near cluster"
        if 1 < op[5] <= 4 and op[4] != 1 and op[3] == 1: return "Sparse microcluster"
        if op[5] > 4 and op[4] != 1 and op[3] == 1: return "Sparse cluster"
        if op[5] != 1 and op[4] != 1 and op[3] >= 3 and op[2] == 1: return "Dense microcluster"
        if 1 < op[5] <= 4 and op[4] != 1 and 1 < op[3] < 3 and op[2] == 1: return "Sparse microcluster"
        if op[5] > 4 and op[4] != 1 and 1 < op[3] < 3 and op[2] == 1: return "Sparse cluster"
        if 1 < op[5] < 6 and op[4] != 1 and op[3] != 1 and op[2] != 1 and op[1] == 1: return "Dense microcluster"
        if op[5] > 6 and op[4] != 1 and op[3] != 1 and op[2] != 1: return "Dense cluster"
        if op[5] != 1 and op[4] != 1 and op[3] != 1 and op[2] != 1 and op[1] != 1: return "Dense cluster"

        return "Unclassified Outlier"

    # Combination of the two labellings (Algorithm 3)
    def _combination_based_labeling(self, NN, Status):
        class_NN = self._nn_based_labeling(NN)
        class_Status = self._status_based_labeling(Status)

        if class_NN == class_Status:
            return class_NN, class_Status, class_NN

        ranking = {
            "Dense cluster": 6,
            "Sparse cluster": 5,
            "Dense microcluster": 4,
            "Sparse microcluster": 3,
            "Near cluster": 2,
            "Isolated": 1,
            "Unclassified Outlier": 0
        }

        rank_NN = ranking.get(class_NN, 0)
        rank_Status = ranking.get(class_Status, 0)

        if rank_NN > rank_Status:
            return class_NN, class_Status, class_NN
        else:
            return class_NN, class_Status, class_Status

    def collect_scores(self,query_df):
        pts = query_df
        if self.window_norm:
            pts=(pts-pts.min())/(pts.max()-pts.min())
        curr_k, r, res, D_full, full_dists = self._dynamic_rk(pts, pts)
        
        self.k_history.append(curr_k)
        self.r_history.append(r)
        
        if curr_k == -1:
            return [(0, "Inlier")] * len(pts)

        # D_full and full_dists come from the parameter search; no second distance computation
        
        scores_and_labels = []
        for i, d in enumerate(D_full[:, curr_k]):
            if d<0 or r<0:
                scores_and_labels.append((0, "Inlier"))
            elif d>r and (d-r)/d>0.05:
                # Calculate counts for R/4, R/2, R, 2R, 4R using the precalculated full_dists
                radii = [r/4, r/2, r, 2*r, 4*r]
                counts = [max(0, np.sum(full_dists[i] <= rad) - 1) for rad in radii]
                
                # 1. Prepare NN input
                NN_normalized = [c / curr_k for c in counts]
                
                # 2. Prepare Status matrix input (5x5 matrix)
                k_vals = [curr_k/4, curr_k/2, curr_k, curr_k*2, curr_k*4]
                Status = []
                for c in counts:
                    row = []
                    for k_val in k_vals:
                        row.append(c < k_val)
                    Status.append(row)
                
                # 3. Get the final label using the combination algorithm
                class_nn, class_status, final_label = self._combination_based_labeling(NN_normalized, Status)
                
                idx = self._current_ids[i] if self._current_ids is not None else None

                self.density_profiles.append({
                    "idx": idx,
                    "label": final_label,
                    "radii": radii,
                    "counts": counts
                })
                
                self.label_telemetry.append({
                    "idx": idx,
                    "score": d,
                    "nn_label": class_nn,
                    "status_label": class_status,
                    "final_label": final_label,
                    "k": curr_k,
                    "r": r
                })
                
                scores_and_labels.append((1, final_label))
            else:
                scores_and_labels.append((0, "Inlier"))
                
        return scores_and_labels
    
    def combinescores(self,final_score,currentdf,ids):
        self._current_ids = ids
        scores_and_labels = self.collect_scores(currentdf)
        
        for (sc, label), ind in zip(scores_and_labels, ids):
            if ind in final_score.keys():
                current_sc = final_score[ind][0]
                if self.policy=="or":
                    if sc > current_sc: final_score[ind] = (sc, label)
                elif self.policy=="and":
                    if sc < current_sc: final_score[ind] = (sc, label)
                elif self.policy=="first":
                    final_score[ind] = final_score[ind]
                elif self.policy=="last":
                    final_score[ind] = (sc, label)
                else:
                    final_score[ind] = (sc, label)
            else:
                final_score[ind] = (sc, label)
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
        labels_to_return=[]
        
        for ind in range(0, len(df)):
            sc, lab = final_score[ind]
            scores_to_return.append(sc)
            labels_to_return.append(lab)
            
        self.outlier_labels = labels_to_return
        
        return np.array(scores_to_return)
