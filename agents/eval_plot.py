import pdb
import graphs
import torch
import utils
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import functools
import nibabel as nib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
from scipy.stats import ttest_ind
import dcor
"""
Contains helper functions for evaluating non-trained scores of a model (i.e., class separability)
Scalars are first collected for each scan in _get_scalars_per_tau
- EvalPlotter uses projections of z along tau's as scalars
- BaseEvalPlotter uses z as scalars
"""
class BaseEvalPlotter:
    def __init__(self, agent, trainloader, valloader):
        self.agent = agent
        self.trainloader = trainloader
        self.valloader = valloader 
        fig_dir = os.path.expanduser("~/loca/results/")
        fig_dir = os.path.join(fig_dir, str(self.agent.config.rand_seed), self.agent.config.exp_id)
        utils.fs.create_dir(fig_dir)
        self.fig_dir = fig_dir


    def _get_scalars_per_tau(self, loader, settype):
        if hasattr(self.agent.model, "get_taus"):
            taus = self.agent.model.get_taus()
        preds = [[] for tau in taus]
        metas = dict()
        def update_preds(i, z, tau, preds):
            tau = tau/tau.norm()
            tau_factor = torch.matmul(z, tau) / torch.dot(tau, tau)
            preds[i].extend(tau_factor.tolist())

        unique_fps_counter = 0
        with torch.no_grad():
            # Get pred age factor and true ages
            for i, X in enumerate(loader):
                z1, batch_size = self._get_scalars_per_tau_get_z(X)
                self._get_scalars_per_tau_update_metas(X, metas, unique_fps_counter)
                unique_fps_counter += batch_size 

                for i, tau in enumerate(taus):
                    update_preds(i, z1, tau, preds)

            # Convert to np.array()
            for i in range(len(preds)):
                preds[i] = np.array(preds[i])

            for meta_name, meta in metas.items():
                metas[meta_name] = np.array(meta)

            eval_scalars_state = dict(
                preds = preds,
                metas = metas,
            )

        return eval_scalars_state
                
    def plot_top_dx(self, is_train=True):
        if is_train:
            settype = "train"
            loader = self.trainloader
        else:
            settype = "val"
            loader = self.valloader
        
        # Get alphas
        state = self._get_scalars_per_tau(loader, settype)
        preds = np.array(state["preds"])
        ntaus = preds.shape[0]
        # # # normalize to 0-1
        for i in range(ntaus):
            preds[i] = (preds[i] - preds[i].min()) / (preds[i].max() - preds[i].min())
        # preds = preds
        metas = state["metas"]      
        # Get t-test scores
        metas_to_split = ["dx"]
        for meta_to_split in metas_to_split:
            data = pd.DataFrame(preds)
            labels = metas[meta_to_split]
            # Caculate t-test and p-value
            t_scores = []
            p_values = []
            for pred in preds:
                stats = ttest_ind(pred[labels==0], pred[labels==1])
                t_scores.append(stats[0])
                p_values.append(stats[1])
            t_scores = [round(abs(t_score), 2) for t_score in t_scores]
        # Select top two dimensions of alpha
        t_scores = np.array(t_scores)
        t_scores = np.abs(t_scores)
        max_indices = t_scores.argsort()[::-1]
        idx1 = max_indices[0]
        idx2 = max_indices[1]

        
        # Scatter plot of top two dimensions
        df = pd.DataFrame()
        data = np.stack(preds, axis=1)
        df['Dim1'] = preds[idx1]
        df['Dim2'] = preds[idx2]
        df["dx"] = labels
        fig = plt.figure(figsize=(16,10))
        sns.scatterplot(
            x="Dim1", y="Dim2",
            hue="dx",
            palette=sns.color_palette("hls", 2),
            data=df,
            legend="auto"
        )
        ax = fig.axes[0]
        ax.set_xlabel('Dim1: {}'.format(round(t_scores[idx1], 2)))
        ax.set_ylabel('Dim2: {}'.format(round(t_scores[idx2], 2)))

        plt_title = "Fold: {} {} {} Top Two Dimensions for dx".format(self.agent.k, settype, self.agent.config.exp_id)
        plt.title(plt_title)
        L=plt.legend()
        L.get_texts()[0].set_text('Control')
        L.get_texts()[1].set_text('Disease')
        scores_dir = os.path.join(self.fig_dir, "scores", "separability")
        utils.fs.create_dir(scores_dir)
        fig_fp = os.path.join(scores_dir, plt_title.replace(" ", "_"))
        plt.tight_layout(pad=1)
        plt.savefig(fig_fp)
        # plt.show()
        # Save t-scores and p-values
        score_fp = os.path.join(scores_dir, "{}_{}_{}_class_sep".format(self.agent.config.dataset, settype, self.agent.k))
        t_scores.sort()
        t_scores = t_scores[::-1]
        p_values.sort()
        stacked_scores = np.stack((t_scores, p_values))
        np.savetxt(score_fp, stacked_scores, delimiter=",")

        """ Get summary in last fold """
        # TODO: Change hardcoded K=5
        if self.agent.k == 4:
            max_t_scores = []
            min_p_vals = []
            for i in range(5):
                fold_score_fp = os.path.join(scores_dir, "{}_{}_{}_class_sep".format(self.agent.config.dataset, settype, i))
                scores = np.loadtxt(fold_score_fp,delimiter=",")
                max_t_scores.append(scores[0][0])
                min_p_vals.append(scores[1][0])
            max_t_scores = np.array(max_t_scores)
            min_p_vals = np.array(min_p_vals)
            mean_t_scores = max_t_scores.mean()
            std_t_scores = max_t_scores.std()
            mean_p_vals = min_p_vals.mean()
            std_p_vals = min_p_vals.std()
            max_t_scores = np.append(max_t_scores, [mean_t_scores, std_t_scores])
            min_p_vals = np.append(min_p_vals, [mean_p_vals, std_p_vals])
            # combine to one array for saving
            stacked_scores = np.stack((max_t_scores, min_p_vals))
            summary_score_fp = os.path.join(scores_dir, "{}_{}_{}_class_sep".format(self.agent.config.dataset, settype, "summary"))
            np.savetxt(summary_score_fp, stacked_scores, delimiter=",")

    def plot_top_age(self, is_train=True):
        if is_train:
            settype = "train"
            loader = self.trainloader
        else:
            settype = "val"
            loader = self.valloader
        
        # Get alphas
        state = self._get_scalars_per_tau(loader, settype, plot_delta=False)
        preds = np.array(state["preds"])
        ntaus = preds.shape[0]
        # # normalize to 0-1
        # for i in range(ntaus):
        #     preds[i] = (preds[i] - preds[i].min()) / (preds[i].max() - preds[i].min())
        preds = preds
        metas = state["metas"]      
        # Get t-test scores

        data = pd.DataFrame(preds)
        # Caculate Correlation
        corrs = []
        for pred in preds:
            corr = abs(np.corrcoef(pred, metas["age"])[0][1])
            corrs.append(corr)

        # Select top two dimensions of alpha
        corrs = np.array(corrs)
        max_indices = corrs.argsort()[::-1]
        idx1 = max_indices[0]
        idx2 = max_indices[1]

        # Scatter plot of top two dimensions
        df = pd.DataFrame()
        data = np.stack(preds, axis=1)
        df['Dim1'] = preds[idx1]
        df['Dim2'] = preds[idx2]

        norm_age = (metas["age"] - metas["age"].min()) / (metas["age"].max() - metas["age"].min())
        norm_age = norm_age*(0.9- 0.1) + 0.1 # scale to min 0.1, max =.9
        df["age"] = metas["age"]
        df["norm_age"] = norm_age
        fig = plt.figure(figsize=(16,10))
        # cmap = sns.cubehelix_palette(rot=-.2, as_cmap=True)
        sns.scatterplot(
            x="Dim1", y="Dim2",
            hue="norm_age",
            # size="age",
            palette=sns.cubehelix_palette(as_cmap=True),
            data=df,
            legend=False
        )
        ax = fig.axes[0]
        ax.set_xlabel('Dim1: {}'.format(round(corrs[idx1], 2)))
        ax.set_ylabel('Dim2: {}'.format(round(corrs[idx2], 2)))
        plt_title = "Fold: {} {} {} Top Two Dimensions for Age".format(self.agent.k, settype, self.agent.config.exp_id)
        plt.title(plt_title)

        fig_fp = os.path.join(self.fig_dir, plt_title.replace(" ", "_"))
        plt.tight_layout(pad=1)
        plt.savefig(fig_fp)
        plt.show()

    def get_dcorr(self, is_train=True):
        if is_train:
            settype = "train"
            loader = self.trainloader
        else:
            settype = "val"
            loader = self.valloader
        
        # Get alphas
        state = self._get_scalars_per_tau(loader, settype)
        preds = np.array(state["preds"])
        ntaus = preds.shape[0]

        metas = state["metas"]    
        X = preds.T
        Y = metas["age"].reshape(-1, 1)
        
        # print("Dcorr {}: {}".format(settype, utils.dcor.dcor(X, Y)))
        u_dcor_sqr = dcor.u_distance_correlation_sqr(X, Y)
        u_dcor = math.sqrt(u_dcor_sqr)
        # print(dcor.u_distance_correlation_sqr(X, Y))
        dcor_scores = np.array([u_dcor_sqr, u_dcor])
        scores_dir = os.path.join(self.fig_dir, "scores", "dcorr")
        utils.fs.create_dir(scores_dir)
        score_fp = os.path.join(scores_dir, "{}_{}_{}_dcorr".format(self.agent.config.dataset, settype, self.agent.k))
        np.savetxt(score_fp, dcor_scores, delimiter=",")

        if self.agent.k == 4:
            u_dcor_scores = []
            u_dcor_sqr_scores = []
            for i in range(5):
                fold_score_fp = os.path.join(scores_dir, "{}_{}_{}_dcorr".format(self.agent.config.dataset, settype, i))
                scores = np.loadtxt(fold_score_fp,delimiter=",")
                u_dcor_sqr_scores.append(scores[0])
                u_dcor_scores.append(scores[1])
            u_dcor_scores = np.array(u_dcor_scores)
            u_dcor_sqr_scores = np.array(u_dcor_sqr_scores)
            mean_dcorr = u_dcor_scores.mean()
            std_dcorr = u_dcor_scores.std()
            mean_dcorr_sqr = u_dcor_sqr_scores.mean()
            std_dcorr_sqr = u_dcor_sqr_scores.std()
            u_dcor_scores = np.append(u_dcor_scores, [mean_dcorr, std_dcorr])
            u_dcor_sqr_scores = np.append(u_dcor_sqr_scores, [mean_dcorr_sqr, std_dcorr_sqr])
            # combine to one array for saving
            stacked_scores = np.stack((u_dcor_scores, u_dcor_sqr_scores))
            summary_score_fp = os.path.join(scores_dir, "{}_{}_{}_dcorr".format(self.agent.config.dataset, settype, "summary"))
            np.savetxt(summary_score_fp, stacked_scores, delimiter=",")
       

    def plot_tsne_dims(self, is_train=True):
        if is_train:
            settype = "train"
            loader = self.trainloader
        else:
            settype = "val"
            loader = self.valloader
        
        # Get alphas
        state = self._get_scalars_per_tau(loader, settype, plot_delta=False)
        preds = np.array(state["preds"])
        ntaus = preds.shape[0]

        metas = state["metas"]    
        X = preds.T
        Y = metas["age"].reshape(-1, 1)

        # df_tsne = pd.DataFrame()
        # df_tsne["y"] = metas["age"]
        # tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=5000)
        # tsne_results = tsne.fit_transform(X)
        # df_tsne['tsne-2d-one'] = tsne_results[:,0]
        # df_tsne['tsne-2d-two'] = tsne_results[:,1]
        # plt.figure(figsize=(16,10))
        # sns.scatterplot(
        #     x="tsne-2d-one", y="tsne-2d-two",
        #     hue="y",
        #     palette=sns.color_palette("viridis", as_cmap=True),
        #     data=df_tsne,
        #     legend=False,
        # )
        # plt.show()

        pca = PCA(n_components=3)
        pca_result = pca.fit_transform(X)
        df = pd.DataFrame()
        df['pca-one'] = pca_result[:,0]
        df['pca-two'] = pca_result[:,1] 
        df['pca-three'] = pca_result[:,2]
        print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
        df["y"] = metas["age"]
        fig = plt.figure(figsize=(16,10))
        sns.scatterplot(
            x="pca-one", y="pca-two",
            hue="y",
            palette=sns.color_palette("viridis", as_cmap=True),
            data=df,
            legend=False,
        )
        corr1 = abs(np.corrcoef(df['pca-one'], metas["age"])[0][1])
        corr2 = abs(np.corrcoef(df['pca-two'], metas["age"])[0][1])
        corrs = [corr1, corr2]

        ax = fig.axes[0]
        ax.set_xlabel('Dim1: {}'.format(round(corrs[0], 2)))
        ax.set_ylabel('Dim2: {}'.format(round(corrs[1], 2)))
        plt_title = "Fold: {} {} {} PCA Top Two Dimensions for Age".format(self.agent.k, settype, self.agent.config.exp_id)
        plt.title(plt_title)

        plt.show()
        pdb.set_trace()


class EvalPlotter(BaseEvalPlotter):
    def __init__(self, agent, trainloader, valloader):
        super().__init__(agent, trainloader, valloader)
        resume_dir = os.path.expanduser("~/loca/artifacts/")
        self.k = self.agent.k
        self.resume_dir = os.path.join(resume_dir, 
            str(self.agent.config.rand_seed), 
            self.agent.config.exp_id, 
            str(self.k))

        data_dir = str(Path(agent.config.data_path).parent)
        fp_dx = pd.read_csv(os.path.join(data_dir, "fp_dx.csv"))
        if agent.config.dataset == "adni":
            dxes_to_int = {"CN": 0, "AD": 1}
        elif agent.config.dataset == "aud":
            dxes_to_int = {"C": 0, "E": 1}
        self.data_dir = data_dir
        self.fp_dx = fp_dx
        self.dxes_to_int = dxes_to_int      

        self.fp_age = pd.read_csv(os.path.join(data_dir, "fp_age.csv"))

    # helper for _get_scalers_per_tau
    def _get_scalars_per_tau_get_z(self, X):
        X1 = X[0]
        batch_size = X1.size(0)

        X1 = X1.to(self.agent.device) # first pair
        z1 = self.agent.model.encoder(X1)
        return z1, batch_size

    # helper for _get_scalers_per_tau
    def _get_scalars_per_tau_update_metas(self, X, metas, unique_fps_counter=0, plot_delta=False):
        meta = X[1]
        dxes = meta["dxes"]
        ages = meta["ages"]
        fps = meta["fps"]
        if "fp" not in metas:
            metas["fp"] = []
        if "age" not in metas:
            metas["age"] = []
        if "dx" not in metas:
            metas["dx"] = []
        
        # Convert all classes to 0 or 1
        dxes[0][dxes[0] != 0] = 1

        if plot_delta:
            metas["fp"].extend(fps[0])
            metas["age"].extend(ages[0].tolist())
            metas["dx"].extend(dxes[0].tolist()) 
        else:
            metas["fp"].extend(fps[0])
            metas["age"].extend(ages[0].tolist())
            metas["dx"].extend(dxes[0].tolist()) 
    


class BaselineEvalPlotter(EvalPlotter):
    def __init__(self, agent, trainloader, valloader):
        super().__init__(agent, trainloader, valloader)

    def _get_scalars_per_tau(self, loader, settype, plot_delta=False):
        preds = []
        metas = dict()

        unique_fps_counter = 0
        with torch.no_grad():
            # Get pred age factor and true ages
            for i, X in enumerate(loader):
                z1, batch_size = self._get_scalars_per_tau_get_z(X)
                # if vae
                if hasattr(self.agent.model, "kl_loss_function"):
                    # z1 = self.agent.model.reparameterize(z1[0], z1[1])
                    z1 = z1[0]
                self._get_scalars_per_tau_update_metas(X, metas, unique_fps_counter, plot_delta=False)
                unique_fps_counter += batch_size                
                # Update preds
                preds.append(z1.cpu())
            preds = np.vstack(preds)
            preds = preds.T

            for meta_name, meta in metas.items():
                metas[meta_name] = np.array(meta)

            eval_scalars_state = dict(
                preds = preds,
                metas = metas,
            )

        return eval_scalars_state