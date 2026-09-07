# import torch
# import numpy as np
# import json
import scipy.sparse as sps 
import pandas as pd
from pathlib import Path
from kontext.clustering import gmm_clustering, kmeans_clustering
from kontext.utils import format_key,format_value,spatial_kernel,expression_kernel_at_indices, add_diagonal, memory_of, neighborhood_product, pca_of_mat, cross_weights, compute_niche_context, lr_code_of, get_LR_matrices,accumulate_lr_stats,compute_LR_score




CLUSTERERS = {
    "gmm": gmm_clustering,
    "kmeans": kmeans_clustering,
}


def signature(params):
    se, snn = params.get('sigma_e'), params.get('snn')
    if isinstance(se, list) or isinstance(snn, list):
        se_l = se if isinstance(se, list) else [se] * len(snn)
        snn_l = snn if isinstance(snn, list) else [snn] * len(se)
        rest = {k: v for k, v in params.items() if k not in ('sigma_e', 'snn')}
        return "+".join(signature(dict(snn=s, sigma_e=e, **rest))
                        for s, e in zip(snn_l, se_l))
    return "_".join(f"{format_key(k)}{format_value(v)}"
                    for k, v in params.items() if v is not None)


class Kontext:

    _ARTIFACTS = ("weights", "E_pca", "C_pca", "E_umap", "C_umap", "clustering", "scores",'LR_scores')

    def __init__(self, sigma_e=1, snn=20, alpha=1, n_comps=50,
                save=None, load=None, verbose=False,chunk=50_000):

        self.sigma_e = sigma_e
        self.snn = snn
        self.alpha = alpha
        self.n_comps = n_comps
        self.verbose = verbose
        self.chunk = chunk
        
        self.save = {k: False for k in self._ARTIFACTS}
        self.load = {k: False for k in self._ARTIFACTS}
        if save is not None:
            self.save.update(save)
        if load is not None:
            self.load.update(load)

        # --- list support ---
        self._is_list = isinstance(sigma_e, list) or isinstance(snn, list)
        if self._is_list:
            se_l = sigma_e if isinstance(sigma_e, list) else [sigma_e] * len(snn)
            snn_l = snn if isinstance(snn, list) else [snn] * len(sigma_e)
            self._pairs = list(zip(snn_l, se_l))
            self._pair_sigs = [signature(dict(snn=s, sigma_e=e, n_comps=n_comps))
                            for s, e in self._pairs]

        self.params_C = dict(snn=snn, sigma_e=sigma_e, n_comps=n_comps)
        self.params_E = {**self.params_C, "alpha": alpha}
        self.sig_C = signature(self.params_C)
        self.sig_E = signature(self.params_E)
        self.clust_sig = None



    def fit(self, adata,add_layer=False):
        if self.verbose:
            print('fit')
        self.ensure_weights(adata)
        self.smooth_and_pca(adata, "w", "E",add_layer=add_layer)
        self.maybe_save(adata, "E_pca")
        return adata

    def cluster(self, adata, n_clusters="ground_truth",
                clustering="gmm", seed=0):
        
        self.init_clustering(adata, n_clusters, clustering, seed)
        if not self.maybe_load(adata, "clustering"):
            if "E_pca" not in adata.obsm:
                if not self.maybe_load(adata, "E_pca"):
                    self.fit(adata)
            self.compute_clustering(adata)
        return adata

    def weights(self, adata):
        self.ensure_weights(adata)
        return adata

    def niche_context(self, adata, matrix="E", cell_type_key="global_cell_type",mask_column=None,exocrine=True,verbose=None):
        verbose = self.verbose if verbose is None else verbose
        self.ensure_weights(adata)
    
        w = adata.obsm["w" if matrix == "E" else "wc"]
        if exocrine:
            if mask_column is None:
                mask_column = cell_type_key
            w = cross_weights(w, adata.obs[mask_column])
        compute_niche_context(adata, w, cell_type_key=cell_type_key,verbose=verbose)

# ---- computation (private) ----

    def ensure_weights(self, adata):
        if "wc" in adata.obsm:
            return
        if not self.maybe_load(adata, "weights"):
            self.compute_weights(adata)

    def compute_weights(self, adata):
        if self._is_list:
            wc = None
            ws = None
            for (snn_i, se_i), sig_i in zip(self._pairs, self._pair_sigs):
                wc_i = self._load_intermediate(adata, sig_i)
                if wc_i is None:
                    ws_i = spatial_kernel(adata.obsm["spatial"], snn_i, sparse=True)
                    we = expression_kernel_at_indices(adata.X, ws_i, se_i,chunk=self.chunk)
                    wc_i = ws_i.multiply(we)
                    if self.verbose:
                        print(ws_i.shape,we.shape,'memory of wc_i : ',memory_of(wc_i))
                    self._save_intermediate(adata, wc_i, sig_i)
                ws = ws_i if ws is None else ws + ws_i
                wc = wc_i if wc is None else wc + wc_i
            adata.obsm["ws"] = ws
            adata.obsm["wc"] = wc
            adata.obsm["w"] = add_diagonal(wc, self.alpha)
            if self.verbose:
                print('memory of w : ',memory_of(adata.obsm['w']))
            self.maybe_save(adata, "weights")
            return
        # --- original scalar path (unchanged) ---
        w = None
        if self.snn is not None:
            ws = spatial_kernel(adata.obsm["spatial"], self.snn, sparse=True)
            adata.obsm["ws"] = ws
            w = ws
        if self.sigma_e is not None:
            we = expression_kernel_at_indices(adata.X, ws, self.sigma_e,chunk=self.chunk)
            adata.obsm["we"] = we
            w = we
        if self.snn is not None and self.sigma_e is not None:
            w = ws.multiply(we)
        adata.obsm["wc"] = w
        adata.obsm["w"] = add_diagonal(w, self.alpha)
        if self.verbose:
            print('memory of w : ',memory_of(adata.obsm['w']))
        self.maybe_save(adata, "weights")



    def smooth_and_pca(self, adata, weight_key, name,add_layer=False,pca=True):
        result = neighborhood_product(adata.X, adata.obsm[weight_key])
        if sps.issparse(adata.X):
            result = sps.csr_matrix(result)
        if add_layer:
            adata.layers[name] = result
        if self.verbose:
            print(f'memory of layer {name} : ',memory_of(result),result.shape)
        if pca:
            pca_of_mat(adata, result, n_comps=self.n_comps, key=name)
        if self.verbose:
            print(f'memory of {name}_pca: ',memory_of(adata.obsm[f'{name}_pca']))
            


    def compute_clustering(self, adata, matrix_key="E_pca"):
        fn = CLUSTERERS[self.clust_params["clustering"]]
        params = {k: v for k, v in self.clust_params.items() if k != "clustering"}
        labels = fn(matrix=adata.obsm[matrix_key], **params)

        adata.obs["clustering"] = pd.Categorical(labels)
        adata.obs[self.spatial_domains_signature] = pd.Categorical(labels)
        self.maybe_save(adata, "clustering")



    def init_clustering(self, adata, n_clusters='ground_truth', clustering='gmm', seed=0):
        if n_clusters == "ground_truth" and "n_clusters" in adata.uns.get("data", {}):
            n_clusters = adata.uns["data"]["n_clusters"]
        self.clust_params = dict(clustering=clustering, seed=seed,
                                 n_clusters=n_clusters)
        self.clust_sig = signature(self.clust_params)
        self.spatial_domains_signature = f'{self.sig_E}_{self.clust_sig}'
        adata.uns['signature'] = self.spatial_domains_signature


    # def scores(self,adata,ground_truth='annotation',clustering='clustering',spatial='spatial'):
            
    #         # adata,failed=False,truth_col='annotation',clustering_col='clustering',embedding_key='X_pca',spatial_key='spatial',purity=False):
    #     """
    #     Computes the ARI and NMI scores of a clustering by comparing the columns `truth_col` and `clustering_col` in adata.obs.
    #     If the clustering has failed, the scores will be None
    #     """
    #     if not self.maybe_load(adata,'scores'):
    #         out = {}
    #         out['ARI'] = adjusted_rand_score(adata.obs[ground_truth],adata.obs[clustering])
    #         out['NMI'] = normalized_mutual_info_score(adata.obs[ground_truth],adata.obs[clustering])
    #         out['CHAOS'] = compute_CHAOS(adata.obs[clustering],adata.obsm[spatial])
    #         out['PAS'] = compute_PAS(adata.obs[clustering],adata.obsm[spatial])
    #         out['HOM'] = homogeneity_score(adata.obs[ground_truth],adata.obs[clustering])
    #         out['COM'] = completeness_score(adata.obs[ground_truth],adata.obs[clustering])
            
    #         if self.verbose:
    #             print(out)
    #         if 'kontext' not in adata.uns:
    #             adata.uns['kontext'] = {}

    #         signature = f"{self.sig_E}_{self.clust_sig}"
    #         out['signature'] = signature
    #         adata.uns['kontext']['signature'] = signature
    #         adata.uns['kontext'][signature] = {
    #                                         'data': adata.uns['data'],
    #                                         'params_E':self.params_E,
    #                                         'clust_params':self.clust_params,
    #                                         'paths': {k: str(self.path(adata, k)) for k in self._ARTIFACTS if k != 'LR_scores'},
    #                                         'scores':out,
    #                                         }

    #         self.maybe_save(adata, "scores")
    



    def clustering_of_merged_adata(self,adata,n_clusters, clustering, seed):

        print('pca of merged adata')
        if 'C_pca' in adata.obsm:
            adata.obsm.pop('C_pca')
        if 'E_pca' in adata.obsm:
            adata.obsm.pop('E_pca')
        pca_of_mat(adata,adata.layers['E'],n_comps=self.n_comps,key='E')
        pca_of_mat(adata,adata.layers['C'],n_comps=self.n_comps,key='C')

        
        print('clustering of merged adata')
        self.init_clustering(adata,n_clusters=n_clusters, clustering=clustering, seed=seed)
        loaded = self.load_clustering(adata)
        if not loaded:
            print('not loaded, computing clustering')
            self.compute_clustering(adata,matrix_key='E_pca')




    # ---- generic save / load ----

    def maybe_save(self, adata, key):
        if self.save.get(key, False):
            self._IO[key]["save"](self, adata)

    def maybe_load(self, adata, key):
        if not self.load.get(key, False):
            return False
        try:
            return self._IO[key]["load"](self, adata)
        except Exception as e:
            log.debug("Could not load %s: %s", key, e)
            return False



    # ---- concrete I/O ----

    def save_weights(self, adata):
        sps.save_npz(self.path(adata, "weights"), sps.csr_matrix(adata.obsm["wc"]))

    def load_weights(self, adata):
        p = self.path(adata, "weights")
        if not p.exists():
            return False
        adata.obsm["wc"] = sps.load_npz(p)
        adata.obsm["w"] = add_diagonal(adata.obsm["wc"], self.alpha)
        return True

    def save_pca(self, adata, key):
        pd.DataFrame(adata.obsm[key], index=adata.obs.index).to_csv(self.path(adata, key))

    def load_pca(self, adata, key):
        p = self.path(adata, key)
        if not p.exists():
            return False
        adata.obsm[key] = pd.read_csv(p, index_col=0).to_numpy()
        return True

    def save_clustering(self, adata):
        adata.obs["clustering"].to_csv(self.path(adata, "clustering"))

    def load_clustering(self, adata,signed=False):
        p = self.path(adata, "clustering")
        try:
            df = pd.read_csv(p, index_col=0)
            if len(df) != len(adata):
                log.warning("Cached clustering size mismatch")
                return False
            adata.obs["clustering"] = pd.Categorical(df.iloc[:, 0])
            if signed:
                signature =str(p).split('/')[-1].replace('.csv','')
                adata.obs[signature] = pd.Categorical(df.iloc[:, 0])
                adata.uns['signature'] = signature
            return True
        except Exception as e:
            if self.verbose:
                print(e)
                print(p)        
            return False
            
    def save_scores(self, adata):
        sig = adata.uns['kontext']['signature']
        out = adata.uns['kontext'][sig]
        p = self.path(adata, "scores")
        with open(p, "w") as f:
            json.dump(out, f, indent=2)

    def load_scores(self, adata):
        p = self.path(adata, "scores")
        try:
            with open(p, "r") as f:
                data = json.load(f)
            # Restore the same state as scores() creates
            if 'kontext' not in adata.uns:
                adata.uns['kontext'] = {}
            # print(data)
            signature = f"{self.sig_E}_{self.clust_sig}"
            adata.uns['kontext']['signature'] = signature
            adata.uns['kontext'][signature] = {
                'params_E': data['params_E'],
                'clust_params': data['clust_params'],
                'scores': data['scores']
            }
            return data
        except Exception as e:
            if self.verbose:
                print(f'scores not loaded {e} {p}')
                print(e)
                print(p)
            return None

    
    # dispatch table
    _IO = {
        "weights":    {"save": save_weights,    "load": load_weights},
        "E_pca":      {"save": lambda s, a: s.save_pca(a, "E_pca"),
                        "load": lambda s, a: s.load_pca(a, "E_pca")},
        "C_pca":      {"save": lambda s, a: s.save_pca(a, "C_pca"),
                        "load": lambda s, a: s.load_pca(a, "C_pca")},
        "E_umap":      {"save": lambda s, a: s.save_pca(a, "E_umap"),
                        "load": lambda s, a: s.load_pca(a, "E_umap")},
        "C_umap":      {"save": lambda s, a: s.save_pca(a, "C_umap"),
                        "load": lambda s, a: s.load_pca(a, "C_umap")},
        "clustering": {"save": save_clustering, "load": load_clustering},
        "scores": {"save": save_scores, "load": load_scores},
    }

    # ---- paths ----

    def base(self, adata):
        d = adata.uns["data"]
        return Path(f"/pasteur/appa/scratch/aozierla/"
                    f"{d['dataset']}/{d['sample']}/{d['pp_str']}/context")

    def path(self, adata, kind):
        base = self.base(adata)
        mapping = {
            "weights":    (base / "weights",    f"{self.sig_C}.npz"),
            "E_pca":      (base / "pca_E",      f"{self.sig_E}.csv"),
            "C_pca":      (base / "pca_C",      f"{self.sig_C}.csv"),
            "E_umap":      (base / "umap_E",      f"{self.sig_E}.csv"),
            "C_umap":      (base / "umap_C",      f"{self.sig_C}.csv"),
            "clustering": (base / "clustering",
                        f"{self.sig_E}_{self.clust_sig}.csv"),
            "scores":     (base / "scores",     f"{self.sig_E}_{self.clust_sig}.json"),
        }
        folder, fname = mapping[kind]
        folder.mkdir(parents=True, exist_ok=True)
        return folder / fname
    
    def _intermediate_path(self, adata, sig):
        folder = self.base(adata) / "weights"
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{sig}.npz"

    def _load_intermediate(self, adata, sig):
        if not self.load.get("weights", False):
            return None
        p = self._intermediate_path(adata, sig)
        try:
            mat = sps.load_npz(p) 
            return mat
        except:
            return None


    def _save_intermediate(self, adata, wc, sig):
        if self.save.get("weights", False):
            sps.save_npz(self._intermediate_path(adata, sig), sps.csr_matrix(wc))





    def LR_scores(self, adata, communication_type="exocrine",groups="cell_type", w_key="wc", groups_of_interest=None):
            """
            Compute per-cell ligand-receptor score matrices (stored in adata.obsm)
            and their obs-/pair-level summaries.

            Each score matrix S is cached individually: if loading is enabled the
            matrix is read from disk instead of recomputed, which also avoids
            building the underlying LR context matrices.
            """

            

            pairs = adata.uns["context_LR"]["pairs"]
            code  = lr_code_of(communication_type, groups)

            groups_list = (adata.obs[groups].unique() if groups_of_interest is None else groups_of_interest)
            cell_type_masks = {ct: (adata.obs[groups] == ct).values for ct in groups_list}

            # one score matrix for single-context types, one per group otherwise
            if communication_type in ("all", "exocrine", "endocrine"):
                score_specs = [(f"LR_R_{code}", code, f"zC_lr_{code}")]
            else:
                score_specs = [(f"LR_R_{code}_c{t}", f"{code}_c{t}", f"zC_lr_{code}_c{t}")
                            for t in groups_list]

            obs_stats,  obs_names  = [], []
            pair_stats, pair_names = [], []

            get_LR_matrices(adata, communication_type, groups, w_key,groups_of_interest=groups_of_interest)
            for score_key, stat_code, context_key in score_specs:
                S = self.maybe_load_lr_score(adata, score_key)
                if S is None:
                    S = compute_LR_score(adata,context_key,score_key)
                    self.maybe_save_lr_score(adata, score_key, S)
                accumulate_lr_stats(S, score_key, stat_code, cell_type_masks,
                                        obs_stats, obs_names, pair_stats, pair_names)
                
                if self.verbose:
                    print(f"size of {score_key}: {memory_of(S)}")

            df_obs   = pd.DataFrame(obs_stats,  index=obs_names,  columns=adata.obs_names).T
            df_pairs = pd.DataFrame(pair_stats, index=pair_names, columns=pairs.index).T
            adata.obs = adata.obs.merge(df_obs, right_index=True, left_index=True)
            adata.uns["context_LR"]["pairs"] = pd.concat([pairs, df_pairs], axis=1)
            return code

    # ---- LR stats helpers ----


    # ---- LR score I/O ----

    def _lr_score_path(self, adata, score_key):
        folder = self.base(adata) / "LR_scores"
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{score_key}.npy"

    def maybe_save_lr_score(self, adata, score_key, S):
        if self.save.get("LR_scores", False):
            np.save(self._lr_score_path(adata, score_key), S)

    def maybe_load_lr_score(self, adata, score_key):
        if not self.load.get("LR_scores", False):
            return None
        p = self._lr_score_path(adata, score_key)
        if not p.exists():
            return None
        try:
            return np.load(p)
        except Exception as e:
            if self.verbose:
                print(f"LR score not loaded {e} {p}")
            return None
