from __future__ import absolute_import
import numpy as np
from ann_benchmarks.algorithms.base import BaseANN
import sys
sys.path.insert(0, "/home/app/hora-python/hora/target/release")
print(sys.path)
import hora
import time


class Hora(BaseANN):
    def __init__(self, metric, params):
        self._metric = metric
        self.t = params.get("index", "")
        
        # HNSW
        self.max_item = params.get("max_item", 100000)
        self.n_neigh = params.get("n_neigh", 16)
        self.n_neigh0 = params.get("n_neigh0", 32)
        self.ef_build = params.get("ef_build", 500)
        self.ef_search = params.get("ef_search", 16)
        self.has_deletion = params.get("has_deletion", False)
        
        #SSG
        self.angle = params.get("angle", 60.0)
        self.init_k = params.get("init_k", 20)
        self.index_size = params.get("index_size", 20)
        self.neighbor_neighbor_size = params.get("neighbor_neighbor_size", 30)
        self.root_size = params.get("root_size", 256)

    def fit(self, X):
        print(hora.__dict__)
        if self.t == "BPTIndex":
            self.index = hora.BPTIndex(int(X.shape[1]), 6, -1)
        elif self.t == "HNSWIndex":
            self.index = hora.HNSWIndex(int(X.shape[1]), self.max_item, self.n_neigh,
                                        self.n_neigh0, self.ef_build, self.ef_search, self.has_deletion)
        elif self.t == "PQIndex":
            self.index = hora.PQIndex(int(X.shape[1]), 10, 4,
                                      100)
        elif self.t == "SSGIndex":
            self.index = hora.SSGIndex(
                int(X.shape[1]), self.neighbor_neighbor_size, self.init_k, self.index_size, self.angle, self.root_size)
        else:
            self.index = hora.BruteForceIndex(int(X.shape[1]))

        for i, x in enumerate(X):
            self.index.add([float(item) for item in x.tolist()], i)
        self.index.build(self._metric)

    def query(self, v, n):
        return self.index.search_np(np.float32(v), n)

    def set_query_arguments(self, epsilon):
        print(epsilon)

    def __str__(self):
        return 'Hora(hora-search) {}'.format(time.time())
