import hnswlib 
import numpy as np

dim = 1536
p = hnswlib.Index(space='l2', dim=dim)
print("\nLoading index\n")

# Increase the total capacity (max_elements), so that it will handle the new data
p.load_index("hnsw_index.bin", max_elements = 1000000)
#p.load_index("/tmp/ace_build/hnsw_index.bin", max_elements = 1000000)

#p.load_index("index/openai_5M/cagra_hnswlib/ace.ibin", max_elements = 1000000)
print("index loaded")

p.set_ef(120)

queries = np.float32(np.random.random((10, dim)))
print("searching neighbors")
res = p.knn_query(queries, k=10)
print("finished neighbor search")
print(res)