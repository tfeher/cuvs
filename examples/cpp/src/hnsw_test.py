import hnswlib 
import numpy as np
import os


def dtype_from_filename(filename):
    ext = os.path.splitext(filename)[1]
    if ext == ".fbin":
        return np.float32
    if ext == ".hbin":
        return np.float16
    elif ext == ".ibin":
        return np.int32
    elif ext == ".u8bin":
        return np.ubyte
    elif ext == ".i8bin":
        return np.byte
    else:
        raise RuntimeError("Not supported file extension" + ext)


def suffix_from_dtype(dtype):
    if dtype == np.float32:
        return ".fbin"
    if dtype == np.float16:
        return ".hbin"
    elif dtype == np.int32:
        return ".ibin"
    elif dtype == np.ubyte:
        return ".u8bin"
    elif dtype == np.byte:
        return ".i8bin"
    else:
        raise RuntimeError("Not supported dtype extension" + dtype)


def memmap_bin_file(
    bin_file, dtype, shape=None, mode="r", size_dtype=np.uint32
):
    extent_itemsize = np.dtype(size_dtype).itemsize
    offset = int(extent_itemsize) * 2
    if bin_file is None:
        return None
    if dtype is None:
        dtype = dtype_from_filename(bin_file)

    if mode[0] == "r":
        a = np.memmap(bin_file, mode=mode, dtype=size_dtype, shape=(2,))
        if shape is None:
            shape = (a[0], a[1])
        else:
            shape = tuple(
                [
                    aval if sval is None else sval
                    for aval, sval in zip(a, shape)
                ]
            )

        return np.memmap(
            bin_file, mode=mode, dtype=dtype, offset=offset, shape=shape
        )
    elif mode[0] == "w":
        if shape is None:
            raise ValueError("Need to specify shape to map file in write mode")

        print("creating file", bin_file)
        dirname = os.path.dirname(bin_file)
        if len(dirname) > 0:
            os.makedirs(dirname, exist_ok=True)
        a = np.memmap(bin_file, mode=mode, dtype=size_dtype, shape=(2,))
        a[0] = shape[0]
        a[1] = shape[1]
        a.flush()
        del a
        fp = np.memmap(
            bin_file, mode="r+", dtype=dtype, offset=offset, shape=shape
        )
        return fp

dim = 1536
n_rows = 1000000
p = hnswlib.Index(space='l2', dim=dim)
print("\nLoading index\n")

# Increase the total capacity (max_elements), so that it will handle the new data
p.load_index("hnsw_index.bin", max_elements = n_rows)
#p.load_index("/tmp/ace_build/hnsw_index.bin", max_elements = n_rows)

#p.load_index("index/openai_5M/cagra_hnswlib/ace.ibin", max_elements = n_rows)
print("index loaded")

p.set_ef(800)

#queries = np.float32(np.random.random((10, dim)))

dataset = memmap_bin_file('openai_5M/base.5M.fbin', np.float32, shape=(n_rows, dim))
queries = np.asarray(dataset[:10,:])

print("searching neighbors")
neighbors, distances = p.knn_query(queries, k=10)
print("finished neighbor search\nNeighbors idx")
print(neighbors)
print('distances')
print(distances)


