from numba import jit, prange
import pandas as pd
import time
import numpy as np

def myfunc(x):
    return 2 * x["A"] if x["B"] % 2 == 0 else x["A"]

@jit(nopython=True, parallel=True)
def myfunc_numba(a_values, b_values):
    result = np.empty(len(a_values))
    for i in prange(len(a_values)):
        result[i] = 2 * a_values[i] if b_values[i] % 2 == 0 else a_values[i]
    return result

# Generate sample data
df = pd.DataFrame({
    "A": np.random.rand(10000000),
    "B": np.random.randint(0, 8, 10000000)
})

tic = time.time()
result = df.apply(myfunc, axis=1)
print("Pandas time:", time.time() - tic)

tic = time.time()
result = myfunc_numba(df["A"].values, df["B"].values)
print("Pandas with numba time:", time.time() - tic)
