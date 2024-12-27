import time
import gc
from os import cpu_count
import numpy as np
import pandas as pd
import dask.dataframe as dd
import polars as pl
import modin.pandas as mpd
import datatable as dt


# Generate sample data
df = pd.DataFrame({
    "A": np.random.rand(300000000),
    "B": np.random.randint(0, 8, 300000000)
})

# Basic pandas
tic = time.time()
mean = df.groupby("B")["A"].mean()
print(mean)
print("Pandas time:", time.time() - tic)

# Dask
ddf = dd.from_pandas(df, npartitions=cpu_count())
tic = time.time()
mean = ddf.groupby("B")["A"].mean().compute()
print("Dask time:", time.time() - tic)

# Garbage collection to save RAM
del ddf
gc.collect()

# Polars
pdf = pl.DataFrame(df)
tic = time.time()
mean = pdf.group_by("B").agg(pl.col("A").mean())
print("Polars time:", time.time() - tic)

del pdf
gc.collect()

# Modin
mpdf = mpd.DataFrame(df)
tic = time.time()
mean = mpdf.groupby("B")["A"].mean()
print("Modin time:", time.time() - tic)

del mpdf
gc.collect()

# Datatable
dt_df = dt.Frame(df)
tic = time.time()
mean = dt_df[:, dt.mean(dt.f.A), dt.by("B")]
print(mean)
print("Datatable time:", time.time() - tic)
