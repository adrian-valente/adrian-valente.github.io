---
layout: post
title: "Parallelized python - a (hopefully more) complete guide"
excerpt: Python has a ton of solutions to parallelize loops on several CPUs, and the choice became even richer with python 3.13 this year. I had written a post 4 years ago on multiprocessing, but it comes short of presenting the available possibilities. Here is an attempt to bring all of them finally in a single place.
use_math: true
---

Parallelization is a very powerful technique, whether you have a laptop with 8 CPU cores or a powerful HPC machine with hundreds of them. Even if you can bring the execution time of a certain simulation from 15 to 2 minutes it can totally change your working day and keep you in the flow, or dramatically improve your statistical power. In python, it is very easy to harness these capabilities, yet I often see people who don't attempt to do so, often because the subject seems complex from afar. It also doesn't help that python offers so many solutions for this single problem, and people often get lost. This is why I had written [this blog post](./2020-02-18-git-primer.md) to demonstrate how easy it can be, but it presented a single possibility (which is also pretty much the worse one), so this blog post aims to cover most of them.

# Introductory notes
For the first four sections of this post, we will parallelize this loop that does a Monte-Carlo simulation of a 2D random walk :
```py
import seaborn as sns
import matplotlib.pyplot as plt
from time import time
import random

T = 1000
endpoints = []
for i in range(10000):
    x = [0, 0]
    for t in range(T):
        x[0] += random.choice((-1, 1))
        x[1] += random.choice((-1, 1))
    endpoints.append(x)
x0, x1 = zip(*endpoints)
sns.jointplot(x=x0, y=x1, kind='hex')
plt.show()
```

Probabilistic simulations, but also bayesian inference, optimization procedures or working on large arrays of data are among the processes that are most suited for parallelization. To see why, notice that above, each instance of the outer loop (`for i in...`) is perfectly independent of the other. They only correspond to individual simulations of an identical process, which is a type of **embarrassingly parallel** problem. By contrast, the inner loop (`for t in...`) is not parallelizable, since each iteration depends on the previous one.

Once you have identified a loop that can be parallelized, you simply need to understand the two core ways in which any software can be executed in parallel:
- you can create several **processes**, that are independent programs with their own chunk of RAM memory, and in the case of python their own instance of the python interpreter. Several of them can be performing operations at the exact same nanosecond on different CPU cores, however they have two disadvantages: (1) there is an overhead to their creation and destruction, due to complex interactions with the operating system and (2) since they don't share memory, they can't easily exchange information between one another. For reasons I will explain below, they are the preferred solution in python, but you should know that they are not very practical for very small operations or if their inputs or outputs are very big arrays.
- you can also create **threads**. Contrarily to processes, the operating system sees threads as part of a single program, and so they all share the same chunk of memory. This removes both the overhead and the difficulties in exchanging information, but the fact that they share memory means several threads could try to modify the same bit at the same time, causing very tough bugs known as **race conditions**. In fact these bugs can be so complex, that python has decided that only one thread of the interpreter can run at the same time, a measure known as the *Global Interpreter Lock* (GIL), and that it is sometimes possible to lift since python 3.13 (October 2024). This means that if you use threads in python, normally only a single line of python can be executed at the same time, and it is not worth using them except in 3 cases: (1) using a library written in C that lifts the GIL (like numpy), (2) waiting for slow I/O operations (like networking, although async programming is a good alternative in this case) or (3) if you are daring enough to lift the GIL with the new python.

Now that you know the concepts, let's dive into the solutions.

# 1. Multiprocessing
This is the one presented in [this earlier post](./2020-04-15-python-multiprocessing.md), and it is very straigthforward to use. As the name suggests, it will use processes. Basically, wrap your task in a function, and then use the `Pool.map` method:
```py
import multiprocessing as mp

def task(T):
    x = [0, 0]
    for t in range(T):
        x[0] += random.choice((-1, 1))
        x[1] += random.choice((-1, 1))
    return x

with __name__ == '__main__':
    with mp.Pool(mp.cpu_count()) as pool:
        args = [1000] * 100000
        endpoints = pool.map(task, args)
    # ... plot results
```

These are the key functions you need to know about:
- `Pool(n_processes)` [[doc](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool)]: this will create the set of processes that will perform your tasks. Here I explicitly gave `mp.cpu_count()` which will be the number of available CPU cores this is the default argument if you don't set anything. In general, there is no need to go above that. Your tasks will be distributed among the created processes, reducing the process creation overhead (there aren't 10k processes created here, only `cpu_count()`). 
- `Pool.map(task, args[, chunksize])`[[doc](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.map)]: where `task` is a function that takes a single argument and `args` is an iterable of arguments. This will create one version of the task for each of the given argument, and distribute them among the pool of processes, and return a list of the returned values when all tasks are done. The optional `chunksize` argument allows to submit chunks of, say, 100 tasks instead of one by one, which is sometimes faster. If your task takes several arguments, you should use:
- `Pool.starmap(task, args)`[[doc](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.starmap)]: where this time `args` is an iterable of tuples, each tuple containing the arguments for one task.
- Don't forget to call `Pool.close` when you're done or define it in a `with` statement as above.

Important remark: this library, although very simple to use, can lead to infamous errors such as `Can't get attribute 'task' on <module '__main__' (built-in)>` or `Can't pickle <function <lambda> on ...`. It is indeed very fragile, and will not work in notebooks, nor with lambda functions, nor in many special cases. It will also cause all instructions outside of the `if __name__ == '__main__':` block to be executed by all processes, which can lead to very strange bugs. If you have such issues, you can try some of the other solutions below like `joblib`.

Side-note: the line `args = [1000] * 100000` can be replaced by `args = itertools.repeat(1000, 100000)` to save on memory.

# 2. concurrent.futures
This one [[doc](https://docs.python.org/3/library/concurrent.futures.html)] is very similar (and keeps similar issues), but more minimalistic and clean:
```py
from concurrent.futures import ProcessPoolExecutor

def task(T):
    # as above

with __name__ == '__main__':
    with ProcessPoolExecutor() as executor:
        args = [1000] * 100000
        endpoints = executor.map(task, args)
    # ... plot results
```

As before we used a with statement to define a pool of processes, that by default has the same number of processes as the number of available CPUs.

The advantages are:
- single `map` function, no need to choose between `map` and `starmap`.
- the `ProcessPoolExecutor` can be replaced by `ThreadPoolExecutor` without further changes to switch from processes to threads (try it! Then try it in a python version with the GIL deactivated. You can obtain it in a conda environment created as follows: `conda create -n py313 python=3.13 python-freethreading -c conda-forge/label/python_rc -c conda-forge`. For me it was much slower than expected sadly, even though it used all my cores, but this is still very experimental).

Remark: the `concurrent.futures` module is a wrapper around the `multiprocessing` module. Although it fixes some of its shortcomings, it also has a lot of the same issues, like the need to be in a `if __name__ == '__main__':` block, or the incapacity to use lambda functions.

# 3. joblib
This one [[user guide](https://joblib.readthedocs.io/en/stable/parallel.html)] finally solves all the issues described above, so fall back onto it if you are having trouble with the ones above:
```py
from joblib import Parallel, delayed

def task(T):
    # as above

endpoints = Parallel(n_jobs=-1)(delayed(task)(1000) for _ in range(100000))
# ... plot results
```
The syntax is weirder but quite elegant in fact: 
- `Parallel(n_jobs=-1)` [[doc](https://joblib.readthedocs.io/en/stable/generated/joblib.Parallel.html)] creates a pool of processes, where `-1` means to use all available CPUs.
- `delayed(task)(1000)` will return an object that will call `task(1000)` in a process when the `Parallel` object will decide it. Similarly, if you had several arguments, you would write something like `delayed(task)(a, b)`.
- the block `delayed(task)(1000) for _ in range(100000)` is a generator expression (like a list comprehension, but it creates an iterator instead of a list. See [here](https://dbader.org/blog/python-generator-expressions) for a blog post, [here](https://docs.python.org/3/reference/expressions.html#generator-expressions) for the reference doc). It creates the tasks in a very versatile way that can combine arguments that are iterated over and arguments that are fixed, as well as positional and keyword arguments. For example this would work: `delayed(task)(i, y=12, z=j) for i in range(100) for j in range(i)`.

Joblib is truly a wonder, and I encourage you to check [its documentation](https://joblib.readthedocs.io/en/stable/parallel.html) to know more. You can for example switch to threads or compute on data as it is being generated. A lot of its robustness stems from the `loky` library which it uses by default, and that will be our contender number 3.5:

## 3.5. loky
[Loky](https://github.com/joblib/loky) is a very nice project with which you can use the same interface as `concurrent.futures` but that will be robust to many of its issues:
```py
from loky import ProcessPoolExecutor

with ProcessPoolExecutor() as pool:
    endpoints = pool.map(task, [1000] * 100000)
```
Although the recommended way to use it is with the `get_reusable_executor` function that wraps around the `ProcessPoolExecutor` class:
```py
from loky import get_reusable_executor

pool = get_reusable_executor()
endpoints = pool.map(task, [1000] * 100000)
```

As loky and joblib are related projects, it seems reasonable to use joblib instead by default, but loky comes in handy if you have code written for `concurrent.futures` and you want to immediately make it robust.

# 4. numba
[Numba](https://numba.readthedocs.io/en/stable/user/5minguide.html) is another wonder of the python universe that I can suggest you to look at today if you didn't already. Essentially it has two killer features that each alone could justify using the framework: (1) it puts into everyone's hands the capacity to write compiled super-fast code from python, without having to deal with Cython or C's pointers and mallocs, and (2) it trivially allows the parallelization of loops for you using multi-threading (yes, real multi-threading even without python3.13!! That's because the code is compiled, so it avoids the GIL). I can only scratch the surface here of what it can do for you, but I'll demonstrate the above task accelerated with numba:

```py
from numba import jit, prange

@jit(nopython=True)
def task(T):
    x = [0, 0]
    for t in range(T):
        x[0] += 1 if random.random() > 0.5 else -1
        x[1] += 1 if random.random() > 0.5 else -1
    return x

@jit(nopython=True, parallel=True)
def parallel_mc(N, T):
    results = np.empty((N, 2))
    for i in prange(N):
        results[i, :] = np.array(task(T))
    return results

results = parallel_mc(10000, 1000)
```

The key functionalities we have used are:
* `@jit([nopython=True, parallel=True])`: this is a decorator, which should go right above the function you want compiled to low-level code. `nopython=True` will be the by default mode you should always be using. Note that the functions that can be compiled need to be written in a specific subset of python, typically they will only support a certain number of pure-python functionalities (detailed [here](https://numba.pydata.org/numba-doc/dev/reference/pysupported.html#supported-python-features)), which include lists (of things of only one type), or sets (of things of only one type as well), but not dictionaries for instance (although there is a `numba.typed.Dict` you can use instead0. They also allow the use of numerous numpy functions, and certain functions from the `math` and `random` modules among others (but not `random.choice`, which is why we switched to the `random.random` function above). Basically, your compiled code should be as bare-bones as possible.
* `prange`: This function is a drop-in replacement for `range` that will run the loop in parallel. It's exactly as simple as shown above!

Important warning: you might notice we have chosen to create a fixed-size array here, and then each iteration of the loop knows in advance where it will need to write its results. We could have wanted to define results as a list and use `.append` in each iteration, but don't do it because it creates a risk of **race condition**. Indeed, we are using real threads here, so this risk that was absent with processes or the GIL is now back! Here are the details if you want them: remember each iteration is running on a separate thread, and imagine the following situation: thread 1 starts appending, so it creates a new position for the n-th element; thread 2 also starts creating a position for the n-th element, since it doesn't know thread 1 is doing it already; the result is the list will have two n-th elements, and will likely contain only that of the last thread finishing its operation, the other one being lost. To avoid race conditions like this one, you should always ask yourself if the operation of one thread depends on synchronization with the others or not. If the answer is yes, find a way around until it doesn't. Note that here I am not sure if the `.append` will actually create race conditions or not, but the mere doubt should be enough for you to avoid it.

# 5. Summary of function-based parallelization
Let's summarize all the above solutions for function-based parallelization in a table:

| Library | Simplicity of use | Threads or Processes | Robustness | Main difficulties |
|---------|-------------------|----------------------|------------|-------------------|
| multiprocessing | Rather simple, difference `map` and `starmap`. | Processes | Not great, not usable in notebooks or with lambdas. | Lack of robustness. |
| concurrent.futures | Very simple, single `map` function. | Processes or threads | Same as multiprocessing | Lack of robustness, and presence of GIL with threads. |
| joblib | Weirder at the beginning, but trivial once you get the hang of it. | Processes or threads | Very robust, can be used anywhere. | Still the GIL with threads. |
| loky | Same as concurrent.futures | Processes | Very robust, can be used anywhere. | Very limited features. |
| numba | Looks simple, but will require experience to know what code can be compiled, and how to avoid race conditions. | Threads (GIL-free!) | Not usable for lambdas, and code will be able to use limited features, and often with a hidden use of the typing system that can cause unexpected bugs. | One needs to be careful about possible race conditions and things that can be compiled. |

# 6. What happens with numpy?
Scientific libraries like numpy or scipy can bring all sorts of complications by sometimes already doing parallelization under the scene, and sometimes also freeing the GIL which allows for multithreading. To experiment a bit, consider the following code that does matrix operations:
```py
import numpy as np
from joblib import Parallel, delayed
import time

def matrix_stuff(n):
    A = np.random.rand(n, n)
    return np.sum(np.log(A))

if __name__ == '__main__':
    tic = time.time()
    results = [matrix_stuff(1000) for _ in range(1000)]
    print('Non-parallelised time: {:.2f} s'.format(time.time() - tic))
    
    tic = time.time()
    results = Parallel(n_jobs=-1)(delayed(matrix_stuff)(1000) for _ in range(1000))
    print('Parallelized time: {:.2f} s'.format(time.time() - tic))
```
And we will be experiment with different `matrix_stuff` functions. With this first experiment, things went faster with the parallelized loop, and I noticed that the first part was using only one CPU core and the second part the eight of them. That's your best use-case, a situation where nothing needs to change compared to the parallelization of standard python code. Now, try with this function instead:
```py
def matrix_stuff(n):
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    return np.sum(A @ B)
```
There, I notice that in the first half, 4 CPUs are already used! This is because numpy uses BLAS for matrix multiplication which can do some parallelization. I still get faster results with the parallelized loop, but the difference is smaller, and in fact works better with only 2 workers (since I have 8 CPUs). Note this will depend on the machine that it is run onto, and its versions of numpy and BLAS. Now let's try this (CAREFUL: if you don't have 16GB of RAM, reduce the number of iterations):
```py
def matrix_stuff(n):
    A = np.random.rand(n, n)
    B = np.random.rand(n, n)
    return np.log(A + B)
```
You should see the parallelized code now taking longer! And that's despite the first part using only one CPU core! This is because we are now returning big matrices, and remember processes can't share memory, so all this data has to travel back to the main process, slowing down the operations. But if you try to add the `prefer="threads"` argument to the `Parallel` constructor to use threads instead:
```py
Parallel(n_jobs=-1, prefer="threads")(delayed(matrix_stuff)(1000) for _ in range(1000))
```
Now you should see a decrease. Indeed, the operation `np.log(A + B)` when running in the numpy C code will release the GIL, allowing for some overlap between these calculations in different threads (still far from 100% but still). Data can stay in the same place in memory, limiting the overhead we saw in processes. 

I hope all this illustrates the possible pitfalls if you try to parallelize code that makes heavy use of numpy or a similar library. It can be a good idea to check first if the operation you are interested in already uses several cores, and to think about how much data will need to be shared (typically a minimal amount if you want to use processes).

# 7. The fast pandas family
Pandas is great, but (as the name unintentionally suggest?) not so fast: it doesn't internally multi-thread and doing your own parallelization with processes would require copying a lot of data around which could end up being even slower. To go around this, the main solution will be to either change to a more friendly pandas-like library, and there are actually a lot more than I initially thought. I will talk here about [polars](https://docs.pola.rs/), [dask](https://docs.dask.org/en/stable/10-minutes-to-dask.html), [modin](https://github.com/modin-project/modin), and [datatable](https://datatable.readthedocs.io/en/latest/), which all handle CPU parallelization, but you might also want to check out [cuDF](https://docs.rapids.ai/api/cudf/stable/) if you have a GPU. I put a table at the end of this section to summarize the differences.

 <div class="centrer">
  <img src="{{site.url}}/assets/parallel2/fast_pandas.gif" width="320"/>
  <br/>
  Who doesn't want faster pandas? <a href="https://www.youtube.com/watch?v=sGF6bOi1NfA" target="_blank" rel="noopener noreferrer">Source</a>
  </div>
  <br/>


When you want to do operations that can be coded entirely in pandas, moving to one of these libaries is a straigthforward and efficient way to use all your cores:
```py
import time
import numpy as np
import pandas as pd

import polars as pl

df = # Generate a dataframe

# Example operation in pandas:
mean = df.groupby("B")["A"].mean()
print("Pandas time:", time.time() - tic)

# Let's try dask: when defining the dataframe, the `npartitions`
# argument is the number of chunks the dataframe will be split into,
# and thus number of parallel workers that can process it. 
import dask.dataframe as dd
ddf = dd.from_pandas(df, npartitions=cpu_count())
mean = ddf.groupby("B")["A"].mean().compute()

# Now with polars:
import polars as pl
pdf = pl.DataFrame(df)
mean = pdf.group_by("B").agg(pl.col("A").mean())

# With modin, amazingly identical to pandas:
import modin.pandas as mpd
mdf = mpd.DataFrame(df)
mean = mdf.groupby("B")["A"].mean()

# And finally datatable:
import datatable as dt
dt_df = dt.Frame(df)
mean = dt_df[:, dt.mean(dt.f.A), dt.by("B")]
```

[This](https://github.com/adrian-valente/adrian-valente.github.io/blob/master/assets/parallel2/fast_pandas_test.py) is the complete code you can try at home to check out the speed of each alternative (again, be careful with your memory, and you will need to pip install `polars`, `"dask[dataframe]"`, `"modin[all]"`). In my experience, polars and datatable really work best for all situations, but in terms of syntax they both deviate significantly from pandas. Dask and modin are good alternatives, with modin offering the advantage of a 0-change to syntax. This is not a complete benchmark but there are a few online, not always as complete and independent as I'd like, for example [here](https://duckdblabs.github.io/db-benchmark/), which is a more recent version of [this one](https://h2oai.github.io/db-benchmark/), or [here](https://www.datacamp.com/tutorial/benchmarking-high-performance-pandas-alternatives).

The above applies well if you can write your operation in pure-pandas code, but that is not always possible and sometimes you need to rely on a python function that is mapped to a Series or Dataframe with the `.apply()` method. In this case, since you are falling back to GIL-blocked python code, even dask or polars would normally not be able to help you, but there is a last resort solution: compile your function with numba, and apply it directly to the `.values` arrays that compose your dataframe. Here is an example (complete code [here](https://github.com/adrian-valente/adrian-valente.github.io/blob/master/assets/parallel2/numba4pandas.py)):

```py
from numba import jit
import pandas as pd

def myfunc(x):
    return 2 * x["A"] if x["B"] % 2 == 0 else x["A"]

@jit(nopython=True, parallel=True)
def myfunc_numba(a_values, b_values):
    result = np.empty(len(a_values))
    for i in prange(len(a_values)):
        result[i] = 2 * a_values[i] if b_values[i] % 2 == 0 else a_values[i]
    return result

# Define df as above...

tic = time.time()
result = df.apply(myfunc, axis=1)
print("Pandas time:", time.time() - tic)

tic = time.time()
result = myfunc_numba(df["A"].values, df["B"].values)
print("Pandas with numba time:", time.time() - tic)
```
See [here](https://docs.pola.rs/user-guide/expressions/user-defined-python-functions/#missing-data-is-not-allowed-when-calling-generalized-ufuncs) for similar features with polars.

Summary table of the pandas family:

| Library | Syntax | Speed | Other features |
|---------|--------|-------|----------------|
| polars | Quite different from pandas, requires some time to learn but great investment. Bears similarities to R dplyr. | Extremely fast | Lazy evaluation, database-support, out-of-memory features |
| dask | Very similar to pandas, but with some limitations. | Not always trivially fast on single machines. | Can handle multi-machine clusters, also lazy evaluation. |
| modin | Identical to pandas. | Not always fast, sometimes even slower than pandas if mishandled. | Can handle clusters through dask, ray or MPI. |
| datatable | Quite different from pandas, but not as much as polars. | Extremely fast, on par with polars. | Brings R's `data.table` to python. Lazy evaluation and out-of-memory dataframes. |

# Going further
You might wonder why `async` is not mentioned here. Asynchronous programming is a very related tool, but it also has nothing to do with parallelization by itself: it will not allow you to perform calculations at the same time using several CPU cores. What it allows you to do is when your program encounters a blocking operation (for example waiting for a network response), to let the interpreter run some other parts of your code in the meantime. This is very useful for I/O operations, but it won't speed up calculations since your code is still using only one CPU core at most. If you want to learn about it, you can check guides [here](https://realpython.com/async-io-python/) or [here](https://superfastpython.com/python-asyncio/).

If you want to go further, Jason Brownlee's [website](https://superfastpython.com/) has a lot of rich and in-depth tutorials. Please tell me if you know of other great resources!

Thanks for reading, I hope this will help you make lightning fast calculations!
