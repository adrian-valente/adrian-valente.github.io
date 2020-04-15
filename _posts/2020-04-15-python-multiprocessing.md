---
layout: post
title: How to parallellize a python loop in one minute
excerpt: Having machines with many CPU cores is a great way to make all your computations run much faster, right? Unless.. You have to modify the code yourself to take advantage of it? But parallellization seems complicated, involves forks and processes...
---

Having machines with many CPU cores is a great way to make all your computations run much faster, right? Unless.. You have to modify the code yourself to take advantage of it? But parallellization seems complicated, involves weird concepts of forks and processes... This is too daunting, let's stick with my python code using only one core and never-ending computations...

This is what I thought for a long time, until I understood that parallellization in python can actually be made extremely easy with a simple pipeline that you can apply to any bit of code in less than one minute. That's right with one minute of moving lines of code around, you will be able to win huuuge amounts of computation time!

In order to do this you need a loop that repeats the same operation several times, in which the result of an iteration is not needed in the next one, so that you can run all iterations in parallel. Typically, this applies very well to computations on Monte-Carlo or bootstrap samples, or optimization processes such as simulated annealing.

So let us take for example the following code which samples 100 random matrices of size 1000x1000 and computes its eigenvalues, to check the [circular law](https://en.wikipedia.org/wiki/Circular_law) (not a mesmerizing example but minimal enough for our purposes)
```py
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

n = 1000
eigvals = []

for i in range(100):
    X = np.random.randn(n, n) / np.sqrt(n)
    eigvals.append(np.linalg.eigvals(X))

eigvals = np.array(eigvals)
real = np.real(eigvals).ravel()
imag = np.imag(eigvals).ravel()
sns.jointplot(real, imag, kind='hex')
plt.show()
```

Here, the loop is a clear target for parallellization, since it is just the same operation repeated 100 times. So to parallellize it, simply copy-paste the code inside the loop in a separate function called `task`. Add all the needed variables as arguments and the ones that will be reused as return values:

```py
def task(n):
    X = np.random.randn(n, n) / np.sqrt(n)
    return np.linalg.eigvals(X)
```

Now, only the 3 following lines are needed: 
```py
pool = mp.Pool(mp.cpu_count())
args = [1000] * 100
res = pool.map(task, args)
```
with the required import statements:
```py
import multiprocessing as mp
import itertools
```
The first one creates the `Pool` which is the object that will distribute the tasks accross cpu cores. You can give as argument the number of concurrent tasks that you want the computer to do. Usually it is fair to go for the number of cores given by `mp.cpu_count()`. In the second line, you create an iterable of arguments for each task. Here it is simple, it is the same argument for all tasks, repeated the number of times we want it to run. Finally the last line will magically execute the task in parallel with the argument list you provided and return all the return values bundled in a list.

Important note: if your task has not one but several arguments you will have to put the arguments in a iterable of tuples, and use the `starmap` function instead of `map`:
```py
args = [(1000, 1000)] * 100
res = pool.starmap(task, args)
```

Finally, to build the iterable less brutally the functions of `itertools` can be useful, for example:
```py
args = itertools.repeat(argument_tuple, 100)
```

Here is the final code for the example program. Enjoy your fast computations!
```py
import multiprocessing as mp
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def task(n):
   X = np.random.randn(n, n) / np.sqrt(n)
   return np.linalg.eigvals(X)

pool = mp.Pool(mp.cpu_count())
args = [1000] * 100
res = pool.map(task, args)
res = np.array(res)
real = np.real(res).ravel()
imag = np.imag(res).ravel()
sns.jointplot(real, imag, kind='hex')
plt.show()
```
