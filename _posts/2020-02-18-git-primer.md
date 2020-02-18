---
layout: post
title: A quick git primer
excerpt: As I come from the computer science community, I find that git is a very useful tool for handling projects, although it can be intimidating at first. This is my attempt at get you going on git in a few minutes.
---

As I come from the computer science community, I find that git is a very useful tool for handling projects, although it can be intimidating at first. This is my attempt at get you going on git in a few minutes. Of course, if you want to really master it, nothing beats the [official tutorial](https://git-scm.com/book/en/v2), but it is maybe too much for a basic usage.

So let us make the presentations: `git` is a software that has been originally developed in 2005 by Linus Torvald also lead developer of the Linux kernel. It has two main functions:
* it acts as a version-control software, which means it maintains a history of your project, enabling you to go back in time if you have broken something for example. This capacity is a nice alternative to storing your own archive of files bearing names such as `model3_12022018.py`.
* it enables you to synchronize the state of your project among different collaborators, so that you can send the modifications that you made and receive those made by your collaborators. It is in this sense an alternative to mail-based synchronization.
  
For this last capacity, an online platform supporting git is needed, since the shared version of the code has to be stored on a server. The most well known one is github.com, but there are many others like gitlab.com. Git and github are often confused, but they are 2 distinct things: git is an open-source software, while github is a website offering services for users of the git software.

## Installing git
- For Mac users, git is already included in the Xcode tools. You can type `git --version` which will check if it already installed or offer to install it
- For Windows users, the best is to install [git for windows](https://gitforwindows.org/)
- For Linux users, use your [favorite package manager](https://git-scm.com/download/linux)

## Creating a repository and committing versions
Now we shall explain the concepts used by git with an example. Let us first create a *repository*, meaning a project in git's vocabulary. For this, simply create a directory where you add a few code files. In my example I have 3 files in my directory: `model.py`, `plot.py` and `data.npz`. Now open a terminal (or git bash for Windows users), move to your project's directory and type:
```sh
git init
```

Voilà, what was a mere directory became a repository. This means that git is ready to handle a history of versions of your directory. Concretely, this history consists in a succession of what git calls *commits*. You can think of those commits as checkpoints or snapshots of the state of your project that you will always be able to retrieve. Let us create a first commit that will contain the initial state of our project.
```sh
git add *.py
git commit -m "initial commit"
```
Ok, so there are 2 steps for this operation: the first command is `git add`, and this is where you tell git which files you want to include in this commit. Indeed, maybe there are some files that you want git to ignore, for example a data file that is very heavy (in this case you should NOT include it), or some png plots. So in this first step you should only include what you want to track along versions, for example code files, notebooks, or maybe simulation parameters or results. Here we use the terminal joker * to add all files ending with .py. We will see later easier ways to select which files are tracked or not.

The second commit effectively adds the commit to the git history. You have to add a small message after the `-m` option to remember what this commit was about (if you forget the `-m` option, git will force you to write some message by opening the vim text editor. [It can be hard to exit this editor](https://stackoverflow.com/questions/11828270/how-do-i-exit-the-vim-editor)).

That's it, now this initial version of your code is stored forever. Now let us say the next day we have an idea to make better plots. We modify the file `plot.py` and once we are satisfied we want to store them in a new commit. You can first look at what has been modified with:
```sh
git status
```
which should tell you that plot.py is tracked and has been modified, and that data.npz is not tracked (meaning git has never stored a version of it). I suggest to always execute `git status` before commiting, to remember what you have modified since last time. That's all good, we can add our new commit with:
```sh
git add plot.py
git commit -m "better plots"
```

Now there are 2 versions of your code. You can visualize the history of your project with:
```sh
git log --oneline --decorate
```
which outputs:
```sh
936d456 (HEAD -> master) better plots
40229b2 initial commit
```
Here you can see the history of your project: the commits are written from most recent to oldest, with a nice little code for each, the name you chose, and a weird indication for the last commit that essentially indicates that this is the current state of your project.

## Working with collaborators
### Setup
One of the main features of git are its synchronization capacities. Let us see how this works with github. Open a github account and create a new repository named test for example (a big green button normally). Now there are 2 cases:
* either you already have some repository set up locally (you created it with `git init`). You can link it to your online repository with 
   ```sh
   git remote add origin https://github.com/username/test.git
   ```
   (the command is listed on the page that opens when you create the new repository). It simply adds a *remote*, which how git calls an online version of the repository. So far, there is still nothing online. To *push* your commits online execute:
   ```sh
   git push -u origin master
   ```
   The first push has to be made like this to link a specific remote (called origin, we will use only one remote here) to a specific *branch* (the default branch is always called master. I might tell more about branches some other time, it is not essential). Once the repository is set up we will just use `git push`

* your collaborator wants to get his copy of the project, and doesn't have a repo setup yet. Then he can run
  ```sh
  git clone https://github.com/username/test.git
  ```
  which will create a new directory containing the same contents as on the online repository.

Note that to push you might have to enter your github identifier and password. If a push is rejected, it probably means you are not listed as a collaborator of the repository, which can be modified on github in the settings page of the repository.

### Online tools
Now the repository is online. You can see the code files, but most of all you can now see all the history of your modifications in a very intuitive way. If you click on the number of commits on the top left of the list of code files, you can see the history of commits. If you click on one of the commits you can visualize all the modifications that have been done in this commit. 

Back to the project home page, if you click on a file name, you will notice that you can click on a history function that will list the commits that have modified this file. This can be quite handy if after a few months you get lost in your project.

### Working together
Now you and your collaborator want to work while keeping a synchronized version of the project. It is all based on *pushing* your changes online, and *pulling* your collaborator's changes. 

If you have made some modifications, add a new commit and push it with:
```sh
git add *.py
git commit -m "new stuff"
git push
```

If your collaborator has pushed things online, retrieve them with:
```
git pull
```
This will synchronize all the files in your directory with their online version. It also adds the commits that have been pushed online to your project's history.

### Conflicts
Of course, no collaborative system can work without a few rules. Here are the main ones:
- you cannot push if you collaborator has already pushed changes that you have not pulled. Said otherwise, the commits that are added online have to follow each other. git cannot accept two concurrent commits. So be sure to always pull right before you push.
  
- when you pull, your local files are overwritten by the online version if they are in the same state as the last commit. For example, say there is a commit 1 and your collaborator pushed a commit 2 that modifies model.py. If you have not touched this file since commit 1, it will simply be replaced by its version in commit 2. This will work even if the meanwhile you have modified plot.py: this file is not in the commit, so your local changes to plot.py are kept. Hence you and your collaborator can gracefully work in parallel, as long as it is on different files.
  
- the tricky part is of course if your collaborator pushes a commit 2 modifying plot.py, and you have also modified plot.py on your computer. Then when you execute `git pull` you will have a message saying there is a conflict. This will eventually happen if you use it, and can be very discouraging situation for a beginner. Here are a few options to save the day:
    - discard your changes: 
        ```
        git checkout -- plot.py
        ```

    - the dirty but effective way: rename your version of `plot.py` by `plot2.py`. Redo `git pull` which will now work without conflict. Then you can manually add your changes to `plot.py` from your personal copy, and push them afterwards.
  
    - trust the magic of git: actually if the changes you and your collaborator don't overlap (they are on different parts of the file), git can merge them automatically. For this you can launch the procedure called a *merge*. It consists of putting your local changes in a commit and then merging your commit with the remote one
      ```sh
      git add plot.py
      git commit -m "my modifications to plot.py"
      git pull
      ```
      If the modifications are not overlapping, git will intelligently merge them with this sequence. However if they are, you will get a message saying that automatic merge failed and that you have to fix conflicts yourself. This means git will add lines in the conflicting files like
      ```
      <<<<<<< HEAD
      # some stuff...
      =======
      # some stuff...
      >>>>>>> a sequence of letters and digits
      ```
      The stuff above the `=====` are local changes and below are the remote changes. git kept both because it judged it couldn't choose between them. So you have to choose what to keep by yourself and remove the lines `<<<<<<< HEAD`, `=====` and `>>>>>>>`. When it is done, you will have to add a *merge commit*:
      ```
      git add plot.py
      git commit -m "merge completed"
      git push
      ```
      After all this has been done, you can visualize what happened with the graph version of `git`:
      ```
      git log --oneline --decorate --graph --all
      ```
      you will see that your local version and the remote one diverged, and then merged back again. 

## gitignore
One important trick is to manage which files should be tracked and which should be ignored by git. For this, git uses a hidden file in your project's directory called `.gitignore` (for those who don't know, hidden files are files whose name start by . and they are not shown by file explorers by default. You can see them with the command `ls -a ` in the terminal).

To use this functionality, open a text editor and create a file called `.gitignore` in your project's directory. You can add in it names of files, patterns, or directories, for example:
```
*.npz
plots/
```

Actually, to make your life easier, a collection of ready-to-use gitignores for different types of projects is available at [gitignore.io](https://www.gitignore.io/).

The advantage is that then to add a commit you can simply run
```sh
git add .
git commit -m "my message"
```
the dot signifying to git that it should add everything except what is in the gitignore.

## Some useful aliases
I want to finish with how to set up some useful aliases I use:
```sh
git config --global alias.st status # cause I use it so often
git config --global alias.logp "log --oneline --decorate"
git config --global alias.logg "log --oneline --decorate --graph --all"
```
This means you can use the commands `git st`, `git logp` and `git logg` (which corresponds to 2 ways to visualize the history of the project in the terminal).
