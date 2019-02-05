# Look Up for System
## Linux
```
nvcc --version
python --version
echo ".py"
htop
ps <pid>
ps -aux | grep python
rm -rf <folder name>
cp filename dirname

conda info --envs
source activate conda_env_murali
set -a ; . /usr/local/nvidia_cuda/scripts/set_cuda9.0_cudnn7.0 ; set +a
jupyter notebook --no-browser --ip '*'

pwd, cd, ls, mkdir, rmdir, rm, rm -r, mv, cp, apt-get, sudo #https://maker.pro/linux/tutorial/basic-linux-commands-for-beginners
touch # to create files # example: touch new.txt, touch main.py
man, --help # get maunal # example: man cd, cd -help
locate # use locate -i new.txt to search in a case insensitive way
cp -a path_src path_dest

echo # The "echo" command helps us move some data, usually text into a file
cat # Use the cat command to display the contents of a file. It is usually used to easily view programs.
chmod +x <filename> # change permissions to directly execute, ex: chmod +x main.py 

tar # https://www.tecmint.com/18-tar-command-examples-in-linux/
tar # examples: tar -cvf, tar -xvf, tar -tvf, etc.
unzip tecmint_files.zip -d /tmp/unziped, zip -r tecmint_files.zip tecmint_files # Use zip to compress files into a zip archive, and unzip to extract files from a zip archive.

hostname -I # to get IP
ping
```
## TMUX
```
tmux (https://hackernoon.com/a-gentle-introduction-to-tmux-8d784c404340)
tmux ls # lists active sessions
tmux attach-session -t session_name || tmux new-session -s session_name
---> source activate conda_env_murali
---> set_cuda9.0_cudnn7.0
---> locate cuda
---> set -a ; . /usr/local/nvidia_cuda/scripts/set_cuda9.0_cudnn7.0 ; set +a
--------<Or, set -a ; . /usr/local/nvidia_cuda/scripts/set_cuda8.0_cudnn6.0 ; set +a>-----------
---> python
--->---> import tensorflow as tf
--->---> sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

Now you can run the stuff and close terminal with tmux session actively doing its work!
To re-open...

---> tmux ls
---> tmux attach-session -t my_session

--->---> tmux kill-session #in session or ctrl+d
--->---> detach: ctrl+b d
```


# Look Up for Tools
## Conda & its SSL
```
C:\ProgramData\Anaconda3\Lib\site-packages\skfuzzy\cluster
set http_proxy=http://IP:PORT
set https_proxy=http://IP:PORT
conda config --set ssl_verify <pathToYourFile>.crt
conda config --set ssl_verify false
Defauly Python path: C:\ProgramData\Anaconda3\python
```
## Git Help
#### Step-0: Create a URL for your repo
#### Step-1: Cloning a repo; I'm sill working on master branch in local repo and since I did clone, i need not git init again
```
git config --global user.name "murali1996"
git config --global user.mail "jsaimurali001@gmail.com"
git config --global http.sslverify false
git config --global http.proxy http://IP:PORT
git config --global https.proxy https://IP:PORT
git config --global --unset http.proxy
git config --global --get http.proxy
git clone https://github.com/thunlp/TensorFlow-TransX
cd TensorFlow-TransX
git remote add upstream https://github.com/thunlp/TensorFlow-TransX
git remote rm origin
git remote add origin https://github.sec.samsung.net/s-jayanthi/tensorflow_transX
git remote -v
git add -A
git commit -m "first commit"
git push origin master
git status
```
#### Step-2: Creating a new_branch (say for example to fix a bug or to add a new feature) and pushing files to remote
```
git checkout -b new_br_name_ origin/master
-----<modify files>---
git add .
git commit -m "done some changes in new branch"
git push origin new_br_name_
-----<alternative: git push {remote name} {branch name}>----
```
#### Step-3(1): You created a new_branch and modified some files and committed your code. Also, you have not pushed your new_branch code. But by then, your local master might or might not have been proceeded with few more commits since the new_branch above got split. If it didn't, then a simple Fast-Forward merge is sufficient without a commit. Else, a 3-Way merge is required. Code for doing this is as follows: <https://www.atlassian.com/git/tutorials/using-branches/git-merge>
```
# Start a new feature
git checkout -b new-feature master
# Edit some files
git add <file>
git commit -m "Start a feature"
# Edit some files
git add <file>
git commit -m "Finish a feature"
# Merge in the new-feature branch
git checkout master
git merge new-feature
-----<This above command can also be used for 3-way merge but that would definitely result in merging with commit>------
git branch -d new-feature
-----<alternative: git merge --no-ff <branch> if you want to also commit after fast forward merge>------
git push origin master
-----<ensures all the new features are available at origin/master now>------
```
#### Step-3(2): For suppose, you have pushed your code to origin/master after adding the new feature above. Thus, your local master is now required to merge with the new head of origin/master. As Step-3(1), your local master might or might not have been proceeded with few more commits since the new_branch above got split. Code for doing this is as follows: <https://www.atlassian.com/git/tutorials/syncing/git-pull>
```
git checkout master
git fetch origin
-----<alternative: fit fetch {remote} {branch required at remote}>-------
-----<check if you are on right local branch by doing: git status>-------
git merge origin/master
-----<alternative: fit fetch {the remote branch that you want to merge with current local branch}>------
```
#### Pull Request
```
Open Source Contribution:
https://www.atlassian.com/git/tutorials/making-a-pull-request
```
#### Git Ignore
```
# https://github.com/github/gitignore/blob/master/Python.gitignore
# https://www.atlassian.com/git/tutorials/saving-changes/gitignore
cd project_repo
touch .gitignore
# add required types
```
#### Stash, Reset hard
```
In case when you have a scenario of 3-way merge but you donot want to commit (or ignore) you local master changes. Then either use stash or reset.
git stash
-----<you can later come to this stashed work using git stash pop, git stash list, etc.>------
-----<https://www.atlassian.com/git/tutorials/saving-changes/git-stash>------
git reset --hard upstream/master
git reset --hard origin/master
```
#### Large Files
```
git lfs clone https://github.com/mmihaltz/word2vec-GoogleNews-vectors.git
git clone https://github.com/mmihaltz/word2vec-GoogleNews-vectors.git
```
#### Collect Logs in CSV
```
Get logs into a csv:
1. Linux
$ git log | grep "commit\|Author\|Date" > filename.csv
2. Windows
> git log | findstr "Author Date commit" > filename.csv
```
## PIP
#### pip freeze
```
# This command is incredibly useful if you need to recreate your project from scratch and need to know the exact libraries/versions you need to install.
cd project_folder
pip freeze > requirements.txt
```
## Chrome Issues
"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" -no-sandbox


# Look Up for Libraries
## Numpy
```
np.ravel() # to flatten
np.hstack(), vstack(), stack() # from list to array or join arrays
enumerate(), np.ndenumerate()
Display of objects not supported in spyder: convert arrays to lists .tolist() and print
print("Predicted model: {a:.3f}x + {b:.3f}".format(a=w_value[0], b=w_value[1]))
print("Predicted model: {.3f}x + {.3f}".format(w_value[0], w_value[1]))
```
## Tensorflow
#### General
```
config = tf.ConfigProto()
config.gpu_options.allow_growth = True #takes the min required memory or
config.gpu_options.per_process_gpu_memory_fraction = 0.4 #fixed memory - 40% of total memory
session = tf.Session(config=config, ...)

echo $CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=1,2
python3 -c "from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())"
```
#### TensorBoard
```
tensorboard --logdir .
```
#### TensorflowHub
```
You can download your model need from url + '?tf-hub-format=compressed'
i tried downloading elmo and it worked
url = https://tfhub.dev/google/elmo/2 + '?tf-hub-format=compressed'
eg: https://tfhub.dev/google/elmo/2?tf-hub-format=compressed
the model will be downloaded as a tarfile to your machine.
once you untar the file, it will have tfhub_module.pb
```





