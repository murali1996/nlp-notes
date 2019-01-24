#Look Up Items

## Anaconda Related and Proxy for python
C:\ProgramData\Anaconda3\Lib\site-packages\skfuzzy\cluster
set http_proxy=http://107.108.167.20:80
set https_proxy=https://107.108.167.20:80
conda config --set ssl_verify <pathToYourFile>.crt
conda config --set ssl_verify D:\Softwares\Basic\Cerificate\SRID_CRT_New.cer
conda config --set ssl_verify false
Defauly Python path: C:\ProgramData\Anaconda3\python

## Step-0: Create a URL for your repo
#### Step-1: Cloning a repo; I'm sill working on master branch in local repo and since I did clone, i need not git init again
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
#### Step-2: Creating a new branch and pushing files to remote
git checkout -b new_br_name_ origin/master
-----<modify files>---
git add .
git commit -m "done some changes in new branch"
git push origin new_br_name_
-----<alternative: git push {remote name} {branch name}>----
#### Step-3: After the owner merges your branch to the remote origin, do a pull (which essentially is (popularly suggested) fetch+merge operation)
git checkout master
git fetch origin
-----<alternative: fit fetch {remote}>-------
git merge origin/master
-----<alternative: fit fetch {remote}/{branch}>------
#### Also..
git reset --hard upstream/master
git reset --hard origin/master
git lfs clone https://github.com/mmihaltz/word2vec-GoogleNews-vectors.git
git clone https://github.com/mmihaltz/word2vec-GoogleNews-vectors.git
#### Also..
1. Linux
$git log | grep "commit\|Author\|Date" > filename.csv
2. Windows
git log | findstr "Author Date commit" > filename.csv

## Some frequently used np and tf methods
np.ravel() # to flatten
np.hstack(), vstack(), stack() # from list to array or join arrays
enumerate(), np.ndenumerate()
Display of objects not supported in spyder: convert arrays to lists .tolist() and print
print("Predicted model: {a:.3f}x + {b:.3f}".format(a=w_value[0], b=w_value[1]))
print("Predicted model: {.3f}x + {.3f}".format(w_value[0], w_value[1]))

## TensorFlow
config = tf.ConfigProto()
config.gpu_options.allow_growth = True #takes the min required memory or
config.gpu_options.per_process_gpu_memory_fraction = 0.4 #fixed memory - 40% of total memory
session = tf.Session(config=config, ...)
echo $CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=1,2
python3 -c "from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())"
torch.cuda.is_available()
tensorboard --logdir .

## TensorFlow HUB
You can download your model need from url + '?tf-hub-format=compressed'
i tried downloading elmo and it worked
url = https://tfhub.dev/google/elmo/2 + '?tf-hub-format=compressed'
eg: https://tfhub.dev/google/elmo/2?tf-hub-format=compressed
the model will be downloaded as a tarfile to your machine.
once you untar the file, it will have tfhub_module.pb


## Linux
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

## TMUX
tmux (https://hackernoon.com/a-gentle-introduction-to-tmux-8d784c404340)
tmux ls # lists active sessions
tmux attach-session -t session_name || tmux new-session -s session_name
---> source activate conda_env_murali
---> set_cuda9.0_cudnn7.0
---> locate cuda
---> set -a ; . /usr/local/nvidia_cuda/scripts/set_cuda9.0_cudnn7.0 ; set +a
---> python
--->---> import tensorflow as tf
--->---> sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

Now you can run the stuff and close terminal with tmux session actively doing its work!
To re-open...

---> tmux ls
---> tmux attach-session -t my_session

--->---> tmux kill-session #in session or ctrl+d
--->---> detach: ctrl+b d

## Chrome Issues
"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe" -no-sandbox

