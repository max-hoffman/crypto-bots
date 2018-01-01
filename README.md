## setup
virtual env:
```
sudo conda install virtualenv
virtualenv --no-site-packages venv
source venv/bin/activate
pip freeze > requirements.txt
```

iterm plotting:
add to .bash_profile:
```
export MPLBACKEND="module://itermplot"
export ITERMPLOT=rv
```