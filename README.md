## setup
  ### dependency management:
  1. virtual env
  ```
  sudo conda install virtualenv
  virtualenv --no-site-packages venv
  source venv/bin/activate
  pip freeze > requirements.txt
  ```
  2. pipreqs
  ```
  [sudo] pip install pipreqs
  pipreqs . --force
  ```

  ### iterm plotting:
  add to .bash_profile:
  ```
  export MPLBACKEND="module://itermplot"
  export ITERMPLOT=rv
  ```

  ### database:
  1. duplicate, fill in and rename database.ex.ini > database.ini (I use and elephantSQL instance)
  2. create tables (use -d flag if dropping existing database first)
  ```
  python create_tables.py
  python create_tables.py -d
  ```

  ### tensorboard
  ```tensorboard --logdir=logs```
