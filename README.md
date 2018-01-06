## setup
  ### dependency gathering for local:
  ```pip install -r requirements.txt```
  
  ### ubuntu daemon for VM:

  #### get pm2:
  ```
  sudo apt-get install yarn
  yarn add pm2
  ```
  add symlink if node is stored as "nodejs"
  ```
  ln -s /usr/bin/nodejs /usr/bin/node
  ```

  #### use virtual env:
  1. make virtual env
  2. install local reqs
  3. copy to unnamed file:
  ```cp main.py main```
  4. run with local interpreter
  ```pm2 start ./main --interpreter ./venv/bin/python```

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
