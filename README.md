## setup
  ### dependency gathering:
  ```pip install -r requirements.txt```
  
  ### ubuntu daemon:
  * http://reustle.org/managing-long-running-processes-with-supervisor.html
  1. install supervisord
  ```sudo apt-get install supervisor -y && sudo supervisord```
  3. create and edit config file:
  ```sudo nano gdax_bot.conf```
  ```
  [program:gdax_bot]
  command=python -u main.py
  directory=/home/ubuntu/src/crypto-bots/
  stdout_logfile=/home/ubuntu/gdax_bot_output.txt
  redirect_stderr=true
  ```
  4. made config available
  ```sudo supervisorctl reread```
  5. run program
  ```sudo supervisorctl add gdax_bot```
  6. check logs

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
