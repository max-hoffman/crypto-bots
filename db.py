#!/usr/bin/python

import psycopg2
from config import config
import sys

class DB():

    def __init__(self):
        self.conn = None
        params = config()
        self.connect(params)

    def connect(self, params):
        try:
            print('Connecting to the PostgreSQL database...')
            self.conn = psycopg2.connect(**params)
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
            sys.exit()
        finally:
            if self.conn is not None:
                return
            else:
                sys.exit()

    def insert_point(self, mean, std, date):
        """ insert a new data point into the net currency table """
        sql = """INSERT INTO gdax_bot_data(mean, std, date)
                VALUES(%s, %s, %s);"""
        try:
            # create a new cursor
            self.cursor = self.conn.cursor()
            # execute the INSERT statement
            self.cursor.execute(sql, (mean, std, date))
            # commit the changes to the database
            self.conn.commit()
            # close communication with the database
            self.cursor.close()
        except (Exception, psycopg2.DatabaseError) as error:
            print(error)
        return
    
    def close(self, conn):
        conn.close()
