#!/usr/bin/python
 
import psycopg2
from config import config

import argparse
parser = argparse.ArgumentParser(description="Indicate to delete before database creation")
parser.add_argument('-d', action='store_true')

def create_tables(delete):
    """ create tables in the PostgreSQL database"""
    commands = (
        """
        DROP TABLE gdax_bot_data
        """,
        """
        CREATE TABLE gdax_bot_data (
            id SERIAL PRIMARY KEY,
            mean FLOAT NOT NULL,
            std FLOAT NOT NULL,
            start FLOAT NOT NULL,
            stop FLOAT NOT NULL,
            high FLOAT NOT NULL,
            low FLOAT NOT NULL,
            date VARCHAR(255) NOT NULL
        )
        """)
    create_db = """
        CREATE TABLE gdax_bot_data (
            id SERIAL PRIMARY KEY,
            mean FLOAT NOT NULL,
            std FLOAT NOT NULL,
            start FLOAT NOT NULL,
            stop FLOAT NOT NULL,
            high FLOAT NOT NULL,
            low FLOAT NOT NULL,
            date VARCHAR(255) NOT NULL
        )
        """
    conn = None
    try:
        # read the connection parameters
        params = config()
        # connect to the PostgreSQL server
        conn = psycopg2.connect(**params)
        cur = conn.cursor()
        # create table one by one
        if delete:
            for command in commands:
                cur.execute(command)
        else:
            cur.execute(create_db)
        # close communication with the PostgreSQL database server
        cur.close()
        # commit the changes
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
        print("Use -d flag to replace existing database")
    finally:
        if conn is not None:
            conn.close()
 
 
if __name__ == '__main__':
    args = parser.parse_args()
    create_tables(delete=args.d)