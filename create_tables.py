#!/usr/bin/python
 
import psycopg2
from config import config
 
def create_tables():
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
            date VARCHAR(255) NOT NULL
        )
        """)
    create_db = """
        CREATE TABLE gdax_bot_data (
            id SERIAL PRIMARY KEY,
            mean FLOAT NOT NULL,
            std FLOAT NOT NULL,
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
        for command in commands:
            cur.execute(command)
        # cur.execute(create_db)
        # close communication with the PostgreSQL database server
        cur.close()
        # commit the changes
        conn.commit()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
 
 
if __name__ == '__main__':
    create_tables()