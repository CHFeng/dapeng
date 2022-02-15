import psycopg2
from typing import List
from datetime import datetime as dt
from pydantic import BaseModel

HOST = "localhost"
DB_NAME = "dapeng"
RECORD_TABLE_NAME = "record"
CONFIG_TABLE_NAME = "config"
USER = "postgres"
PWD = "postgres"
SSL_MODE = "require"


class Record(BaseModel):
    camId: str
    time: dt
    type: str
    inValue: int
    outValue: int
    inAvgSpeed: int
    outAvgSpeed: int


class Database:
    def __init__(self) -> None:
        '''
        initialize database connection and make sure the object counter table exists
        '''
        # create connection
        self.conn = psycopg2.connect(database=DB_NAME, user=USER, password=PWD, host=HOST)
        # set auto commit is true
        self.conn.autocommit = True
        # get instance of cursor
        self.cursor = self.conn.cursor()
        # check version
        self.cursor.execute('SELECT version()')
        result = self.cursor.fetchall()
        print("Opened database successfully! Version:{}".format(result[0]))
        # install uuid-ossp extension
        self.cursor.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
        # check object detection records table exist or not
        checkTableExisted = """SELECT EXISTS (
                SELECT FROM pg_tables
                WHERE  schemaname = 'public'
                AND    tablename  = '{}');""".format(RECORD_TABLE_NAME)
        self.cursor.execute(checkTableExisted)
        result = self.cursor.fetchall()
        # if table doesn't exist create that
        if False in result[0]:
            createTable = """CREATE TABLE IF NOT EXISTS {} (
                id uuid DEFAULT uuid_generate_v4 (),
                camId VARCHAR(100) NOT NULL,
                time timestamp NOT NULL,
                type VARCHAR(20) NOT NULL,
                inCounter int NOT NULL,
                outCounter int NOT NULL,
                inAvgSpeed int NOT NULL,
                outAvgSpeed int NOT NULL,
                PRIMARY KEY (id)
                );""".format(RECORD_TABLE_NAME)
            self.cursor.execute(createTable)
            self.conn.commit()
            print("{} created successfully".format(RECORD_TABLE_NAME))
        else:
            print("Table '{}' is existed!".format(RECORD_TABLE_NAME))
        # check account table existed or not
        checkTableExisted = """SELECT EXISTS (
                SELECT FROM pg_tables
                WHERE  schemaname = 'public'
                AND    tablename  = '{}');""".format(CONFIG_TABLE_NAME)
        self.cursor.execute(checkTableExisted)
        result = self.cursor.fetchall()
        # if table doesn't exist create that
        if False in result[0]:
            createTable = """CREATE TABLE IF NOT EXISTS {} (
                id int NOT NULL,
                account VARCHAR(20) NOT NULL,
                password VARCHAR(20) NOT NULL,
                host VARCHAR(50),
                port int,
                PRIMARY KEY (id)
                );""".format(CONFIG_TABLE_NAME)
            self.cursor.execute(createTable)
            self.conn.commit()
            print("{} created successfully".format(CONFIG_TABLE_NAME))
            insertData = """INSERT INTO {} (id,account,password,host,port)
                VALUES (0, '{}', '{}', '{}', {});""".format(CONFIG_TABLE_NAME, "root", "root", "localhost", 80)

            self.cursor.execute(insertData)
            self.conn.commit()
            print("insert default config successfully")
        else:
            print("Table '{}' is existed!".format(CONFIG_TABLE_NAME))

    def add_record(self, record: Record) -> None:
        insertData = """INSERT INTO {} (camId,time,type,inCounter,outCounter,inAvgSpeed,outAvgSpeed)
        VALUES ('{}', '{}', '{}', {}, {}, {}, {});""".format(RECORD_TABLE_NAME, record.camId, record.time, record.type, record.inValue,
                                                             record.outValue, record.inAvgSpeed, record.outAvgSpeed)

        self.cursor.execute(insertData)
        self.conn.commit()

        print("insert record successfully")

    def get_all_records(self) -> List[Record]:
        data = []
        getAllData = "SELECT * FROM {};".format(RECORD_TABLE_NAME)

        self.cursor.execute(getAllData)
        rows = self.cursor.fetchall()
        for row in rows:
            record = {
                'id': row[0],
                'camId': row[1],
                'time': row[2],
                'type': row[3],
                'inCounter': row[4],
                'outCounter': row[5],
                'inAvgSpeed': row[6],
                'outAvgSpeed': row[7],
            }
            data.append(record)

        return data

    def get_record(self, id=None, camId=None, start_time=None, end_time=None, type=None) -> List[Record]:
        data = []
        query = ''
        if id:
            query += "id='{}' ".format(id)
        if camId:
            if query:
                query += 'AND '
            query += "camId='{}' ".format(camId)
        if start_time and end_time:
            if query:
                query += 'AND '
            query += "time >= '{}' AND time < '{}' ".format(start_time, end_time)
        if type:
            if query:
                query += 'AND '
            query += "type='{}' ".format(type)

        query = "SELECT * FROM {} WHERE ".format(RECORD_TABLE_NAME) + query
        print("Get record Query:", query)
        self.cursor.execute(query)
        rows = self.cursor.fetchall()
        for row in rows:
            record = {
                'id': row[0],
                'camId': row[1],
                'time': row[2],
                'type': row[3],
                'inCounter': row[4],
                'outCounter': row[5],
                'inAvgSpeed': row[6],
                'outAvgSpeed': row[7],
            }
            data.append(record)

        return data

    def delete_record(self, id: str) -> None:
        deleteData = "DELETE FROM {} WHERE id='{}';".format(RECORD_TABLE_NAME, id)

        self.cursor.execute(deleteData)
        self.conn.commit()

        print("delete record successfully")

    def get_config(self) -> dict:
        '''
        return dict {'account', 'password', 'host', 'port'}
        '''
        data = {}
        getAllData = "SELECT * FROM {};".format(CONFIG_TABLE_NAME)

        self.cursor.execute(getAllData)
        rows = self.cursor.fetchall()
        data['account'] = rows[0][1]
        data['password'] = rows[0][2]
        data['host'] = rows[0][3]
        data['port'] = rows[0][4]

        return data

    def update_config(self, account: str, password: str, host: str, port: int) -> dict:
        updateData = ''
        if account:
            updateData += "account = '{}'".format(account)
        if password:
            if updateData:
                updateData += ','
            updateData += "password = '{}'".format(password)
        if host:
            if updateData:
                updateData += ','
            updateData += "host = '{}'".format(host)
        if port:
            if updateData:
                updateData += ','
            updateData += "port = {}".format(port)
        if updateData != '':
            updateData = "UPDATE {} SET ".format(CONFIG_TABLE_NAME) + updateData + "WHERE id = 0;"

            self.cursor.execute(updateData)
            self.conn.commit()

            print("update config successfully")
        data = self.get_config()

        return data
