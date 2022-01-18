import psycopg2

HOST = 'localhost'
DB_NAME = 'dapeng'
TABLE_NAME = 'object_counter'
USER = 'postgres'
PWD = 'postgres'
SSL_MODE = 'require'

db_conn = None


def init():
    '''
    initialize database connection and make sure the object counter table exists
    '''
    # create connection
    db_conn = psycopg2.connect(database=DB_NAME, user=USER,
                               password=PWD, host=HOST)
    # set auto commit is true
    db_conn.autocommit = True
    # get instance of cursor
    cursor = db_conn.cursor()
    # check version
    cursor.execute('SELECT version()')
    result = cursor.fetchall()
    print("Opened database successfully! Version:{}".format(result[0]))
    # install uuid-ossp extension
    cursor.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
    # check table existed
    checkTableExisted = "SELECT EXISTS ( \
            SELECT FROM pg_tables \
            WHERE  schemaname = 'public' \
            AND    tablename  = '{}');".format(TABLE_NAME)
    cursor.execute(checkTableExisted)
    result = cursor.fetchall()
    # if table doesn't exist create that
    if False in result[0]:
        createTable = "CREATE TABLE IF NOT EXISTS {} ( \
            id uuid DEFAULT uuid_generate_v4 (), \
            camId VARCHAR(20) NOT NULL, \
            time timestamp NOT NULL, \
            type VARCHAR(20) NOT NULL, \
            inCounter int NOT NULL, \
            outCounter int NOT NULL, \
            PRIMARY KEY (id) \
            )".format(TABLE_NAME)
        cursor.execute(createTable)
        db_conn.commit()
        print("{} created successfully".format(TABLE_NAME))
    else:
        print("{} is existed!".format(TABLE_NAME))


init()
