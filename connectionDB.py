import psycopg2
from psycopg2 import OperationalError
# Define connection parameters
dbname = "telecomdb"
user = "postgres"
password = "admin"
host = "localhost"
port = "5432"

    # Establish a connection to the databas
conn = psycopg2.connect(
    dbname=dbname,
    user=user,
    password=password,
    host=host,
    port=port
    )
    
   