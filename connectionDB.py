import psycopg2
import pandas as pd

# Define connection parameters
dbname = "userSatisfaction"
user = "postgres"
password = "admin"
host = "localhost"
port = "5432"

# Establish a connection to the PostgreSQL database
conn = psycopg2.connect(
    dbname=dbname,
    user=user,
    password=password,
    host=host,
    port=port
)

# SQL query to select the data you want to export
sql_query = "SELECT * FROM user_satisfaction_scores"  # Replace 'xdr_data' with your actual table name

# Fetch data from the database
cursor = conn.cursor()
cursor.execute(sql_query)
data = cursor.fetchall()

# Convert data to DataFrame
columns = [description[0] for description in cursor.description]
df = pd.DataFrame(data, columns=columns)

# Export DataFrame to CSV
df.to_csv('user_data.csv', index=False)
