import psycopg2
import pandas as pd

# Define connection parameters
dbname = "clusteredData"
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
sql_query = "SELECT * FROM clustered_data"  # Replace 'clustered_data' with your actual table name

# Fetch data from the database
cursor = conn.cursor()
cursor.execute(sql_query)
data = cursor.fetchall()

# Convert data to DataFrame
columns = [description[0] for description in cursor.description]
clustered_data = pd.DataFrame(data, columns=columns)

# Export DataFrame to CSV
clustered_data.to_csv('clustered_data.csv', index=False)  # Replace 'clustered_data.csv' with your desired CSV filename

# Close database connection
cursor.close()
conn.close()
