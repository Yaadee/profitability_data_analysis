# import pandas as pd

# from connectionDB import conn;
# cursor = conn.cursor()

#     # Fetch all results into a DataFrame
# columns = [desc[0] for desc in cursor.description]
# user_data = pd.DataFrame(cursor.fetchall(), columns=columns)

import pandas as pd
from connectionDB import conn
cursor = conn.cursor()
# Execute a query using the cursor
cursor.execute("SELECT * FROM xdr_data")

# Check if the cursor description is not None before fetching results
if cursor.description is not None:
    columns = [desc[0] for desc in cursor.description]
    user_data = pd.DataFrame(cursor.fetchall(), columns=columns)
else:
    # Handle the case when cursor description is None (e.g., empty result set)
    user_data = pd.DataFrame()  # Create an empty DataFrame or handle it based on your needs

# Now you can work with the user_data DataFrame
