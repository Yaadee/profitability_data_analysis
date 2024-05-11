#!/usr/bin/env python
# coding: utf-8

# In[1]:


import psycopg2
import pandas as pd


# In[2]:


import numpy as np


# In[8]:


from adjustText import adjust_text


# In[3]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer


# In[4]:


import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor


# In[5]:


from sklearn.decomposition import PCA


# In[6]:


import seaborn as sns


# In[7]:


import psycopg2
from psycopg2 import OperationalError
# Define connection parameters
dbname = "telecomdb"
user = "postgres"
password = "admin"
host = "localhost"
port = "5432"
try:
    # Establish a connection to the database
    conn = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port
    )
    
    # Create a cursor object using the connection
    cursor = conn.cursor()
    
    # Query to fetch columns of the table "xdr_data"
    query = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = %s AND table_name = %s
    """
    
    # Execute the query with parameters
    cursor.execute(query, ('public', 'xdr_data'))

    # Fetch all results
    columns = cursor.fetchall()
    
    # Print the column names
    print("Columns of the table 'xdr_data':")
    for column in columns:
        print(column[0])

except OperationalError as e:
    print("Error: Could not connect to the database.", e)

finally:
    # Close the cursor and connection
    if 'cursor' in locals() and cursor is not None:
        cursor.close()
    if 'conn' in locals() and conn is not None:
        conn.close()


# In[8]:


# Define connection parameters
dbname = "telecomdb"
user = "postgres"
password = "admin"
host = "localhost"
port = "5432"

try:
    # Establish a connection to the database
    conn = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port
    )
    print("Connected to the database.")
    
    # Create a cursor object using the connection
    cursor = conn.cursor()

    # SQL query to aggregate per user information
    query = """
    SELECT "MSISDN/Number",
           COUNT(*) AS num_sessions,
           SUM("Dur. (ms)") AS total_session_duration,
           SUM("Total UL (Bytes)") AS total_ul_data,
           SUM("Total DL (Bytes)") AS total_dl_data,
           SUM("Social Media DL (Bytes)" + "Social Media UL (Bytes)") AS social_media_data,
           SUM("Google DL (Bytes)" + "Google UL (Bytes)") AS google_data,
           SUM("Email DL (Bytes)" + "Email UL (Bytes)") AS email_data,
           SUM("Youtube DL (Bytes)" + "Youtube UL (Bytes)") AS youtube_data,
           SUM("Netflix DL (Bytes)" + "Netflix UL (Bytes)") AS netflix_data,
           SUM("Gaming DL (Bytes)" + "Gaming UL (Bytes)") AS gaming_data,
           SUM("Other DL (Bytes)" + "Other UL (Bytes)") AS other_data
    FROM xdr_data
    GROUP BY "MSISDN/Number"
    """

    # Execute the query
    cursor.execute(query)

    # Fetch all results into a DataFrame
    columns = [desc[0] for desc in cursor.description]
    user_data = pd.DataFrame(cursor.fetchall(), columns=columns)

    # Close the cursor
    cursor.close()
    
    # Print first few rows of the DataFrame
    print(user_data.head())

    # Close the connection
    conn.close()
    print("Connection to the database closed.")

except psycopg2.Error as e:
    print("Error: Could not connect to the database.", e)


# In[12]:


user_data.head()


# In[13]:


# Describe all relevant variables and associated data types
print(user_data.info())


# In[14]:


# Analyze basic metrics
print("Basic Metrics:")
print(user_data.describe())


# In[31]:


# Compute dispersion parameters for each quantitative variable
dispersion_parameters = user_data.var()
print("Dispersion Parameters:")
print(dispersion_parameters)


# In[32]:


user_data.columns


# In[17]:


# pip uninstall matplotlib


# In[18]:


#pip install matplotlib


# In[42]:


class DataVisualizer:
    def __init__(self, user_data, applications):
        self.user_data = user_data
        self.applications = applications
        self.download_data = None
        self.upload_data = None

    def plot_data_usage(self):
        # Extracting the applications and corresponding data
        download_data = self.user_data[['total_dl_data'] + self.applications]
        upload_data = self.user_data[['total_ul_data'] + self.applications]

        # Storing data for access outside the method
        self.download_data = download_data
        self.upload_data = upload_data

        # Plotting
        num_apps = len(self.applications)
        bar_width = 0.35
        index = np.arange(num_apps)

        # Grouped bar plot for download data
        plt.bar(index, download_data.mean()[1:], bar_width, label='Download')
        # Grouped bar plot for upload data (shifted by bar_width for separation)
        plt.bar(index + bar_width, upload_data.mean()[1:], bar_width, label='Upload')

        plt.xlabel('Applications')
        plt.ylabel('Average Data Usage')
        plt.title('Average Data Usage by Application for Download and Upload')
        plt.xticks(index + bar_width / 2, self.applications, rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        plt.show()

# Example usage:
applications = ['social_media_data', 'google_data', 'email_data',
                'youtube_data', 'netflix_data', 'gaming_data', 'other_data']

visualizer = DataVisualizer(user_data, applications)
visualizer.plot_data_usage()
print(visualizer.download_data)
print(visualizer.upload_data)


# In[20]:


# Calculate total data usage for each application
total_data_usage = user_data[applications].sum()

# Calculate total data usage as percentages
total_data_usage_percent = (total_data_usage / total_data_usage.sum()) * 100

# Plotting pie chart
plt.figure(figsize=(8, 8))
plt.pie(total_data_usage_percent, labels=applications, autopct='%1.1f%%', startangle=140)
plt.title('Proportion of Data Usage by Application')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()
print("Total data usage for each application in % " ,total_data_usage_percent)


# In[21]:


# Compute the correlation matrix
correlation_matrix = user_data[applications + ['total_ul_data', 'total_dl_data']].corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap: Application Data vs Total DL+UL Data')
plt.show()

correlation_matrix


# In[22]:


# Segment the users into decile classes based on total session duration
deciles = pd.qcut(user_data['total_session_duration'], q=5, labels=False)
user_data['decile'] = deciles
# Compute total data (DL+UL) per decile class
total_data_per_decile = user_data.groupby('decile')[['total_ul_data', 'total_dl_data']].sum()
print("Total Data (DL+UL) per Decile Class:")
print(total_data_per_decile)
# Plot Total Data (DL+UL) per Decile Class
total_data_per_decile.plot(kind='bar', figsize=(10, 6))
plt.title('Total Data (DL+UL) per Decile Class')
plt.xlabel('Decile')
plt.ylabel('Total Data (Bytes)')
plt.xticks(rotation=0)
plt.legend(['Total UL Data', 'Total DL Data'])
plt.show()


# In[23]:


num_decile_classes = user_data['decile'].nunique()
print("Number of Decile Classes:", num_decile_classes)


# In[24]:


# # PCA

# from sklearn.manifold import TSNE
# from adjustText import adjust_text
# # Reduce dimensionality with t-SNE
# tsne = TSNE(n_components=2, random_state=42)
# tsne_components = tsne.fit_transform(user_data[['social_media_data', 'google_data', 'email_data', 
#                                                  'youtube_data', 'netflix_data', 'gaming_data', 'other_data']])

# # Visualize results
# tsne_df = pd.DataFrame(data=tsne_components, columns=['TSNE1', 'TSNE2'])
# plt.figure(figsize=(10, 8))
# plt.scatter(tsne_df['TSNE1'], tsne_df['TSNE2'], color='blue')

# # Add labels to data points
# texts = []
# for i, txt in enumerate(user_data.index):
#     texts.append(plt.text(tsne_df['TSNE1'][i], tsne_df['TSNE2'][i], txt))

# # Adjust the positions of text labels to avoid overlap
# adjust_text(texts, arrowprops=dict(arrowstyle='->', color='red'))

# plt.xlabel('t-SNE Component 1')
# plt.ylabel('t-SNE Component 2')
# plt.title('t-SNE Visualization')
# plt.grid(True)
# plt.show()


# In[25]:


# Separate features and target variable
X = user_data.drop(columns=['MSISDN/Number', 'decile'])  # Features
y = user_data['decile']  # Target variable

# Fit Random Forest model to get feature importances
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# Get feature importances
feature_importances = rf.feature_importances_
feature_names = X.columns

# Step 4: Feature Importance
print("Feature Importances:")
for feature_name, importance in zip(feature_names, feature_importances):
    print(f"{feature_name}: {importance}")

# Step 5: Visualization
# Visualize feature importances
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_names)
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()


# In[26]:


# Analysis:
# - The Random Forest model indicates that 'total_session_duration' is the most important feature for predicting the 'decile' value, while all other features have negligible importance.
# - There appears to be a strong relationship between total session duration and the target variable, with longer sessions likely corresponding to higher decile values.

# Recommendations:
# - Focus on strategies that increase total session duration, such as improving user engagement, enhancing content relevance, or optimizing user experience.
# - Explore ways to personalize content or services based on user session behavior to encourage longer and more frequent sessions.
# - Consider implementing features or incentives that incentivize users to extend their session duration or increase their engagement with the platform.

# Step 7: Risk Assessment

# Considerations:
# - Validate Results: Validate the findings of the Random Forest model by conducting additional analyses or experiments to confirm the importance of 'total_session_duration' and understand its impact on user behavior.
# - External Factors: Consider external factors that may influence session duration, such as seasonality, marketing campaigns, or changes in user preferences or demographics.
# - Data Quality: Ensure the accuracy and reliability of data related to session duration to avoid biases or inaccuracies in the analysis.

# Step 8: Continuous Monitoring

# Considerations:
# - Monitor Changes: Continuously monitor changes in user behavior, session patterns, and platform usage to identify any shifts in trends or patterns.
# - Experimentation: Conduct experiments or A/B tests to evaluate the effectiveness of strategies aimed at increasing session duration and measure their impact on user engagement and satisfaction.
# - Adaptation: Adapt strategies and interventions based on ongoing monitoring and feedback to optimize session duration and enhance user experience over time.

# Emphasize the importance of focusing on 'total_session_duration' as a key driver of user engagement and business success, while remaining vigilant about external factors and data quality considerations.


# # Task 2

# # Step 1: Aggregating Engagement Metrics Per Customer aggregate session frequency, session duration, and total session traffic per customer.
# # Report the top 10 customers per engagement metric.

# In[ ]:


class DataAggregator:
    def __init__(self, user_data):
        self.user_data = user_data

    def aggregate_engagement_metrics(self):
        # Aggregate session frequency, duration, and total traffic per customer
        engagement_metrics = self.user_data.groupby('MSISDN/Number').agg({
            'num_sessions': 'count',
            'total_session_duration': 'sum',
            'total_ul_data': 'sum',
            'total_dl_data': 'sum'
        })

        # Plot histograms for session frequency, duration, and total traffic
        self.plot_histograms(engagement_metrics)

        return engagement_metrics

    def plot_histograms(self, engagement_metrics):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.hist(engagement_metrics['num_sessions'], bins=20, color='skyblue')
        plt.title('Session Frequency')
        plt.xlabel('Number of Sessions')
        plt.ylabel('Frequency')

        plt.subplot(1, 3, 2)
        plt.hist(engagement_metrics['total_session_duration'], bins=20, color='salmon')
        plt.title('Session Duration')
        plt.xlabel('Session Duration (ms)')
        plt.ylabel('Frequency')

        plt.subplot(1, 3, 3)
        plt.hist(engagement_metrics['total_ul_data'] + engagement_metrics['total_dl_data'], bins=20, color='lightgreen')
        plt.title('Total Session Traffic')
        plt.xlabel('Total Traffic (Bytes)')
        plt.ylabel('Frequency')

        plt.tight_layout()
        plt.show()

data_aggregator = DataAggregator(user_data)

# Call the aggregate_engagement_metrics method
engagement_metrics = data_aggregator.aggregate_engagement_metrics()
engagement_metrics


# # Step 2: Normalize Engagement Metrics Normalize each engagement metric to prepare for clustering.Run a k-means clustering algorithm with appriopraite k- means value from elbow analysis to classify customers based on engagement.

# In[28]:


# Check for NaN values
nan_values = user_data.isna().sum()

# Print NaN values for each column
print("NaN values in each column:")
print(nan_values)


# # Normalization for clustering

# In[29]:


def elbow_method(data, max_k):
    # Impute missing values
    imputer = SimpleImputer(strategy='mean')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    
    # Normalize the data
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    
    # Calculate WCSS for different values of k
    wcss = []
    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(normalized_data)
        wcss.append(kmeans.inertia_)  # Inertia is the within-cluster sum of squares
        
    # Plot the elbow curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_k + 1), wcss, marker='o', linestyle='-')
    plt.title('Elbow Method')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.grid(True)
    plt.show()

# Example usage:
# Define your data and maximum value of K
max_k = 10

# Call the elbow_method function with your data and maximum K
elbow_method(user_data, max_k)


# In[30]:


def clustering(engagement_metrics):
    # Normalize engagement metrics
    scaler = MinMaxScaler()
    normalized_engagement_metrics = scaler.fit_transform(engagement_metrics)

    # Perform k-means clustering with k=5
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(normalized_engagement_metrics)
    cluster_labels = kmeans.labels_
    engagement_metrics['Cluster'] = cluster_labels

    # Plot clustered metrics
    plot_clustered_metrics(engagement_metrics)

    return engagement_metrics

def plot_clustered_metrics(engagement_metrics):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.scatter(engagement_metrics['num_sessions'], engagement_metrics['total_session_duration'], c=engagement_metrics['Cluster'], cmap='viridis')
    plt.title('Sessions vs. Duration')
    plt.xlabel('Number of Sessions')
    plt.ylabel('Total Session Duration (ms)')

    plt.subplot(1, 3, 2)
    plt.scatter(engagement_metrics['num_sessions'], engagement_metrics['total_ul_data'] + engagement_metrics['total_dl_data'], c=engagement_metrics['Cluster'], cmap='viridis')
    plt.title('Sessions vs. Total Traffic')
    plt.xlabel('Number of Sessions')
    plt.ylabel('Total Traffic (Bytes)')

    plt.subplot(1, 3, 3)
    plt.scatter(engagement_metrics['total_session_duration'], engagement_metrics['total_ul_data'] + engagement_metrics['total_dl_data'], c=engagement_metrics['Cluster'], cmap='viridis')
    plt.title('Duration vs. Total Traffic')
    plt.xlabel('Total Session Duration (ms)')
    plt.ylabel('Total Traffic (Bytes)')

    plt.tight_layout()
    plt.show()

# Example usage:
# Call the function to cluster and plot the metrics
clustered_metrics = clustering(engagement_metrics)
print(clustered_metrics)


# In[31]:


def compute_cluster_statistics(engagement_metrics):
    """
    Computes statistics for each cluster based on engagement metrics.

    Parameters:
        engagement_metrics (DataFrame): DataFrame containing engagement metrics.

    Returns:
        DataFrame: Statistics for each cluster.
    """
    cluster_statistics = engagement_metrics.groupby('Cluster').agg({
        'num_sessions': ['min', 'max', 'mean', 'sum'],
        'total_session_duration': ['min', 'max', 'mean', 'sum'],
        'total_ul_data': ['min', 'max', 'mean', 'sum'],
        'total_dl_data': ['min', 'max', 'mean', 'sum']
    })
    return cluster_statistics


def plot_cluster_statistics(cluster_statistics):
    """
    Plots statistics for each cluster.

    Parameters:
        cluster_statistics (DataFrame): DataFrame containing statistics for each cluster.
    """
    metrics = cluster_statistics.columns.levels[0]
    num_plots = len(metrics)

    fig, axes = plt.subplots(num_plots, 1, figsize=(10, 6*num_plots))

    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.bar(cluster_statistics.index, cluster_statistics[(metric, 'mean')], alpha=0.5, align='center', capsize=5)
        ax.set_title(f'{metric.capitalize()} Statistics')
        ax.set_xlabel('Cluster')
        ax.set_ylabel(metric.capitalize())
        ax.set_xticks(cluster_statistics.index)

    plt.tight_layout()
    plt.show()


# Call the function to compute and plot cluster statistics
cluster_statistics = compute_cluster_statistics(engagement_metrics)
print(cluster_statistics)
plot_cluster_statistics(cluster_statistics)


# In[32]:


print(user_data.columns)


# In[33]:


class ApplicationEngagement:
    def __init__(self, user_data):
        self.user_data = user_data

    def top_engaged_users_per_app(self):
        user_traffic_per_app = self.user_data.groupby('MSISDN/Number')[['social_media_data', 'google_data',
                                                                        'email_data', 'youtube_data',
                                                                        'netflix_data', 'gaming_data',
                                                                        'other_data']].sum()

        top_10_users_per_app = {}
        for column in user_traffic_per_app.columns:
            top_10_users_per_app[column] = user_traffic_per_app[column].nlargest(10)

        self.plot_top_engaged_users(top_10_users_per_app)
        return top_10_users_per_app

    def plot_top_engaged_users(self, top_10_users_per_app):
        num_apps = len(top_10_users_per_app)
        num_rows = 2  # Adjust the number of rows and columns as needed
        num_cols = int(np.ceil(num_apps / num_rows))

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 12))

        for i, (app, top_users) in enumerate(top_10_users_per_app.items()):
            ax = axes[i // num_cols, i % num_cols] if num_apps > 1 else axes
            top_users.plot(kind='bar', color='skyblue', ax=ax)
            ax.set_title(f'Top 10 Users - {app}')
            ax.set_xlabel('MSISDN/Number')
            ax.set_ylabel('Total Traffic (Bytes)')
            ax.tick_params(axis='x', rotation=45)
        
        for j in range(i + 1, num_rows * num_cols):
            fig.delaxes(axes[j // num_cols, j % num_cols])

        plt.tight_layout()
        plt.show()

# Example usage:
app_engagement = ApplicationEngagement(user_data)
top_users_per_app = app_engagement.top_engaged_users_per_app()
print(top_users_per_app)


# In[34]:


# Top three mostly used apps


# In[43]:


# Calculate total data usage for each application
total_data_usage = user_data[applications].sum()

# Sort applications based on total data usage and select the top three
top_three_apps = total_data_usage.nlargest(3)

# Prepare labels with both application names, percentages, and usage values for top three apps
labels = [f"{app}\n({total_data_usage[app]:,.0f} MB, {percent:.1f}%)"
          for app, percent in zip(top_three_apps.index, (top_three_apps / top_three_apps.sum() * 100))]

# Plotting pie chart for top three apps
plt.figure(figsize=(8, 8))
patches, texts, autotexts = plt.pie(top_three_apps, labels=labels, autopct='%1.1f%%', startangle=140)

# Adjust layout and spacing of labels
for text, autotext in zip(texts, autotexts):
    text.set_size('smaller')
    autotext.set_size('x-small')

plt.title('Proportion of Data Usage by Top Three Applications')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

# Add legend outside the pie chart
plt.legend(top_three_apps.index, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

plt.tight_layout()
plt.show()
total_data_usage
top_three_apps


# In[9]:


# Define connection parameters
dbname = "telecomdb"
user = "postgres"
password = "admin"
host = "localhost"
port = "5432"

try:
    # Establish a connection to the database
    conn = psycopg2.connect(
        dbname=dbname,
        user=user,
        password=password,
        host=host,
        port=port
    )
    
    # Create a cursor object using the connection
    cursor = conn.cursor()
    
    # Query to fetch all columns of the table "xdr_data"
    query = """
    SELECT *
    FROM public.xdr_data
    """
    
    # Execute the query
    cursor.execute(query)
    
    # Fetch all results into a DataFrame
    user_data1 = pd.DataFrame(cursor.fetchall(), columns=[desc[0] for desc in cursor.description])

    # Close the cursor
    cursor.close()

except OperationalError as e:
    print("Error: Could not connect to the database.", e)

finally:
    # Close the connection
    if 'conn' in locals() and conn is not None:
        conn.close()

# Now you have the complete user_data DataFrame
# You can use this DataFrame in your UserExperienceAnalysis class


# In[25]:


user_data.head()


# In[14]:


user_data1.info()


# In[28]:


user_data1.columns


# In[ ]:


# Task 3 - Experience Analytics


# # Aggregate per customer information

# In[22]:


class UserExperienceAnalysis:
    def __init__(self, user_data1):
        self.user_data1 = user_data1

    def aggregate_per_customer(self):
        # Aggregate per customer information
        per_customer_info = self.user_data1.groupby('MSISDN/Number').agg({
            'Avg RTT DL (ms)': 'mean',
            'Avg Bearer TP DL (kbps)': 'mean',
            'Handset Type': lambda x: x.mode()[0] if len(x.mode()) > 0 else None,
            'TCP DL Retrans. Vol (Bytes)': 'mean'
        })
        return per_customer_info

    def summary_statistics(self, per_customer_info):
        # Summary statistics
        summary_stats = per_customer_info.describe()
        return summary_stats

# Instantiate the UserExperienceAnalysis class with your user_data DataFrame
experience_analytics = UserExperienceAnalysis(user_data1)

# Call the method to aggregate per customer information
per_customer_info = experience_analytics.aggregate_per_customer()

# Compute and print summary statistics
summary_stats = experience_analytics.summary_statistics(per_customer_info)
print(summary_stats)


# In[25]:


#  Compute and list top, bottom, and most frequent value


# In[15]:


import matplotlib.pyplot as plt

class UserExperienceAnalysis:
    def __init__(self, user_data1):
        self.user_data1 = user_data1

    def aggregate_per_customer(self):
        # Aggregate per customer information
        per_customer_info = self.user_data1.groupby('MSISDN/Number').agg({
            'Avg RTT DL (ms)': 'mean',
            'Avg Bearer TP DL (kbps)': 'mean',
            'Handset Type': lambda x: x.mode()[0] if len(x.mode()) > 0 else None,
            'TCP DL Retrans. Vol (Bytes)': 'mean'
        })
        return per_customer_info

    def summary_statistics(self, per_customer_info):
        # Summary statistics
        summary_stats = per_customer_info.describe()
        return summary_stats

    def compute_top_bottom_frequent_values(self):
        # Compute top, bottom, and most frequent values for TCP, RTT, and Throughput
        top_bottom_frequent_values = {}

        # TCP values
        top_bottom_frequent_values['TCP'] = {
            'Top': self.user_data1['TCP DL Retrans. Vol (Bytes)'].nlargest(10),
            'Bottom': self.user_data1['TCP DL Retrans. Vol (Bytes)'].nsmallest(10),
            'Most Frequent': self.user_data1['TCP DL Retrans. Vol (Bytes)'].value_counts().head(10)
        }

        # RTT values
        top_bottom_frequent_values['RTT'] = {
            'Top': self.user_data1['Avg RTT DL (ms)'].nlargest(10),
            'Bottom': self.user_data1['Avg RTT DL (ms)'].nsmallest(10),
            'Most Frequent': self.user_data1['Avg RTT DL (ms)'].value_counts().head(10)
        }

        # Throughput values
        top_bottom_frequent_values['Throughput'] = {
            'Top': self.user_data1['Avg Bearer TP DL (kbps)'].nlargest(10),
            'Bottom': self.user_data1['Avg Bearer TP DL (kbps)'].nsmallest(10),
            'Most Frequent': self.user_data1['Avg Bearer TP DL (kbps)'].value_counts().head(10)
        }

        return top_bottom_frequent_values

    def plot_top_bottom_most_frequent(self, values, metric):
        plt.figure(figsize=(10, 6))

        # Plot top values
        plt.subplot(1, 3, 1)
        values['Top'].plot(kind='bar', color='blue')
        plt.title(f'Top 10 {metric} Values')
        plt.xlabel('Index')
        plt.ylabel(metric)

        # Plot bottom values
        plt.subplot(1, 3, 2)
        values['Bottom'].plot(kind='bar', color='red')
        plt.title(f'Bottom 10 {metric} Values')
        plt.xlabel('Index')
        plt.ylabel(metric)

        # Plot most frequent values
        plt.subplot(1, 3, 3)
        values['Most Frequent'].plot(kind='bar', color='green')
        plt.title(f'Most Frequent {metric} Values')
        plt.xlabel(metric)
        plt.ylabel('Count')

        plt.tight_layout()
        plt.show()

# Instantiate the UserExperienceAnalysis class with your user_data DataFrame
experience_analytics = UserExperienceAnalysis(user_data1)

# Call the method to compute top, bottom, and most frequent values for TCP, RTT, and Throughput
top_bottom_frequent_values = experience_analytics.compute_top_bottom_frequent_values()

# Plot TCP values
experience_analytics.plot_top_bottom_most_frequent(top_bottom_frequent_values['TCP'], 'TCP')

# Plot RTT values
experience_analytics.plot_top_bottom_most_frequent(top_bottom_frequent_values['RTT'], 'RTT')

# Plot Throughput values
experience_analytics.plot_top_bottom_most_frequent(top_bottom_frequent_values['Throughput'], 'Throughput')


# In[10]:


class UserExperienceAnalysis:
    def __init__(self, user_data1):
        self.user_data1 = user_data1

    def compute_top_bottom_frequent_values(self):
        # Compute top, bottom, and most frequent values
        top_tcp = self.user_data1['TCP DL Retrans. Vol (Bytes)'].nlargest(10)
        bottom_tcp = self.user_data1['TCP DL Retrans. Vol (Bytes)'].nsmallest(10)
        most_frequent_tcp = self.user_data1['TCP DL Retrans. Vol (Bytes)'].mode()

        top_rtt = self.user_data1['Avg RTT DL (ms)'].nlargest(10)
        bottom_rtt = self.user_data1['Avg RTT DL (ms)'].nsmallest(10)
        most_frequent_rtt = self.user_data1['Avg RTT DL (ms)'].mode()

        top_throughput = self.user_data1['Avg Bearer TP DL (kbps)'].nlargest(10)
        bottom_throughput = self.user_data1['Avg Bearer TP DL (kbps)'].nsmallest(10)
        most_frequent_throughput = self.user_data1['Avg Bearer TP DL (kbps)'].mode()

        return {
            'top_tcp': top_tcp,
            'bottom_tcp': bottom_tcp,
            'most_frequent_tcp': most_frequent_tcp,
            'top_rtt': top_rtt,
            'bottom_rtt': bottom_rtt,
            'most_frequent_rtt': most_frequent_rtt,
            'top_throughput': top_throughput,
            'bottom_throughput': bottom_throughput,
            'most_frequent_throughput': most_frequent_throughput
        }

# Instantiate the UserExperienceAnalysis class with your user_data DataFrame
experience_analytics = UserExperienceAnalysis(user_data1)

# Call the method to compute top, bottom, and most frequent values
result = experience_analytics.compute_top_bottom_frequent_values()

# Print the result
print(result)


# In[ ]:


# Compute and report distribution


# In[31]:


import pandas as pd
import matplotlib.pyplot as plt

# Group the data by handset type and calculate the average throughput and TCP retransmission
per_handset_statistics = user_data1.groupby('Handset Type').agg({
    'Avg Bearer TP DL (kbps)': 'mean',  # Average throughput
    'TCP DL Retrans. Vol (Bytes)': 'mean'  # Average TCP retransmission
})
grouped_handset_statistics = per_handset_statistics.groupby(lambda x: x[:1]).mean()

# Plotting the distributions
plt.figure(figsize=(12, 6))

# Plot for average throughput per handset type
plt.subplot(1, 2, 1)
grouped_handset_statistics['Avg Bearer TP DL (kbps)'].plot(kind='bar', color='blue', alpha=0.7)
plt.title('Average Throughput per Handset Type')
plt.xlabel('Handset Type Group')
plt.ylabel('Average Throughput (kbps)')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

# Plot for average TCP retransmission per handset type
plt.subplot(1, 2, 2)
grouped_handset_statistics['TCP DL Retrans. Vol (Bytes)'].plot(kind='bar', color='red', alpha=0.7)
plt.title('Average TCP Retransmission per Handset Type')
plt.xlabel('Handset Type Group')
plt.ylabel('Average TCP Retransmission Volume (Bytes)')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability

plt.tight_layout()
plt.show()


# In[ ]:


#  Perform k-means clustering


# In[32]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class UserExperienceAnalysis:
    def __init__(self, user_data1):
        self.user_data1 = user_data1

    def normalize_and_cluster(self, columns):
        # Select the relevant columns and drop missing values
        selected_data = self.user_data1[columns].dropna()

        # Normalize the data
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(selected_data)

        # Perform k-means clustering with k=3
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(normalized_data)
        cluster_labels = kmeans.labels_

        # Add cluster labels to the DataFrame
        selected_data['Cluster'] = cluster_labels

        return selected_data

    def visualize_clusters(self, clustered_data):
        plt.figure(figsize=(10, 6))

        # Plot each cluster separately
        for cluster in clustered_data['Cluster'].unique():
            cluster_data = clustered_data[clustered_data['Cluster'] == cluster]
            plt.scatter(cluster_data.iloc[:, 0],
                        cluster_data.iloc[:, 1],
                        label=f'Cluster {cluster}')

        plt.title('User Experience Clusters')
        plt.xlabel('Average Bearer TP DL (kbps)')
        plt.ylabel('Average RTT DL (ms)')
        plt.legend()
        plt.grid(True)
        plt.show()

# Instantiate the UserExperienceAnalysis class with your user_data1 DataFrame
experience_analytics = UserExperienceAnalysis(user_data1)

# Define the columns for k-means clustering
columns_for_clustering = ['Avg Bearer TP DL (kbps)', 'Avg RTT DL (ms)', 'TCP DL Retrans. Vol (Bytes)']

# Call the method to normalize and cluster
clustered_data = experience_analytics.normalize_and_cluster(columns_for_clustering)

# Visualize the clusters
experience_analytics.visualize_clusters(clustered_data)


# In[51]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
class UserExperienceAnalysis:
    def __init__(self, user_data):
        self.user_data = user_data

    def normalize_and_cluster(self):
        # Select relevant columns for clustering
        upload_download_data = self.user_data[['Total UL (Bytes)', 'Total DL (Bytes)']]

        # Drop rows with missing values
        upload_download_data = upload_download_data.dropna()

        # Normalize the data
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(upload_download_data)

        # Perform k-means clustering with k=3
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(normalized_data)
        cluster_labels = kmeans.labels_

        # Add cluster labels to the DataFrame
        upload_download_data['Cluster'] = cluster_labels

        return upload_download_data

    def visualize_clusters(self, clustered_data):
        plt.figure(figsize=(10, 6))

        # Plot each cluster separately
        for cluster in clustered_data['Cluster'].unique():
            cluster_data = clustered_data[clustered_data['Cluster'] == cluster]
            plt.scatter(cluster_data['Total UL (Bytes)'],
                        cluster_data['Total DL (Bytes)'],
                        label=f'Cluster {cluster}')

        plt.title('User Experience Clusters based on Upload and Download')
        plt.xlabel('Total UL (Bytes)')
        plt.ylabel('Total DL (Bytes)')
        plt.legend()
        plt.grid(True)
        plt.show()

# Assuming user_data1 is your DataFrame containing the user data
experience_analytics = UserExperienceAnalysis(user_data1)

# Call the method to normalize and cluster
clustered_data = experience_analytics.normalize_and_cluster()

# Visualize the clusters
experience_analytics.visualize_clusters(clustered_data)

# Calculate centroids for less engaged and worst experience clusters
less_engaged_cluster_centroid = clustered_data[clustered_data['Cluster'] == 0][['Total UL (Bytes)', 'Total DL (Bytes)']].mean().values
worst_experience_cluster_centroid = clustered_data[clustered_data['Cluster'] == 2][['Total UL (Bytes)', 'Total DL (Bytes)']].mean().values

print("Centroid for less engaged cluster:", less_engaged_cluster_centroid)
print("Centroid for worst experience cluster:", worst_experience_cluster_centroid)


# 
# Based on the clustering results, we have three clusters distinguished by the average bearer throughput (Avg Bearer TP DL), average round-trip time (Avg RTT DL), and TCP download retransmission volume (TCP DL Retrans. Vol):
# 
# Cluster 0:
# 
# Average Bearer Throughput (Avg Bearer TP DL): High
# Average Round-Trip Time (Avg RTT DL): Low
# TCP Download Retransmission Volume (TCP DL Retrans. Vol): Low
# This cluster represents users with high throughput, low latency, and minimal TCP retransmission, indicating a stable and efficient network connection.
# Cluster 1:
# 
# Average Bearer Throughput (Avg Bearer TP DL): Moderate
# Average Round-Trip Time (Avg RTT DL): Moderate
# TCP Download Retransmission Volume (TCP DL Retrans. Vol): Moderate
# This cluster consists of users with moderate throughput, moderate latency, and moderate TCP retransmission, suggesting average network performance.
# Cluster 2:
# 
# Average Bearer Throughput (Avg Bearer TP DL): Low
# Average Round-Trip Time (Avg RTT DL): High
# TCP Download Retransmission Volume (TCP DL Retrans. Vol): High
# Users in this cluster experience low throughput, high latency, and high TCP retransmission, indicating poor network conditions and potential network congestion or packet loss issues.
# These clusters provide insights into different user experiences based on network performance metrics, allowing for targeted optimization strategies to improve overall user satisfaction and network efficiency.

# In[35]:


user_data1.head()


# In[34]:


# Task 4 - Satisfaction Analysis


# In[91]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

class UserExperienceAnalysis:
    def __init__(self, user_data):
        self.user_data = user_data

    def normalize_and_cluster(self):
        upload_download_data = self.user_data[['Avg Bearer TP DL (kbps)', 'Avg RTT DL (ms)']]
        upload_download_data = upload_download_data.dropna()
        
        # Normalize the data
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(upload_download_data)

        # Perform k-means clustering with k=3
        kmeans = KMeans(n_clusters=3)
        kmeans.fit(normalized_data)
        cluster_labels = kmeans.labels_

        # Add cluster labels to the DataFrame
        upload_download_data['Cluster'] = cluster_labels

        return upload_download_data

    def visualize_clusters(self, clustered_data):
        plt.figure(figsize=(10, 6))

        for cluster in clustered_data['Cluster'].unique():
            cluster_data = clustered_data[clustered_data['Cluster'] == cluster]
            plt.scatter(cluster_data['Avg Bearer TP DL (kbps)'],
                        cluster_data['Avg RTT DL (ms)'],
                        label=f'Cluster {cluster}')

        plt.title('User Experience Clusters based on Upload and Download')
        plt.xlabel('Avg Bearer TP DL (kbps)')
        plt.ylabel('Avg RTT DL (ms)')
        plt.legend()
        plt.grid(True)
        plt.show()

class SatisfactionAnalysis:
    def __init__(self, clustered_data, less_engaged_cluster, worst_experience_cluster):
        self.clustered_data = clustered_data
        self.less_engaged_cluster = less_engaged_cluster
        self.worst_experience_cluster = worst_experience_cluster

    def calculate_engagement_score(self, user_data_point):
        # Reshape cluster centroid to match the shape of the user data point
        less_engaged_cluster_reshaped = np.reshape(self.less_engaged_cluster, (1, -1))

        # Calculate Euclidean distance between user data point and less engaged cluster centroid
        engagement_score = np.linalg.norm(user_data_point - less_engaged_cluster_reshaped)
        return engagement_score

    def calculate_experience_score(self, user_data_point):
        # Reshape cluster centroid to match the shape of the user data point
        worst_experience_cluster_reshaped = np.reshape(self.worst_experience_cluster, (1, -1))

        # Calculate Euclidean distance between user data point and worst experience cluster centroid
        experience_score = np.linalg.norm(user_data_point - worst_experience_cluster_reshaped)
        return experience_score

# Instantiate SatisfactionAnalysis with clustered_data, less_engaged_cluster, and worst_experience_cluster
satisfaction_analysis = SatisfactionAnalysis(clustered_data, less_engaged_cluster_centroid, worst_experience_cluster_centroid)

# Example: Calculate engagement score for a user data point
user_data_point = np.array([1000, 50])
engagement_score = satisfaction_analysis.calculate_engagement_score(user_data_point)
print("Engagement Score:", engagement_score)

# Example: Calculate experience score for a user data point
experience_score = satisfaction_analysis.calculate_experience_score(user_data_point)
print("Experience Score:", experience_score)


# In[89]:


clustered_data.head()


# Based on the output:
# - The engagement score is higher than the experience score, indicating that the user's behavior is relatively further away from the centroid of the less engaged cluster compared to the centroid of the worst experience cluster.
# - The higher engagement score suggests that the user's behavior is less aligned with the characteristics of users in the less engaged cluster, potentially indicating a higher level of engagement or activity.
# - The lower experience score suggests that the user's behavior is more aligned with the characteristics of users in the worst experience cluster, implying that the user may have a poorer overall experience compared to users in other clusters.
# 
# In summary, based on the output, we can infer that the user's engagement level is relatively higher, but their experience might not be as positive compared to users in other clusters. Further analysis and contextual information would be needed to understand the specific factors contributing to these scores and to provide more accurate interpretations.

# #  Calculate Satisfaction Score and Report Top 10 Satisfied Customers

# In[119]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
class SatisfactionAnalysis:
    def __init__(self, clustered_data, less_engaged_cluster, worst_experience_cluster):
        self.clustered_data = clustered_data
        self.less_engaged_cluster = less_engaged_cluster
        self.worst_experience_cluster = worst_experience_cluster
        self.actual_satisfaction_scores = None

    def calculate_engagement_score(self, user_data_point):
        # Reshape cluster centroid to match the shape of the user data point
        less_engaged_cluster_reshaped = np.reshape(self.less_engaged_cluster, (1, -1))

        # Calculate Euclidean distance between user data point and less engaged cluster centroid
        engagement_score = np.linalg.norm(user_data_point - less_engaged_cluster_reshaped)
        return engagement_score

    def calculate_experience_score(self, user_data_point):
        # Reshape cluster centroid to match the shape of the user data point
        worst_experience_cluster_reshaped = np.reshape(self.worst_experience_cluster, (1, -1))

        # Calculate Euclidean distance between user data point and worst experience cluster centroid
        experience_score = np.linalg.norm(user_data_point - worst_experience_cluster_reshaped)
        return experience_score

    def calculate_satisfaction_score(self, engagement_score, experience_score):
        # Calculate average of engagement and experience scores as satisfaction score
        satisfaction_score = (engagement_score + experience_score) / 2
        return satisfaction_score

    def top_satisfied_customers(self, satisfaction_scores, n=10):
        # Report top n satisfied customers based on satisfaction scores
        top_satisfied_customers = satisfaction_scores.nlargest(n, 'Satisfaction Score')
        return top_satisfied_customers

    def build_regression_model(self, features, target):
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        
        # Build and train the regression model
        regression_model = LinearRegression()
        regression_model.fit(X_train, y_train)
        
        return regression_model

    def store_actual_satisfaction_scores(self, actual_satisfaction_scores):
        self.actual_satisfaction_scores = actual_satisfaction_scores

# Instantiate SatisfactionAnalysis with clustered_data, less_engaged_cluster, and worst_experience_cluster
satisfaction_analysis = SatisfactionAnalysis(clustered_data, less_engaged_cluster_centroid, worst_experience_cluster_centroid)

# Calculate engagement score for all users
clustered_data['Engagement Score'] = clustered_data.apply(lambda row: satisfaction_analysis.calculate_engagement_score(row[['Avg Bearer TP DL (kbps)', 'Avg RTT DL (ms)']].values), axis=1)

# Calculate experience score for all users
clustered_data['Experience Score'] = clustered_data.apply(lambda row: satisfaction_analysis.calculate_experience_score(row[['Avg Bearer TP DL (kbps)', 'Avg RTT DL (ms)']].values), axis=1)

# Calculate satisfaction score for all users
clustered_data['Satisfaction Score'] = clustered_data.apply(lambda row: satisfaction_analysis.calculate_satisfaction_score(row['Engagement Score'], row['Experience Score']), axis=1)

# Report top 10 satisfied customers
top_satisfied_customers = satisfaction_analysis.top_satisfied_customers(clustered_data[['Satisfaction Score', 'Engagement Score', 'Experience Score']])
print("Top 10 Satisfied Customers:")
print(top_satisfied_customers)

# Store actual satisfaction scores
actual_satisfaction_scores = clustered_data['Satisfaction Score']
satisfaction_analysis.store_actual_satisfaction_scores(actual_satisfaction_scores)


# In[140]:


actual_satisfaction_scores.head()


# In[104]:


# Extract the top 10 satisfied customers and their satisfaction scores
top_satisfied_customers = satisfaction_analysis.top_satisfied_customers(clustered_data[['Satisfaction Score', 'Engagement Score', 'Experience Score']])

# Plot the satisfaction scores using a bar graph
plt.figure(figsize=(10, 6))
plt.bar(top_satisfied_customers.index.astype(str), top_satisfied_customers['Satisfaction Score'], color='blue')
plt.xlabel('Customer Index')
plt.ylabel('Satisfaction Score')
plt.title('Top 10 Satisfied Customers')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()


# # Task 4.3: Build a Regression Model for Satisfaction Prediction

# In[137]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

class SatisfactionAnalysis:
    def __init__(self, clustered_data, less_engaged_cluster, worst_experience_cluster):
        self.clustered_data = clustered_data
        self.less_engaged_cluster = less_engaged_cluster
        self.worst_experience_cluster = worst_experience_cluster

    def build_regression_model(self, features, target):
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        
        # Build and train the regression model
        regression_model = LinearRegression()
        regression_model.fit(X_train, y_train)
        
        return regression_model

    def predict_satisfaction(self, regression_model, features):
        # Predict satisfaction scores using the regression model
        satisfaction_predictions = regression_model.predict(features)
        return satisfaction_predictions

# Instantiate SatisfactionAnalysis with clustered_data, less_engaged_cluster, and worst_experience_cluster
satisfaction_analysis = SatisfactionAnalysis(clustered_data, less_engaged_cluster_centroid, worst_experience_cluster_centroid)

# Example: Build regression model for satisfaction prediction
features = clustered_data[['Engagement Score', 'Experience Score']]
target = clustered_data['Satisfaction Score']
regression_model = satisfaction_analysis.build_regression_model(features, target)

# Example: Generate predictions for satisfaction
satisfaction_predictions = satisfaction_analysis.predict_satisfaction(regression_model, features)

# Convert the NumPy array to a pandas DataFrame
satisfaction_predictions_df = pd.DataFrame(satisfaction_predictions, columns=['Predictions'])
print("Satisfaction Predictions:")
print(satisfaction_predictions)


# In[139]:


satisfaction_predictions_df.head()


# In[116]:


# Plot satisfaction predictions against index
plt.figure(figsize=(10, 6))
plt.plot(range(len(satisfaction_predictions)), satisfaction_predictions, marker='o', linestyle='', color='b')
plt.title('Satisfaction Predictions')
plt.xlabel('User Index')
plt.ylabel('Predicted Satisfaction Score')
plt.grid(True)
plt.show()


# # Task 4.4: k-means (k=2) on the engagement & the experience score.

# In[141]:


# Combine engagement and experience scores
X_cluster = satisfaction_analysis.clustered_data[['Engagement Score', 'Experience Score']]
# Perform k-means clustering with k=2
kmeans = KMeans(n_clusters=2)
kmeans.fit(X_cluster)
cluster_labels = kmeans.labels_

# Add cluster labels to the DataFrame
satisfaction_analysis.clustered_data['Cluster'] = cluster_labels

# Visualize the clusters
plt.figure(figsize=(8, 6))
for cluster in satisfaction_analysis.clustered_data['Cluster'].unique():
    cluster_data = satisfaction_analysis.clustered_data[satisfaction_analysis.clustered_data['Cluster'] == cluster]
    plt.scatter(cluster_data['Engagement Score'], cluster_data['Experience Score'], label=f'Cluster {cluster}')
plt.title('Engagement and Experience Clusters')
plt.xlabel('Engagement Score')
plt.ylabel('Experience Score')
plt.legend()
plt.grid(True)
plt.show()


# # Task 4.5: Aggregate the average satisfaction & experience score per cluster.

# In[143]:


# Aggregate average satisfaction and experience score per cluster
cluster_agg = satisfaction_analysis.clustered_data.groupby('Cluster')[['Satisfaction Score', 'Experience Score']].mean()
print("Aggregate Scores per Cluster:")
print(cluster_agg)

# Plotting
cluster_agg.plot(kind='bar', figsize=(10, 6))
plt.title('Aggregate Satisfaction and Experience Scores per Cluster')
plt.xlabel('Cluster')
plt.ylabel('Score')
plt.xticks(rotation=0)
plt.legend(['Satisfaction Score', 'Experience Score'])
plt.grid(True)
plt.show()


# In[ ]:





# # Task 4.6: Export your final table containing all user ID + engagement, experience & satisfaction scores in your local MySQL database.

# In[155]:


import psycopg2
from sqlalchemy import create_engine

# Connect to PostgreSQL database
engine = create_engine('postgresql+psycopg2://postgres:admin@localhost/userSatisfaction')

# Export DataFrame to PostgreSQL database
satisfaction_analysis.clustered_data.to_sql('user_satisfaction_scores', engine, if_exists='replace', index=False)


# In[156]:


import psycopg2
from sqlalchemy import create_engine

# Connect to PostgreSQL database
engine = create_engine('postgresql+psycopg2://postgres:admin@localhost/satisfactionPrediction')

# Export DataFrame to PostgreSQL database
satisfaction_analysis.clustered_data.to_sql('user_satisfaction_scores', engine, if_exists='replace', index=False)

