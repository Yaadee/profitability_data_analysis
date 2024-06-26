import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import mlflow
from mlflow.tracking import MlflowClient
import sqlite3
import sys

# Import SatisfactionAnalysis class from EDAanalysis module
sys.path.insert(0, '/path/to/notebook')
from EDAanalysis import SatisfactionAnalysis

class Model:
    def __init__(self, clustered_data, less_engaged_cluster, worst_experience_cluster):
        self.clustered_data = clustered_data
        self.less_engaged_cluster = less_engaged_cluster
        self.worst_experience_cluster = worst_experience_cluster
        self.actual_satisfaction_scores = None
        self.regression_model = None

# Fetch clustered data from SQLite database and convert to DataFrame
conn = sqlite3.connect('clusteredData.db')
clustered_data = pd.read_sql_query("SELECT * FROM clustered_data", conn)
conn.close()

# Define centroids for less engaged and worst experience clusters
less_engaged_cluster_centroid = np.array([10, 20])
worst_experience_cluster_centroid = np.array([30, 40])

# Instantiate SatisfactionAnalysis object
satisfaction_analysis = SatisfactionAnalysis(clustered_data, less_engaged_cluster_centroid, worst_experience_cluster_centroid)

# Build regression model
features = clustered_data[['Engagement Score', 'Experience Score']]
target = clustered_data['Satisfaction Score']
regression_model = satisfaction_analysis.build_regression_model(features, target)

# Model deployment tracking
with mlflow.start_run(run_name="Model Deployment Tracking"):
    mlflow.log_param("code_version", "1.0")
    mlflow.log_param("source", "test.py")
    mlflow.log_params(regression_model.get_params())  # Log model parameters
    mlflow.log_metric("loss_convergence", 0.05)
    mlflow.log_artifact("clustered_data.csv")  # Save clustered data to CSV and log as artifact
    mlflow.log_model(regression_model, "model")  # Log the regression model

# Example usage: Predict satisfaction for new data
new_data = np.array([[15, 25], [35, 45]])  # Example new data
satisfaction_predictions = satisfaction_analysis.predict_satisfaction(new_data)
print("Satisfaction Predictions for New Data:")
print(satisfaction_predictions)
