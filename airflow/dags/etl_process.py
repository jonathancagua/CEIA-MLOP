import datetime
from airflow.decorators import dag, task

markdown_text = """
### ETL Process for Spotify Data from a Song Collection

This DAG extracts data from the original CSV file stored in Drive [File Link](https://drive.google.com/file/d/13DDhnS2FoXN-xqWM9PWryQyUBgRTqcKg/view?usp=sharing), preprocesses by creating dummy variables and scaling numerical features.

Then, the data is saved back to an S3 bucket as two separate CSV files: one for training and one for testing, with a stratified 70/30 split.

"""

default_args = {
    'owner': "Jonathan-Juan",
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15)
}

@dag(
    dag_id="process_etl_spotify_data",
    description="ETL process for spotify data, separating the dataset into training and testing sets.",
    doc_md=markdown_text,
    tags=["ETL", "spotify"],
    default_args=default_args,
    catchup=False,
)
def process_etl_spotify_data():

    @task.virtualenv(
        task_id="obtain_original_data",
        requirements=["awswrangler==3.6.0"],
        system_site_packages=True
    )
    def get_data():
        """
        Load the raw data from Drive
        """
        import awswrangler as wr
        from airflow.models import Variable
        import pandas as pd
        import requests

        # Load the dataset
        with requests.get(
            "https://drive.google.com/uc?export=download&id=13DDhnS2FoXN-xqWM9PWryQyUBgRTqcKg"
        ) as r, open("data_playlist.csv", "wb") as f:
            for chunk in r.iter_content():
                f.write(chunk)
        spotify_df = pd.read_csv("data_playlist.csv")
        data_path = "s3://data/raw/data_playlist.csv"
        
        # Save the DataFrame to the specified S3 path
        wr.s3.to_csv(df=spotify_df, path=data_path, index=False)


    @task.virtualenv(
        task_id="make_feat_eng_variables",
        requirements=["awswrangler==3.6.0",
                      "scikit-learn==1.3.2"],
        system_site_packages=True
    )
    def make_feat_eng_variables():
        """
        Perform Feature Engineering
        """
        import json
        import datetime
        import boto3
        import botocore.exceptions
        import mlflow
        import awswrangler as wr
        import numpy as np
        from airflow.models import Variable
        from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
        import pandas as pd

        data_original_path = "s3://data/raw/data_playlist.csv"
        data_end_path = "s3://data/raw/data_playlist_feat_eng.csv"
        json_path = "data_info/data.json"
        spotify_df = wr.s3.read_csv(data_original_path)

        # Remove duplicates
        spotify_df_filter = spotify_df[~spotify_df.duplicated()]
        # Reset indices if necessary
        spotify_df_filter = spotify_df_filter.reset_index(drop=True)

        # Define variables to scale
        variables = ['duration', 'tempo', 'loudness']

        spotify_df_filter_scaler = spotify_df_filter.drop(variables, axis=1)
        
        # Scale each variable with MinMaxScaler
        for variable in variables:
            scaler = MinMaxScaler()
            scaled_values = scaler.fit_transform(spotify_df_filter[[variable]])
            scaled_values = pd.DataFrame(scaled_values, columns=[variable])
            spotify_df_filter_scaler[variable] = scaled_values

        # Select features for Feature Engineering
        keep_features = ['speechiness','energy','danceability','acousticness','label']
        spotify_df_filter_scaler_feat_eng = spotify_df_filter_scaler[keep_features]

        # Save the DataFrame to the specified S3 path
        wr.s3.to_csv(df=spotify_df_filter_scaler_feat_eng, path=data_end_path, index=False)

        # Save information of the dataset
        client = boto3.client('s3')

        # Attempt to load existing JSON, or create a new one if not present
        data_dict = {}
        try:
            client.head_object(Bucket='data', Key=json_path)
            result = client.get_object(Bucket='data', Key=json_path)
            text = result["Body"].read().decode()
            data_dict = json.loads(text)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] != "404":
                # Something else has gone wrong.
                raise e

        # Extract column information for JSON
        data_dict['columns'] = spotify_df.columns.to_list()
        data_dict['columns_after_scaling'] = spotify_df_filter_scaler_feat_eng.columns.to_list()
        data_dict['date'] = datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S')
        data_dict['columns_dtypes'] = {k: str(v) for k, v in spotify_df.dtypes.to_dict().items()}
        data_dict['columns_dtypes_after_scaling'] = {k: str(v) for k, v in spotify_df_filter_scaler_feat_eng.dtypes.to_dict().items()}

        # Save updated JSON file to S3
        data_string = json.dumps(data_dict, indent=2)
        client.put_object(Bucket='data', Key=json_path, Body=data_string)

        # Configure MLflow and register the ETL
        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Spotify data")

        with mlflow.start_run(run_name='ETL_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S'),
                            experiment_id=experiment.experiment_id):
            mlflow.log_param("source", data_original_path)
            mlflow.log_param("output", data_end_path)
            mlflow.log_dict(data_dict, "data_info.json")

            # Create and register datasets in MLflow
            mlflow_dataset = mlflow.data.from_pandas(spotify_df,
                                                    source=data_original_path,
                                                    name="spotify_data_complete")
            mlflow_dataset_scaled = mlflow.data.from_pandas(spotify_df_filter_scaler_feat_eng,
                                                            source=data_end_path,
                                                            name="spotify_data_with_feat_eng")
            
            mlflow.log_input(mlflow_dataset, context="Dataset")
            mlflow.log_input(mlflow_dataset_scaled, context="Dataset")

        print("ETL process completed and registered in MLflow.")

    @task.virtualenv(
        task_id="split_dataset",
        requirements=["awswrangler==3.6.0",
                      "scikit-learn==1.3.2"],
        system_site_packages=True
    )
    def split_dataset():
        """
        Generate a dataset split into a training part and a test part
        """
        import awswrangler as wr
        from sklearn.model_selection import train_test_split
        from airflow.models import Variable

        def save_to_csv(df, path):
            wr.s3.to_csv(df=df, path=path, index=False)

        data_original_path = "s3://data/raw/data_playlist_feat_eng.csv"
        dataset = wr.s3.read_csv(data_original_path)

        test_size = Variable.get("test_size_spotify")
        target_col = Variable.get("target_col_spotify")

        X = dataset.drop(columns=target_col)
        y = dataset[[target_col]]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y)

        # Remove duplicates
        dataset.drop_duplicates(inplace=True, ignore_index=True)

        save_to_csv(X_train, "s3://data/final/train/spotify_X_train.csv")
        save_to_csv(X_test, "s3://data/final/test/spotify_X_test.csv")
        save_to_csv(y_train, "s3://data/final/train/spotify_y_train.csv")
        save_to_csv(y_test, "s3://data/final/test/spotify_y_test.csv")

    @task.virtualenv(
        task_id="register_info_data",
        requirements=["awswrangler==3.6.0",
                      "scikit-learn==1.3.2",
                      "mlflow==2.10.2"],
        system_site_packages=True
    )
    def register_info_data():
        """
        Register dataset information
        """
        import json
        import mlflow
        import boto3
        import botocore.exceptions

        import awswrangler as wr
        import pandas as pd

        def save_to_csv(df, path):
            wr.s3.to_csv(df=df, path=path, index=False)

        X_train = wr.s3.read_csv("s3://data/final/train/spotify_X_train.csv")
        X_test = wr.s3.read_csv("s3://data/final/test/spotify_X_test.csv")

        # Calculate mean and standard deviation for X_train
        mean_values = X_train.mean().tolist()
        std_values = X_train.std().tolist()

        # Check if metadata file exists in S3 and load it
        client = boto3.client('s3')
        data_dict = {}

        try:
            client.head_object(Bucket='data', Key='data_info/data.json')
            result = client.get_object(Bucket='data', Key='data_info/data.json')
            text = result["Body"].read().decode()
            data_dict = json.loads(text)
        except botocore.exceptions.ClientError as e:
                # Something else has gone wrong.
                raise e

        # Update metadata with dataset info and statistics
        data_dict['train_observations'] = X_train.shape[0]
        data_dict['test_observations'] = X_test.shape[0]
        data_dict['feature_names'] = X_train.columns.tolist()
        data_dict['mean'] = mean_values
        data_dict['std'] = std_values

        # Save updated metadata to S3
        data_string = json.dumps(data_dict, indent=2)
        client.put_object(Bucket='data', Key='data_info/data.json', Body=data_string)

        # Log information in MLflow
        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Spotify data")

        # Obtain the last experiment run_id to log the new information
        list_run = mlflow.search_runs([experiment.experiment_id], output_format="list")

        with mlflow.start_run(run_id=list_run[0].info.run_id):
            mlflow.log_param("Train observations", X_train.shape[0])
            mlflow.log_param("Test observations", X_test.shape[0])
            mlflow.log_param("Feature names", X_train.columns.tolist())
            mlflow.log_param("Mean values", mean_values)
            mlflow.log_param("Standard deviation values", std_values)

        print("Dataset information (including mean and std) saved to S3 and logged in MLflow.")
#

    get_data() >> make_feat_eng_variables() >> split_dataset() >> register_info_data()


dag = process_etl_spotify_data()