# Import necessary libraries and modules
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.lab import (
    load_data,
    data_preprocessing,
    build_save_model,
    load_model_elbow,
    train_kmeans_best,
    train_gmm_best,
    write_predictions_csv,
)

# NOTE:
# In Airflow 3.x, enabling XCom pickling should be done via environment variable:
# export AIRFLOW__CORE__ENABLE_XCOM_PICKLING=True
# The old airflow.configuration API is deprecated.

# Define default arguments for your DAG
default_args = {
    'owner': 'your_name',
    'start_date': datetime(2025, 1, 15),
    'retries': 0,  # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5),  # Delay before retries
}

# Create a DAG instance named 'Airflow_Lab1' with the defined default arguments
with DAG(
    'Airflow_Lab1',
    default_args=default_args,
    description='Dag example for Lab 1 of Airflow series',
    catchup=False,
) as dag:

    # Task to load data, calls the 'load_data' Python function
    load_data_task = PythonOperator(
        task_id='load_data_task',
        python_callable=load_data,
    )

    # Task to perform data preprocessing, depends on 'load_data_task'
    data_preprocessing_task = PythonOperator(
        task_id='data_preprocessing_task',
        python_callable=data_preprocessing,
        op_args=[load_data_task.output],
    )

    # Task to build and save a model, depends on 'data_preprocessing_task'
    build_save_model_task = PythonOperator(
        task_id='build_save_model_task',
        python_callable=build_save_model,
        op_args=[data_preprocessing_task.output, "model.sav"],
    )

    # Task to load a model using the 'load_model_elbow' function, depends on 'build_save_model_task'
    load_model_task = PythonOperator(
        task_id='load_model_task',
        python_callable=load_model_elbow,
        op_args=["model.sav", build_save_model_task.output],
    )

        # Task: Train KMeans (best k via silhouette) and save model
    train_kmeans_task = PythonOperator(
        task_id="train_kmeans_task",
        python_callable=train_kmeans_best,
        op_args=[data_preprocessing_task.output, "kmeans.sav"],
    )

    # Task: Train GMM (best k via silhouette) and save model
    train_gmm_task = PythonOperator(
        task_id="train_gmm_task",
        python_callable=train_gmm_best,
        op_args=[data_preprocessing_task.output, "gmm.sav"],
    )

    # Task: Write a CSV with both KMeans and GMM predictions
    write_predictions_task = PythonOperator(
        task_id="write_predictions_task",
        python_callable=write_predictions_csv,
        op_args=[
            data_preprocessing_task.output,
            train_kmeans_task.output,
            train_gmm_task.output,
            "cluster_predictions.csv",
        ],
    )


    # Set task dependencies
    load_data_task >> data_preprocessing_task >> build_save_model_task >> load_model_task
    data_preprocessing_task >> [train_kmeans_task, train_gmm_task] >> write_predictions_task

# If this script is run directly, allow command-line interaction with the DAG
if __name__ == "__main__":
    dag.test()
