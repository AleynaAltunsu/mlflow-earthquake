# mlflow-earthquake
earthquake magnitude prediction system with using mlflow and flask
train.py is the main project and the other files are additional but not necesary.

Dataset Description
The dataset for this project contains earthquake records from Kandilli Observatory, which covers the period from 1915 to 2021. This dataset includes a variety of attributes related to each earthquake occurrence, including:
•	Deprem Kodu (Earthquake Code): Unique code for each earthquake event
•	Olus Tarihi (Date of Occurrence): Date of the earthquake occurrence
•	Olus Zamani (Time of Occurrence): Time of the earthquake occurrence
•	Enlem (Latitude): Latitude coordinate of the earthquake epicenter
•	Boylam (Longitude): Longitude coordinate of the earthquake epicenter
•	Derinlik (Depth): Depth of the earthquake in kilometers
•	xM: The largest magnitude value among MD, ML, Mw, Ms, and Mb
•	MD, ML, Mw, Ms, Mb: Different types of magnitude measurements
•	Yer (Place): Place where the earthquake occurred
The dataset contains a total of 17,370 records. It provides both spatial and temporal features, making it suitable for building predictive models for earthquake occurrences. From Kaggle.
________________________________________
Project Objectives
The goal of this project is to demonstrate how MLflow can be used to manage the entire machine learning lifecycle, including the following objectives:
1.	Experiment Tracking: Using MLflow to log different experiments, parameters, metrics, and outputs.
2.	Model Training and Hyperparameter Tuning: Implementing machine learning models and tuning hyperparameters using MLflow to track the training process.
3.	Model Deployment: Using MLflow’s model packaging and deployment capabilities to serve the model via an API.
4.	Performance Monitoring: Monitoring the deployed model over time to track performance and detect any model drift.
5.	Model Versioning and Registry: Using MLflow’s Model Registry to manage model versions and stage transitions from staging to production.

