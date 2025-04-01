# AI-Powered Personalized Fitness and Nutrition Recommendation Engine

[![Professional](https://img.shields.io/badge/Professional-Quality-blue)](https://www.linkedin.com/in/yourprofile)

## ✨ Project Overview

This project implements an AI-powered engine designed to provide personalized fitness and nutrition recommendations. By leveraging machine learning models, the engine analyzes user profiles and exercise capabilities to predict fitness levels, suggest tailored training plans, and recommend appropriate dietary needs for individuals aiming to build mass and strength. This solution is built to be easily deployed and run on Google Colab, making it accessible for users with varying levels of technical expertise.

## 📜 Table of Contents

1.  [Project Objective](#project-objective)
2.  [File and Folder Structure](#file-and-folder-structure)
3.  [Code Explanation](#code-explanation)
    * [Configuration Management](#configuration-management)
    * [Data Loading and Preprocessing](#data-loading-and-preprocessing)
    * [Model Training](#model-training)
    * [Hyperparameter Tuning](#hyperparameter-tuning)
    * [Prediction Engine](#prediction-engine)
    * [Training Function](#training-function)
    * [Main Prediction Function](#main-prediction-function)
    * [Conceptual Deployment](#conceptual-deployment)
4.  [Input](#input)
    * [Data Files (CSV)](#data-files-csv)
    * [User Profile Dictionary](#user-profile-dictionary)
5.  [Output](#output)
    * [Predicted Fitness Level](#predicted-fitness-level)
    * [Predicted Training Plan](#predicted-training-plan)
    * [Predicted Dietary Needs](#predicted-dietary-needs)
6.  [Usage](#usage)
    * [Prerequisites](#prerequisites)
    * [Setup and Execution in Google Colab](#setup-and-execution-in-google-colab)

## 🎯 Project Objective

The primary objective of this project is to create a robust and accurate AI model that can:

* **Predict a user's fitness level** (e.g., Beginner, Intermediate, Advanced) based on their profile and exercise capabilities.
* **Generate a personalized training plan** by recommending sets and repetitions for key exercises, specifically tailored for building mass and strength.
* **Recommend daily dietary needs** in terms of calorie intake and macronutrient targets (protein, carbohydrates, and fats) to support muscle growth.

This project aims to provide users with data-driven recommendations to help them achieve their fitness goals more effectively.

## 📂 File and Folder Structure

The project is organized with the following file and folder structure:

```
Projects/
└── Get_Fit_App/
    └── Ai_Model/
        ├── Data_Source/
        │   ├── fitness_level_data.csv
        │   ├── training_params_data.csv
        │   ├── dietary_needs_data.csv
        │   └── exercise_database.csv
        ├── Generated_Models/
        │   ├── fitness_level_model.pkl
        │   ├── training_params_model.pkl
        │   ├── dietary_needs_model.pkl
        │   ├── fitness_preprocessor.pkl
        │   ├── training_preprocessor.pkl
        │   └── dietary_needs_preprocessor.pkl
        ├── ai_model.ipynb         # The main Python code for the AI model (to be run in Google Colab)
        └── README.md            # This file
```

**Explanation:**

* **`Data_Source/`**: This folder contains the CSV files used to train the AI models.
    * `fitness_level_data.csv`: Data for training the fitness level classification model.
    * `training_params_data.csv`: Data for training the training parameter prediction model.
    * `dietary_needs_data.csv`: Data for training the dietary needs prediction model.
    * `exercise_database.csv`: A database of exercises with relevant information.
* **`Generated_Models/`**: This folder will store the trained AI models and preprocessing objects after the training process is complete.
    * `*.pkl` files: These are the serialized (saved) machine learning models and preprocessing pipelines.
* **`ai_model.ipynb`**: This is the main Jupyter Notebook file containing the Python code for the AI model. You will run this file in Google Colab.
* **`README.md`**: This file provides an overview of the project, instructions for usage, and other relevant information.

## 💻 Code Explanation

The Python code in `ai_model.ipynb` is structured into several modules to ensure clarity and maintainability:

### Configuration Management

The `Configuration` class (using `dataclass`) manages all the important file paths, model names, and hyperparameters used throughout the project. This makes it easy to update settings in one central location.

### Data Loading and Preprocessing

The `DataPreprocessor` class handles loading data from the CSV files, performing basic data cleaning (handling missing values, infinite values), and implementing feature engineering (e.g., calculating BMI). It also includes functions for splitting data into training and testing sets and creating preprocessing pipelines using `sklearn.compose.ColumnTransformer` for numerical (scaling) and categorical (one-hot encoding) features.

### Model Training

The `ModelTrainer` class is responsible for training and evaluating individual machine learning models. It supports different model types (fitness level classification using `RandomForestClassifier`, training parameter regression using `RandomForestRegressor`, and dietary needs regression using `LinearRegression`). It includes functions to train a specified model, evaluate its performance using relevant metrics (accuracy, classification report, mean squared error, R-squared), and save/load the trained model using `joblib`.

### Hyperparameter Tuning

The `HyperparameterTuner` class (currently not actively used in the provided code but included for potential future enhancements) is designed to perform hyperparameter optimization using `sklearn.model_selection.GridSearchCV` to find the best parameters for the AI models.

### Prediction Engine

The `PredictionEngine` class handles loading the trained AI models and their associated preprocessing objects. It provides separate functions (`predict_fitness_level`, `predict_training_plan`, `predict_dietary_needs`) to make predictions based on a user's profile provided as a Python dictionary. It ensures that the input data for prediction is preprocessed in the same way as the training data.

### Training Function

The `train_models` function orchestrates the training process for all three AI models. It loads the respective datasets, preprocesses them, splits them into training and testing sets, creates and fits preprocessing pipelines, trains the models using the `ModelTrainer`, evaluates their performance, and saves the trained models and preprocessing objects to the `Generated_Models` folder.

### Main Prediction Function

The `main` function demonstrates how to load the trained models using the `PredictionEngine` and make predictions for a sample user profile. It prints the predicted fitness level, training plan (sets and reps for example exercises), and dietary needs (calories and macronutrient targets).

### Conceptual Deployment

The `deploy_model_conceptual` function provides a placeholder for how the trained models could be deployed in a real-world application (e.g., using an API).

## 📥 Input

The AI model requires two main types of input:

### Data Files (CSV)

These files are stored in the `Data_Source` folder on your Google Drive and are used for training the models. The structure and content of these files are crucial for the model's performance.

* **`fitness_level_data.csv`**: Contains historical data of users with features like age, gender, weight, height, exercise capabilities (e.g., pushups, squats, plank duration), and the target variable `fitness_level` (Beginner, Intermediate, Advanced).
* **`training_params_data.csv`**: Contains historical data linking user profiles and exercise capabilities to recommended training parameters (sets and repetitions) for specific exercises like Barbell Squat and Bench Press, specifically for the goal of building mass and strength.
* **`dietary_needs_data.csv`**: Contains historical data linking user profiles, activity levels (or proxies like exercise capabilities), and the target goal ("building mass and strength") to recommended daily calorie intake and macronutrient targets (protein, carbs, fat).
* **`exercise_database.csv`**: A structured database of various exercises with information about the muscle groups they target, fitness level suitability, equipment required, and recommended sets/reps for different goals (including building mass and strength).

### User Profile Dictionary

For making predictions, the `PredictionEngine` expects a user's profile to be provided as a Python dictionary. This dictionary should contain key-value pairs where the keys are the feature names (corresponding to the columns in your training data) and the values are the user's information. An example of this dictionary is provided in the `main()` function:

```python
        sample_user_profile = {
            "age": 30,
            "gender": "male",
            "weight": 80,  # kg
            "height": 180, # cm
            "exercise_capabilities": {"pushups": 10, "squats": 20, "bench_press_weight": 60},
            "country_of_residence": "Egypt",
            "target_goal": "building mass and strength"
        }
```

**Note:** The keys in this dictionary must match the column names used during model training. The `exercise_capabilities` dictionary within the `sample_user_profile` might need to be flattened into individual features (e.g., "pushups", "squats", "bench_press_weight") depending on how you structured your training data.

## 📤 Output

The `main()` function demonstrates the output of the AI model:

* **Predicted Fitness Level**: A string indicating the predicted fitness level of the user (e.g., "Intermediate").
* **Predicted Training Plan**: A dictionary containing a suggested training plan, including workout frequency and a list of exercises with recommended sets and repetitions. The specific exercises and parameters will depend on the trained model and the user's profile. For example:

    ```
    --- Predicted Training Plan ---
    {'workout_frequency': '3 days per week (example)', 'exercises': [{'name': 'Barbell Squat', 'sets': 3, 'repetitions': 8}, {'name': 'Bench Press', 'sets': 3, 'repetitions': 8}]}
    ```

* **Predicted Dietary Needs**: A dictionary containing the recommended daily calorie intake and macronutrient targets (protein, carbohydrates, and fat) in grams. For example:

    ```
    --- Predicted Dietary Needs ---
    {'daily_calories': 3000, 'macronutrient_targets': {'protein': 150, 'carbs': 300, 'fat': 100}}
    ```

## 🚀 Usage

Follow these steps to set up and run the AI model in Google Colab:

### Prerequisites

* **Google Account:** You will need a Google account to use Google Colab and Google Drive.
* **Data Files:** Ensure you have created the four CSV data files (`fitness_level_data.csv`, `training_params_data.csv`, `dietary_needs_data.csv`, `exercise_database.csv`) and placed them in the `Projects/Get_Fit_App/Ai_Model/Data_Source` folder on your Google Drive.
* **Project Folder Structure:** Make sure you have the folder structure `Projects/Get_Fit_App/Ai_Model` on your Google Drive. The `Generated_Models` folder will be created automatically when you run the code.

### Setup and Execution in Google Colab

1.  **Open the Notebook:** Open a new or existing Google Colab notebook.
2.  **Mount Google Drive:** In the first code cell, paste and run the following code to mount your Google Drive:

    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

    Follow the on-screen instructions to authorize Colab to access your Drive.
3.  **Copy and Paste the Code:** Copy the entire Python code provided in the `ai_model.ipynb` file and paste it into a single code cell in your Colab notebook (you can replace any existing code in that cell).
4.  **Verify Configuration:** Double-check the `Configuration` class in the code to ensure the `DRIVE_ROOT`, `PROJECT_FOLDER`, and `DATA_FOLDER` variables correctly point to the location of your data files on Google Drive.
5.  **Train the Models:** Run the code cell containing the pasted code. The `train_models(config)` function call in the `if __name__ == "__main__":` block will be executed, training the AI models and saving them in the `Generated_Models` folder on your Google Drive. You will see logging messages indicating the progress.
6.  **Run Predictions:** After the training is complete, comment out the `train_models(config)` line and uncomment the `main()` line in the execution block. Run the code cell again. This will load the trained models and print the personalized fitness and nutrition recommendations for the `sample_user_profile`.

Congratulations! You have successfully set up and run the AI-powered personalized fitness and nutrition recommendation engine in Google Colab. You can now experiment with different data and user profiles to see the model in action. Remember to continuously improve your training data for better accuracy and more personalized recommendations.
