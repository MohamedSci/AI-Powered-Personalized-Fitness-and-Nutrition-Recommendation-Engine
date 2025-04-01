# -*- coding: utf-8 -*-
"""AI-Powered Personalized Fitness and Nutrition Recommendation Engine - Production-Ready - TensorFlow

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/13v7fdSScCSdXXsyA7t9wUGvkCMqNdgI7
"""

!pip install tensorflow
!pip install torch
!pip install skl2onnx
!pip install tf2onnx



# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 2.x

import tensorflow as tf
print(tf.__version__)

# -*- coding: utf-8 -*-
"""
AI-Powered Personalized Fitness and Nutrition Recommendation Engine - Production-Ready Code for Google Colab
(Generates Models Suitable for Flutter App via TensorFlow Lite Conversion)

This script implements a comprehensive AI model for generating personalized fitness and
nutrition plans and includes functionality to convert the trained scikit-learn models
to TensorFlow Lite format for use in Flutter applications.

Author: Mohamed Said Ibrahim (Refactored and Enhanced by Gemini)
Date: April 1, 2025
Version: 1.4 (Refactored, Enhanced, and Flutter Ready)
"""

# --- 1. Mount Google Drive ---
from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, r2_score
import joblib
import logging
import os
from typing import Dict, List, Union, Tuple
from dataclasses import dataclass, field
import tensorflow as tf
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# --- 2. Configuration Management ---
@dataclass
class Configuration:
    """Configuration class to manage file paths, hyperparameters, and model settings."""
    DRIVE_ROOT: str = '/content/drive/MyDrive'
    PROJECT_FOLDER: str = 'Projects/Get_Fit_App/Ai_Model'
    DATA_FOLDER: str = 'Data_Source'
    DATA_DIR: str = os.path.join(DRIVE_ROOT, PROJECT_FOLDER, DATA_FOLDER)
    MODEL_DIR: str = os.path.join(DRIVE_ROOT, PROJECT_FOLDER, 'models')  # Folder for original scikit-learn models
    TFLITE_MODEL_DIR: str = os.path.join(DRIVE_ROOT, PROJECT_FOLDER, 'tflite_models')  # Folder for TensorFlow Lite models
    FITNESS_LEVEL_DATA_FILE: str = 'fitness_level_data_example.csv'
    TRAINING_PARAMS_DATA_FILE: str = 'training_params_data_example.csv'
    DIETARY_NEEDS_DATA_FILE: str = 'dietary_needs_data_example.csv'
    EXERCISE_DATABASE_FILE: str = 'exercise_database_example.csv'
    FITNESS_LEVEL_MODEL_NAME: str = 'fitness_level_model.pkl'
    TRAINING_PARAMS_MODEL_NAME: str = 'training_params_model.pkl'
    DIETARY_NEEDS_MODEL_NAME: str = 'dietary_needs_model.pkl'
    N_SPLITS_CV: int = 5
    RANDOM_STATE: int = 42
    LOG_LEVEL: int = logging.INFO
    FITNESS_LEVEL_MODEL_TYPE: str = 'random_forest'  # Options: 'random_forest', 'linear_regression' (for testing)
    TRAINING_PARAMS_MODEL_TYPE: str = 'random_forest' # Options: 'random_forest', 'linear_regression'
    DIETARY_NEEDS_MODEL_TYPE: str = 'linear_regression' # Options: 'linear_regression', 'random_forest'
    TRAINING_PLAN_CONFIG: Dict = field(default_factory=lambda: {
        "workout_frequency": "3 days per week",
        "exercises": [
            {"name": "Barbell Squat", "sets_index": 0, "reps_index": 1},
            {"name": "Bench Press", "sets_index": 2, "reps_index": 3},
            {"name": "Deadlift", "sets_index": 4, "reps_index": 5},
            {"name": "Overhead Press", "sets_index": 6, "reps_index": 7}
            # Add more exercises and their corresponding prediction indices based on your model output
        ]
    })

# Initialize configuration and logging
config = Configuration()
os.makedirs(config.MODEL_DIR, exist_ok=True)
os.makedirs(config.TFLITE_MODEL_DIR, exist_ok=True)

logging.basicConfig(level=config.LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

# --- 3. Data Loading and Preprocessing Module ---
class DataPreprocessor:
    """Handles loading, validation, and preprocessing of data."""
    def load_data(self, file_path: str, expected_columns: List[str] = None) -> pd.DataFrame:
        try:
            df = pd.read_csv(file_path)
            if df.empty:
                logging.warning(f"Loaded data from {file_path} is empty.")
            if expected_columns:
                missing_columns = [col for col in expected_columns if col not in df.columns]
                if missing_columns:
                    logging.error(f"Missing expected columns in {file_path}: {missing_columns}")
                    raise ValueError(f"Missing columns: {missing_columns}")
                logging.info(f"Data loaded from {file_path} with expected columns.")
            else:
                logging.info(f"Data loaded from {file_path}.")
            return df
        except FileNotFoundError:
            logging.error(f"Data file not found at: {file_path}")
            raise
        except Exception as e:
            logging.error(f"Error loading data from {file_path}: {e}")
            raise

    def preprocess_user_data(self, df: pd.DataFrame) -> pd.DataFrame:
        logging.info("Preprocessing user data...")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        for col in df.select_dtypes(include=np.number).columns:
            df[col] = df[col].fillna(df[col].mean())
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].fillna(df[col].mode()[0])
        if 'weight' in df.columns and 'height' in df.columns:
            df['bmi'] = df['weight'] / (df['height'] / 100)**2
        return df

    def split_data(self, df: pd.DataFrame, target_column: Union[str, List[str]], test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        X = df.drop(columns=target_column, errors='ignore')
        y = df[target_column] if isinstance(target_column, str) and target_column in df.columns else df[target_column] if isinstance(target_column, list) and all(col in df.columns for col in target_column) else None
        if y is None:
            raise ValueError(f"Target column(s) '{target_column}' not found in DataFrame.")
        return train_test_split(X, y, test_size=test_size, random_state=config.RANDOM_STATE)

    def create_preprocessing_pipeline(self, X: pd.DataFrame) -> ColumnTransformer:
        numerical_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(include='object').columns.tolist()

        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore')

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)])
        return preprocessor

# --- 4. Model Training Module ---
class ModelTrainer:
    """Trains and evaluates machine learning models."""
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.model = self._initialize_model()

    def _initialize_model(self):
        if self.model_type == 'fitness_level':
            if config.FITNESS_LEVEL_MODEL_TYPE == 'random_forest':
                return RandomForestClassifier(random_state=config.RANDOM_STATE)
            elif config.FITNESS_LEVEL_MODEL_TYPE == 'linear_regression':
                return LinearRegression() # Consider Logistic Regression for classification
            else:
                raise ValueError(f"Unsupported model type for fitness level: {config.FITNESS_LEVEL_MODEL_TYPE}")
        elif self.model_type == 'training_params':
            if config.TRAINING_PARAMS_MODEL_TYPE == 'random_forest':
                return RandomForestRegressor(random_state=config.RANDOM_STATE)
            elif config.TRAINING_PARAMS_MODEL_TYPE == 'linear_regression':
                return LinearRegression()
            else:
                raise ValueError(f"Unsupported model type for training parameters: {config.TRAINING_PARAMS_MODEL_TYPE}")
        elif self.model_type == 'dietary_needs':
            if config.DIETARY_NEEDS_MODEL_TYPE == 'linear_regression':
                return LinearRegression()
            elif config.DIETARY_NEEDS_MODEL_TYPE == 'random_forest':
                return RandomForestRegressor(random_state=config.RANDOM_STATE)
            else:
                raise ValueError(f"Unsupported model type for dietary needs: {config.DIETARY_NEEDS_MODEL_TYPE}")
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")

    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series):
        logging.info(f"Training the {self.model_type} model...")
        self.model.fit(X_train, y_train)
        logging.info(f"{self.model_type} model training complete.")

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series):
        logging.info(f"Evaluating the {self.model_type} model...")
        if self.model_type == 'fitness_level':
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred)
            logging.info(f"{self.model_type} model accuracy: {accuracy:.4f}")
            logging.info(f"{self.model_type} model classification report:\n{report}")
            return accuracy
        elif self.model_type == 'training_params' or self.model_type == 'dietary_needs':
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            logging.info(f"{self.model_type} model mean squared error: {mse:.4f}")
            logging.info(f"{self.model_type} model R-squared: {r2:.4f}")
            return mse
        return None

    def save_model(self, filename: str):
        model_path = os.path.join(config.MODEL_DIR, filename)
        try:
            joblib.dump(self.model, model_path)
            logging.info(f"{self.model_type} model saved to: {model_path}")
        except Exception as e:
            logging.error(f"Error saving {self.model_type} model: {e}")

    def load_model(self, filename: str):
        model_path = os.path.join(config.MODEL_DIR, filename)
        try:
            self.model = joblib.load(model_path)
            logging.info(f"{self.model_type} model loaded from: {model_path}")
        except FileNotFoundError:
            logging.error(f"Model file not found at: {model_path}")
            raise
        except Exception as e:
            logging.error(f"Error loading {self.model_type} model: {e}")

# --- 5. Hyperparameter Tuning Module ---
class HyperparameterTuner:
    """Tunes the hyperparameters of the AI models using GridSearchCV."""
    def __init__(self, model_type: str, model, param_grid: Dict):
        self.model_type = model_type
        self.model = model
        self.param_grid = param_grid
        self.best_model = None

    def tune_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, scoring: str = None, cv: int = config.N_SPLITS_CV):
        logging.info(f"Tuning hyperparameters for the {self.model_type} model...")
        grid_search = GridSearchCV(estimator=self.model, param_grid=self.param_grid, cv=cv, scoring=scoring, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        self.best_model = grid_search.best_estimator_
        logging.info(f"Best hyperparameters for {self.model_type}: {grid_search.best_params_}")

    def get_best_model(self):
        return self.best_model

# --- 6. Prediction Module ---
class PredictionEngine:
    """Handles loading trained models and making predictions."""
    def __init__(self):
        self.fitness_level_model = None
        self.training_params_model = None
        self.dietary_needs_model = None
        self.preprocessor = DataPreprocessor()
        self.feature_encoders = {}  # Store fitted preprocessors
        self.exercise_database = self._load_exercise_database()

    def _load_exercise_database(self) -> Dict:
        """Loads the exercise database from the configured CSV file."""
        exercise_db = {}
        file_path = os.path.join(config.DATA_DIR, config.EXERCISE_DATABASE_FILE)
        try:
            df = pd.read_csv(file_path)
            for index, row in df.iterrows():
                exercise_db[row['name']] = row.drop('name').to_dict()
            logging.info(f"Loaded exercise database from: {file_path}")
        except FileNotFoundError:
            logging.warning(f"Exercise database file not found at: {file_path}. Training plan will have basic info.")
        except Exception as e:
            logging.error(f"Error loading exercise database: {e}")
        return exercise_db

    def load_models(self):
        try:
            trainer_fitness = ModelTrainer(model_type='fitness_level')
            trainer_fitness.load_model(config.FITNESS_LEVEL_MODEL_NAME)
            self.fitness_level_model = trainer_fitness.model

            trainer_training_params = ModelTrainer(model_type='training_params')
            trainer_training_params.load_model(config.TRAINING_PARAMS_MODEL_NAME)
            self.training_params_model = trainer_training_params.model

            trainer_dietary_needs = ModelTrainer(model_type='dietary_needs')
            trainer_dietary_needs.load_model(config.DIETARY_NEEDS_MODEL_NAME)
            self.dietary_needs_model = trainer_dietary_needs.model

            # Load the fitted preprocessors
            self.feature_encoders['fitness_level'] = joblib.load(os.path.join(config.MODEL_DIR, 'fitness_preprocessor.pkl'))
            self.feature_encoders['training_params'] = joblib.load(os.path.join(config.MODEL_DIR, 'training_preprocessor.pkl'))
            self.feature_encoders['dietary_needs'] = joblib.load(os.path.join(config.MODEL_DIR, 'dietary_preprocessor.pkl'))

            logging.info("All models loaded successfully.")

        except FileNotFoundError as e:
            logging.error(f"Error loading models: {e}")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading models: {e}")
            raise

    def predict_fitness_level(self, user_profile: Dict) -> str:
        if self.fitness_level_model is None or 'fitness_level' not in self.feature_encoders:
            logging.error("Fitness level model or preprocessor not loaded.")
            return "Error"
        try:
            user_df = pd.DataFrame([user_profile])
            feature_names = joblib.load(os.path.join(config.MODEL_DIR, 'fitness_feature_names.pkl'))
            common_features = [feature for feature in feature_names if feature in user_df.columns]
            if not common_features:
                logging.warning(f"Missing features in user profile for fitness level prediction: {set(feature_names) - set(user_df.columns)}")
                user_df = user_df.reindex(columns=feature_names, fill_value=0) # Fill missing with 0 (handle with care)
            else:
                user_df = user_df[common_features] # Use only available features

            processed_data = self.feature_encoders['fitness_level'].transform(user_df)
            prediction = self.fitness_level_model.predict(processed_data)[0]
            return prediction
        except Exception as e:
            logging.error(f"Error predicting fitness level: {e}")
            return "Error"

    def predict_training_plan(self, user_profile: Dict) -> Dict:
        if self.training_params_model is None or 'training_params' not in self.feature_encoders:
            logging.error("Training parameters model or preprocessor not loaded.")
            return {"error": "Model not loaded"}
        try:
            user_df = pd.DataFrame([user_profile])
            feature_names = joblib.load(os.path.join(config.MODEL_DIR, 'training_params_feature_names.pkl'))
            common_features = [feature for feature in feature_names if feature in user_df.columns]
            if not common_features:
                logging.warning(f"Missing features in user profile for training plan prediction: {set(feature_names) - set(user_df.columns)}")
                user_df = user_df.reindex(columns=feature_names, fill_value=0) # Fill missing with 0 (handle with care)
            else:
                user_df = user_df[common_features] # Use only available features

            processed_data = self.feature_encoders['training_params'].transform(user_df)
            prediction = self.training_params_model.predict(processed_data)[0].tolist()

            plan = {"workout_frequency": config.TRAINING_PLAN_CONFIG.get("workout_frequency", "3 days per week"), "exercises": []}

            for exercise_config in config.TRAINING_PLAN_CONFIG.get("exercises", []):
                name = exercise_config.get("name")
                sets_index = exercise_config.get("sets_index")
                reps_index = exercise_config.get("reps_index")
                if name is not None and sets_index is not None and reps_index is not None and sets_index < len(prediction) and reps_index < len(prediction):
                    exercise_info = self.exercise_database.get(name, {})
                    plan["exercises"].append({
                        "name": name,
                        "sets": round(prediction[sets_index]),
                        "repetitions": round(prediction[reps_index]),
                        "details": exercise_info.get("description"),
                        "muscle_group": exercise_info.get("muscle_group")
                        # Add more details from exercise_info as needed
                    })
                elif name:
                    logging.warning(f"Could not generate parameters for {name}.")

            return plan
        except Exception as e:
            logging.error(f"Error predicting training plan: {e}")
            return {"error": str(e)}

    def predict_dietary_needs(self, user_profile: Dict) -> Dict:
        if self.dietary_needs_model is None or 'dietary_needs' not in self.feature_encoders:
            logging.error("Dietary needs model or preprocessor not loaded.")
            return {"error": "Model not loaded"}
        try:
            user_df = pd.DataFrame([user_profile])
            feature_names = joblib.load(os.path.join(config.MODEL_DIR, 'dietary_needs_feature_names.pkl'))
            common_features = [feature for feature in feature_names if feature in user_df.columns]
            if not common_features:
                logging.warning(f"Missing features in user profile for dietary needs prediction: {set(feature_names) - set(user_df.columns)}")
                user_df = user_df.reindex(columns=feature_names, fill_value=0) # Fill missing with 0 (handle with care)
            else:
                user_df = user_df[common_features] # Use only available features
            processed_data = self.feature_encoders['dietary_needs'].transform(user_df)
            prediction = self.dietary_needs_model.predict(processed_data)[0].tolist() # Assuming output is [calories, protein, carbs, fat]
            needs = {
                "daily_calories": round(prediction[0]),
                "macronutrient_targets": {
                    "protein": round(prediction[1]),
                    "carbs": round(prediction[2]),
                    "fat": round(prediction[3])
                }
            }
            return needs
        except Exception as e:
            logging.error(f"Error predicting dietary needs: {e}")
            return {"error": str(e)}

# --- 7. Training Function ---
def train_models(config: Configuration):
    """Trains and saves the AI models."""
    preprocessor = DataPreprocessor()

    # --- Hyperparameter Tuning Example (Uncomment and configure to use) ---
    # from sklearn.model_selection import ParameterGrid
    #
    # # Define parameter grid for fitness level model
    # fitness_param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}
    # tuner_fitness = HyperparameterTuner(model_type='fitness_level', model=RandomForestClassifier(random_state=config.RANDOM_STATE), param_grid=fitness_param_grid)
    # # Assuming you have X_train_processed_fitness and y_train_fitness from the data loading and preprocessing step
    # tuner_fitness.tune_hyperparameters(X_train_processed_fitness, y_train_fitness, scoring='accuracy')
    # best_fitness_model = tuner_fitness.get_best_model()
    # if best_fitness_model:
    #     trainer_fitness.model = best_fitness_model
    #     logging.info(f"Using best hyperparameters for fitness level: {tuner_fitness.best_model.get_params()}")

    # --- Train Fitness Level Classification Model ---
    try:
        # expected_cols_fitness = ['age', 'gender', 'weight', 'height', 'pushups', 'squats', 'weight_lifted_squat_max', 'weight_lifted_bench_max', 'activity_level', 'fitness_level'] # Adjust based on your data
        fitness_data = preprocessor.load_data(os.path.join(config.DATA_DIR, config.FITNESS_LEVEL_DATA_FILE)) # Removed expected_cols for example data
        fitness_data = preprocessor.preprocess_user_data(fitness_data)
        X_fitness, y_fitness = fitness_data.drop('fitness_level', axis=1, errors='ignore'), fitness_data['fitness_level']
        X_train_fitness, X_test_fitness, y_train_fitness, y_test_fitness = preprocessor.split_data(fitness_data, 'fitness_level')

        fitness_preprocessor = preprocessor.create_preprocessing_pipeline(X_train_fitness)
        X_train_processed_fitness = fitness_preprocessor.fit_transform(X_train_fitness)
        X_test_processed_fitness = fitness_preprocessor.transform(X_test_fitness)

        trainer_fitness = ModelTrainer(model_type='fitness_level')
        trainer_fitness.train_model(X_train_processed_fitness, y_train_fitness)
        trainer_fitness.evaluate_model(X_test_processed_fitness, y_test_fitness)
        trainer_fitness.save_model(config.FITNESS_LEVEL_MODEL_NAME)
        joblib.dump(fitness_preprocessor, os.path.join(config.MODEL_DIR, 'fitness_preprocessor.pkl'))
        joblib.dump(X_train_fitness.columns.tolist(), os.path.join(config.MODEL_DIR, 'fitness_feature_names.pkl'))

    except FileNotFoundError:
        logging.warning("Fitness level training data not found. Skipping training.")
    except ValueError as ve:
        logging.error(f"Data validation error for fitness level model: {ve}")
    except Exception as e:
        logging.error(f"Error training fitness level model: {e}")

    # --- Train Training Parameters Regression Model ---
    try:
        target_columns_tp = ['squat_sets', 'squat_reps', 'bench_sets', 'bench_reps', 'deadlift_sets', 'deadlift_reps', 'overhead_sets', 'overhead_reps'] # Adjust based on your data
        # expected_cols_tp = ['age', 'gender', 'weight', 'height', 'activity_level'] + target_columns_tp # Add all expected columns
        training_params_data = preprocessor.load_data(os.path.join(config.DATA_DIR, config.TRAINING_PARAMS_DATA_FILE)) # Removed expected_cols for example data
        training_params_data = preprocessor.preprocess_user_data(training_params_data)
        X_training_params, y_training_params = training_params_data.drop(target_columns_tp, axis=1, errors='ignore'), training_params_data[target_columns_tp]
        X_train_tp, X_test_tp, y_train_tp, y_test_tp = preprocessor.split_data(training_params_data, target_columns_tp)

        training_preprocessor = preprocessor.create_preprocessing_pipeline(X_train_tp)
        X_train_processed_tp = training_preprocessor.fit_transform(X_train_tp)
        X_test_processed_tp = training_preprocessor.transform(X_test_tp)

        trainer_training_params = ModelTrainer(model_type='training_params')
        trainer_training_params.train_model(X_train_processed_tp, y_train_tp)
        trainer_training_params.evaluate_model(X_test_processed_tp, y_test_tp)
        trainer_training_params.save_model(config.TRAINING_PARAMS_MODEL_NAME)
        joblib.dump(training_preprocessor, os.path.join(config.MODEL_DIR, 'training_preprocessor.pkl'))
        joblib.dump(X_train_tp.columns.tolist(), os.path.join(config.MODEL_DIR, 'training_params_feature_names.pkl'))

    except FileNotFoundError:
        logging.warning("Training parameters data not found. Skipping training.")
    except ValueError as ve:
        logging.error(f"Data validation error for training parameters model: {ve}")
    except Exception as e:
        logging.error(f"Error training training parameters model: {e}")

    # --- Train Dietary Needs Regression Model ---
    try:
        target_columns_dn = ['calories', 'protein', 'carbs', 'fat']
        # expected_cols_dn = ['age', 'gender', 'weight', 'height', 'activity_level'] + target_columns_dn
        dietary_needs_data = preprocessor.load_data(os.path.join(config.DATA_DIR, config.DIETARY_NEEDS_DATA_FILE)) # Removed expected_cols for example data
        dietary_needs_data = preprocessor.preprocess_user_data(dietary_needs_data)
        X_dietary_needs, y_dietary_needs = dietary_needs_data.drop(target_columns_dn, axis=1, errors='ignore'), dietary_needs_data[target_columns_dn]
        X_train_dn, X_test_dn, y_train_dn, y_test_dn = preprocessor.split_data(dietary_needs_data, target_columns_dn)

        dietary_preprocessor = preprocessor.create_preprocessing_pipeline(X_train_dn)
        X_train_processed_dn = dietary_preprocessor.fit_transform(X_train_dn)
        X_test_processed_dn = dietary_preprocessor.transform(X_test_dn)

        trainer_dietary_needs = ModelTrainer(model_type='dietary_needs')
        trainer_dietary_needs.train_model(X_train_processed_dn, y_train_dn)
        trainer_dietary_needs.evaluate_model(X_test_processed_dn, y_test_dn)
        trainer_dietary_needs.save_model(config.DIETARY_NEEDS_MODEL_NAME)
        joblib.dump(dietary_preprocessor, os.path.join(config.MODEL_DIR, 'dietary_preprocessor.pkl'))
        joblib.dump(X_train_dn.columns.tolist(), os.path.join(config.MODEL_DIR, 'dietary_needs_feature_names.pkl'))

    except FileNotFoundError:
        logging.warning("Dietary needs training data not found. Skipping training.")
    except ValueError as ve:
        logging.error(f"Data validation error for dietary needs model: {ve}")
    except Exception as e:
        logging.error(f"Error training dietary needs model: {e}")

# --- 8. TensorFlow Lite Conversion Function ---
def convert_sklearn_model_to_tflite(model_path: str, output_path: str, input_shape: Tuple):
    """Converts a trained scikit-learn model to TensorFlow Lite format."""
    try:
        # Load the scikit-learn model
        sklearn_model = joblib.load(model_path)

        # Define the input specification for ONNX conversion
        initial_type = [('float_input', FloatTensorType(input_shape))]

        # Convert the scikit-learn model to ONNX format
        onnx_model = convert_sklearn(sklearn_model, initial_types=initial_type)
        onnx_file_path = output_path.replace(".tflite", ".onnx")
        with open(onnx_file_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        logging.info(f"ONNX model saved to: {onnx_file_path}")

        # Convert the ONNX model to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_onnx(onnx_file_path)
        tflite_model = converter.convert()

        # Save the TensorFlow Lite model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        logging.info(f"TensorFlow Lite model saved to: {output_path}")
        os.remove(onnx_file_path) # Clean up the ONNX file

    except FileNotFoundError:
        logging.error(f"Model not found at: {model_path}")
    except Exception as e:
        logging.error(f"Error converting {model_path} to TensorFlow Lite: {e}")

def convert_sklearn_to_tflite(config: Configuration):
    """Converts trained scikit-learn models to TensorFlow Lite format."""
    # Ensure TensorFlow version is compatible with ONNX conversion (>= 2.7 recommended)
    try:
        import tensorflow as tf
        if tf.__version__ < '2.7':
            logging.warning(f"TensorFlow version {tf.__version__} might be older than recommended for ONNX to TFLite conversion. Consider upgrading to 2.7 or higher.")
    except ImportError:
        logging.error("TensorFlow not found.")
        return

    convert_sklearn_model_to_tflite(
        os.path.join(config.MODEL_DIR, config.FITNESS_LEVEL_MODEL_NAME),
        os.path.join(config.TFLITE_MODEL_DIR, 'fitness_level_model.tflite'),
        (None, joblib.load(os.path.join(config.MODEL_DIR, 'fitness_feature_names.pkl')).__len__())
    )
    convert_sklearn_model_to_tflite(
        os.path.join(config.MODEL_DIR, config.TRAINING_PARAMS_MODEL_NAME),
        os.path.join(config.TFLITE_MODEL_DIR, 'training_params_model.tflite'),
        (None, joblib.load(os.path.join(config.MODEL_DIR, 'training_params_feature_names.pkl')).__len__())
    )
    convert_sklearn_model_to_tflite(
        os.path.join(config.MODEL_DIR, config.DIETARY_NEEDS_MODEL_NAME),
        os.path.join(config.TFLITE_MODEL_DIR, 'dietary_needs_model.tflite'),
        (None, joblib.load(os.path.join(config.MODEL_DIR, 'dietary_needs_feature_names.pkl')).__len__())
    )

# --- 9. Main Function to Run Predictions ---
def main():
    """Loads trained models and demonstrates prediction for a sample user."""
    prediction_engine = PredictionEngine()
    preprocessor = DataPreprocessor() # Instantiate DataPreprocessor here
    try:
        prediction_engine.load_models()

        # Example user profile (replace with actual user input)
        sample_user_profile = {
            "age": 30,
            "gender": "male",
            "weight": 80,   # kg
            "height": 180,  # cm
            "pushups": 10,
            "squats": 20,
            "weight_lifted_squat_max": 90.0,
            "weight_lifted_bench_max": 70.0,
            "activity_level": "Moderately Active",
            "target_goal": "strength", # Example for dietary needs and training parameters
            "dietary_preference": "none", # Example for dietary needs
            # Add other features based on your training data. Ensure these match the features used during training.
            "plank_duration": 60,
            "running_endurance": 15,
            "weight_lifted_squat": 100,
        }

        # Preprocess the sample user profile to calculate 'bmi'
        sample_user_profile_df = pd.DataFrame([sample_user_profile])
        processed_user_profile_df = preprocessor.preprocess_user_data(sample_user_profile_df)
        processed_user_profile = processed_user_profile_df.iloc[0].to_dict()

        fitness_level = prediction_engine.predict_fitness_level(processed_user_profile)
        print(f"\nPredicted Fitness Level: {fitness_level}")

        training_plan = prediction_engine.predict_training_plan(processed_user_profile)
        print("\n--- Predicted Training Plan ---")
        print(training_plan)

        dietary_needs = prediction_engine.predict_dietary_needs(processed_user_profile)
        print("\n--- Predicted Dietary Needs ---")
        print(dietary_needs)

    except Exception as e:
        logging.error(f"Error during prediction: {e}")

# --- 10. Conceptual Deployment (Illustrative) ---
def deploy_model_conceptual(config: Configuration):
    """Conceptual model deployment function."""
    logging.info("Conceptual model deployment started...")
    logging.info(f"Trained scikit-learn models are saved in: {config.MODEL_DIR}")
    logging.info(f"TensorFlow Lite models are saved in: {config.TFLITE_MODEL_DIR}")
    logging.info("Conceptual model deployment finished.")

# --- 11. Execution Block ---
if __name__ == "__main__":
    # Ensure TensorFlow version is set (run this if you haven't already)
    # %pip install tensorflow>=2.7
    # import tensorflow as tf
    # print(f"TensorFlow version: {tf.__version__}")

    # To train the models and convert them to TensorFlow Lite, uncomment the following lines:
    train_models(config)
    convert_sklearn_to_tflite(config)

    # To run predictions using the trained scikit-learn models:
    main()

    # To simulate deployment (conceptual):
    deploy_model_conceptual(config)