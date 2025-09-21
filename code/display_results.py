# display_results.py
import pandas as pd
import numpy as np
import pickle
import os
import time
import warnings
import matplotlib.pyplot as plt
from matplotlib import cm
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer

shap.initjs()

plt.rcParams.update({
    'font.family': 'Times New Roman',
    'font.weight': 'bold',
    'font.size': 18,
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'axes.titlecolor': 'black'
})

warnings.filterwarnings('ignore')


class ModelTrainingPipeline:
    def __init__(self, config_file=None):
        self.optimized_models = {}
        self.results_df = None
        self.config = {
            'data_path': r'D:\Calc_RHEA\dataset.xlsx',
            'save_dir': 'model_results',
            'force_retrain': False,
            'n_iter': 50,
            'generate_visualizations': True
        }

        if config_file and os.path.exists(config_file):
            self.load_config(config_file)

        os.makedirs(self.config['save_dir'], exist_ok=True)

    def load_config(self, config_file):
        try:
            with open(config_file, 'rb') as f:
                loaded_config = pickle.load(f)
            self.config.update(loaded_config)
            print("Configuration file loaded successfully")
        except Exception as e:
            print(f"Profile loading failed: {e}")

    def save_config(self):
        config_path = os.path.join(self.config['save_dir'], 'training_config.pkl')
        with open(config_path, 'wb') as f:
            pickle.dump(self.config, f)
        print(f"Configuration files are saved to the following path: {config_path}")

    def check_existing_results(self):
        results_file = os.path.join(self.config['save_dir'], 'model_results.pkl')
        models_file = os.path.join(self.config['save_dir'], 'optimized_models.pkl')
        return os.path.exists(results_file) and os.path.exists(models_file)

    def load_existing_results(self):
        try:
            results_file = os.path.join(self.config['save_dir'], 'model_results.pkl')
            models_file = os.path.join(self.config['save_dir'], 'optimized_models.pkl')

            with open(results_file, 'rb') as f:
                self.results_df = pickle.load(f)

            with open(models_file, 'rb') as f:
                self.optimized_models = pickle.load(f)

            print("The previously saved model and results have been loaded!")
            return True
        except Exception as e:
            print(f"Failed to load saved results:{e}")
            return False

    def get_model_performance(self):
        if self.results_df is not None:
            return self.results_df
        else:
            print("No performance data is available.")
            return None

    def get_best_model(self, metric='R2_test'):
        if self.results_df is not None:
            if metric == 'R2_test':
                best_model = self.results_df[metric].idxmax()
            else:
                best_model = self.results_df[metric].idxmin()
            return best_model, self.results_df.loc[best_model]
        else:
            print("No performance data is available.")
            return None, None


def display_training_results():


    pipeline = ModelTrainingPipeline()
    pipeline.config['save_dir'] = 'model_results_v2'
    pipeline.config['force_retrain'] = False


    if pipeline.load_existing_results():
        print(" Model results loaded successfully!")
        print("=" * 80)


        performance = pipeline.get_model_performance()
        print("All Model Performance Comparison:")
        print("=" * 80)
        print(performance.round(4))
        print("\n")

        # Display rankings for each metric
        print("Metric Rankings:")
        print("=" * 80)

        r2_ranked = performance['R2_test'].sort_values(ascending=False)
        print("R² Ranking (Higher is better):")
        for i, (model, score) in enumerate(r2_ranked.items(), 1):
            print(f"  {i}. {model}: {score:.4f}")
        print("\n")


        rmse_ranked = performance['RMSE_test'].sort_values(ascending=True)
        print("RMSE Ranking (Lower is better):")
        for i, (model, score) in enumerate(rmse_ranked.items(), 1):
            print(f"  {i}. {model}: {score:.4f}")
        print("\n")

        mae_ranked = performance['MAE_test'].sort_values(ascending=True)
        print("MAE Ranking (Lower is better):")
        for i, (model, score) in enumerate(mae_ranked.items(), 1):
            print(f"  {i}. {model}: {score:.4f}")
        print("\n")


        mre_ranked = performance['MRE_test'].sort_values(ascending=True)
        print("MRE Ranking (Lower is better):")
        for i, (model, score) in enumerate(mre_ranked.items(), 1):
            print(f"  {i}. {model}: {score:.4f}")
        print("\n")

        # Display best models
        print(" Best Model Statistics:")
        print("=" * 80)

        best_r2_model, best_r2_score = pipeline.get_best_model('R2_test')
        best_rmse_model, best_rmse_score = pipeline.get_best_model('RMSE_test')
        best_mae_model, best_mae_score = pipeline.get_best_model('MAE_test')
        best_mre_model, best_mre_score = pipeline.get_best_model('MRE_test')

        print(f"Best R² Model: {best_r2_model} (Score: {best_r2_score['R2_test']:.4f})")
        print(f"Best RMSE Model: {best_rmse_model} (Score: {best_rmse_score['RMSE_test']:.4f})")
        print(f"Best MAE Model: {best_mae_model} (Score: {best_mae_score['MAE_test']:.4f})")
        print(f"Best MRE Model: {best_mre_model} (Score: {best_mre_score['MRE_test']:.4f})")
        print("\n")


        print(" Detailed Performance for Each Model:")
        print("=" * 80)
        for model_name in performance.index:
            model_perf = performance.loc[model_name]
            print(f"{model_name}:")
            print(f"  R²: {model_perf['R2_test']:.4f}")
            print(f"  RMSE: {model_perf['RMSE_test']:.4f}")
            print(f"  MAE: {model_perf['MAE_test']:.4f}")
            print(f"  MRE: {model_perf['MRE_test']:.4f}")
            print("-" * 40)

        return True
    else:
        print(" Failed to load model results. Please check file path or retrain models")
        return False


if __name__ == "__main__":
    display_training_results()
