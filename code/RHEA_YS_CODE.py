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
from sklearn.ensemble import StackingRegressor
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

    def save_results(self):

        try:

            results_file = os.path.join(self.config['save_dir'], 'model_results.pkl')
            with open(results_file, 'wb') as f:
                pickle.dump(self.results_df, f)


            models_file = os.path.join(self.config['save_dir'], 'optimized_models.pkl')
            with open(models_file, 'wb') as f:
                pickle.dump(self.optimized_models, f)

            csv_file = os.path.join(self.config['save_dir'], '../model_performance.csv')
            self.results_df.to_csv(csv_file)


            self.save_config()

            print(f"ALL results have been saved to the directory: {self.config['save_dir']}")
            return True
        except Exception as e:
            print(f"Failed to save results: {e}")
            return False

    def run_training(self):

        start_time = time.time()


        if not self.config['force_retrain'] and self.check_existing_results():
            if self.load_existing_results():
                print("Skip training using saved model results")

                if self.config['generate_visualizations']:
                    self.generate_all_visualizations()
                return True

        print("Initiate the new training process...")

        try:
            # 1. Data Loading and Checking
            file_path = self.config['data_path']
            if not os.path.exists(file_path):
                raise FileNotFoundError("The file path does not exist.Please check the path.")
            data = pd.read_excel(file_path)

            # Check if the target column and features exist
            if 'YS' not in data.columns:
                raise ValueError("The target column 'YS' does not exist, please check the data.")

            # Check for missing values in the data
            if data.isnull().sum().any():
                print("There are missing values in the data, please handle the missing values first.")
                print(data.isnull().sum())
                raise ValueError("The data contains missing values.")

            # 2. Feature and Target Separation
            y = data['YS'].values
            X = data.iloc[:360, 1:24]

            # Dataset partitioning
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # Data standardization
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Save the standardizer for subsequent predictions
            self.scaler = scaler
            self.X_test = X_test
            self.y_test = y_test
            self.feature_names = X.columns.tolist()

            # 3. Model Definition
            models = {
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(n_jobs=-1),
                "AdaBoost": AdaBoostRegressor(),
                "Artificial Network": MLPRegressor(max_iter=2000, early_stopping=True),
                "Xgboost": XGBRegressor(n_jobs=-1, tree_method='hist'),
                "SVM": SVR(kernel='rbf'),
                "GBDT": GradientBoostingRegressor(),
                "CatBoost": CatBoostRegressor(verbose=0, thread_count=-1)
            }

            param_space = {
                "Decision Tree": {
                    'max_depth': Integer(5, 50),
                    'min_samples_split': Integer(2, 20),
                    'min_samples_leaf': Integer(1, 20),
                    'criterion': ['squared_error', 'absolute_error']
                },
                "Random Forest": {
                    'n_estimators': Integer(200, 2000),
                    'max_depth': Integer(10, 50),
                    'min_samples_split': Integer(2, 20),
                    'min_samples_leaf': Integer(1, 20),
                    'max_features': Real(0.5, 1.0)
                },
                "AdaBoost": {
                    'n_estimators': Integer(100, 1000),
                    'learning_rate': Real(0.01, 1.0, prior='log-uniform')
                },
                "Artificial Network": {
                    'hidden_layer_sizes': Integer(100, 500),
                    'activation': ['tanh', 'relu'],
                    'solver': ['adam'],
                    'alpha': Real(1e-6, 1e-2, prior='log-uniform'),
                    'learning_rate_init': Real(1e-4, 1e-2, prior='log-uniform'),
                    'batch_size': Integer(16, 256)
                },
                "GBDT": {
                    'n_estimators': Integer(200, 2000),
                    'learning_rate': Real(0.005, 0.2),
                    'max_depth': Integer(5, 15),
                    'min_samples_split': Integer(2, 20),
                    'min_samples_leaf': Integer(1, 20),
                    'max_features': Real(0.5, 1.0)
                },
                "SVM": {
                    'C': Real(1e3, 1e5, prior='log-uniform'),
                    'epsilon': Real(0.001, 0.1)
                },
                "Xgboost": {
                    'n_estimators': Integer(200, 2000),
                    'max_depth': Integer(5, 15),
                    'learning_rate': Real(0.005, 0.3),
                    'gamma': Real(0, 5),
                    'reg_alpha': Real(0, 1),
                    'reg_lambda': Real(0, 1),
                    'min_child_weight': Integer(1, 20),
                    'subsample': Real(0.6, 1),
                    'colsample_bytree': Real(0.6, 1)
                },
                "CatBoost": {
                    'iterations': Integer(1000, 5000),
                    'depth': Integer(6, 12),
                    'learning_rate': Real(0.005, 0.3),
                    'l2_leaf_reg': Integer(1, 10)
                }
            }

            # 4. Bayesian Parameter Optimization
            optimized_models = {}
            for name, model in models.items():
                print(f"Optimizing the model: {name}...")
                try:
                    if name in param_space:
                        search = BayesSearchCV(
                            model,
                            param_space[name],
                            n_iter=50,
                            cv=5,
                            scoring='r2',
                            n_jobs=-1,
                            random_state=42
                        )
                        search.fit(X_train_scaled, y_train)
                        optimized_models[name] = search.best_estimator_
                        print(f"Optimal parameters: {search.best_params_}")
                        print(f"Optimal R²: {search.best_score_:.4f}")
                    else:
                        model.fit(X_train_scaled, y_train)
                        optimized_models[name] = model
                except Exception as e:
                    print(f"Model {name} optimization failed: {e}")

            # 5. Model Evaluation
            scores = {}
            for name, model in optimized_models.items():
                y_pred_test = model.predict(X_test_scaled)


                mre_test = np.mean(np.abs((y_test - y_pred_test) / np.clip(y_pred_test, 1e-10, None)))

                scores[name] = {
                    "R2_test": r2_score(y_test, y_pred_test),
                    "MSE_test": mean_squared_error(y_test, y_pred_test),
                    "RMSE_test": np.sqrt(mean_squared_error(y_test, y_pred_test)),
                    "MAE_test": mean_absolute_error(y_test, y_pred_test),
                    "MRE_test": mre_test
                }


            self.results_df = pd.DataFrame(scores).T
            self.optimized_models = optimized_models
            self.X_test_scaled = X_test_scaled
            self.y_test = y_test


            self.save_results()

            # 6. Evaluation Metrics Plots
            if self.config['generate_visualizations']:
                self.generate_all_visualizations()

            end_time = time.time()
            training_time = end_time - start_time
            print(f"Training complete !Total time: {training_time:.2f} second")

            return True

        except Exception as e:
            print(f"Errors occurred during training: {e}")
            import traceback
            traceback.print_exc()
            return False

    def generate_all_visualizations(self):

        print("Generate Visualization Charts...")


        if not hasattr(self, 'X_test_scaled') or not hasattr(self, 'y_test'):
            print("Without test data, visualization cannot be generated.")
            return

        try:

            self.generate_performance_bar_charts()


            self.generate_scatter_and_residual_plots()


            self.generate_radar_chart()

            # SHAP Analysis
            self.generate_shap_analysis()

        except Exception as e:
            print(f"An error occurred while generating the visualization: {e}")
            import traceback
            traceback.print_exc()

    def generate_performance_bar_charts(self):

        metrics = ['R2_test', 'RMSE_test', 'MAE_test', 'MRE_test']


        all_models = self.results_df.index.tolist()
        n_models = len(all_models)
        start_color = np.array([0.12, 0.47, 0.71, 1.0])
        end_color = np.array([1.0, 0.5, 0.05, 1.0])
        colors_palette = [start_color + (end_color - start_color) * i / max(1, n_models - 1) for i in range(n_models)]
        model_colors = {model: colors_palette[i] for i, model in enumerate(all_models)}

        for metric in metrics:
            plt.figure(figsize=(16, 8))

            ascending = False if metric == "R2_test" else True
            sorted_models = self.results_df.sort_values(metric, ascending=ascending).index

            colors = [model_colors[model] for model in sorted_models]

            values = self.results_df.loc[sorted_models, metric].values
            max_value = values.max() if metric != "MRE_test" else values.min()
            y_offset = max_value * 0.2 if "R2" in metric else max_value * 0.25

            bars = plt.bar(range(len(sorted_models)), values,
                           color=colors,
                           alpha=0.8,
                           edgecolor='black',
                           width=0.6)


            for bar in bars:
                height = bar.get_height()
                label_pos = height + y_offset if "R2" in metric else height - y_offset * 1.2
                va = 'bottom' if "R2" in metric else 'top'
                plt.text(bar.get_x() + bar.get_width() / 2., label_pos,
                         f'{height:.4f}',
                         ha='center', va=va,
                         rotation=45,
                         fontsize=12,
                         color='black')


            if metric in ["R2_test", "MRE_test"]:
                plt.ylim(0, min(1, max(1, values.max() * 1.1)))


            plt.xticks(range(len(sorted_models)), sorted_models, rotation=40, ha='right')
            plt.ylabel(metric.replace('_test', ''), fontsize=18)
            plt.title(f'Model Performance - {metric.split("_")[0]}', pad=20)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.config['save_dir'], f'enhanced_{metric}_comparison.png'),
                        dpi=300, facecolor='white')
            plt.close()

    def generate_scatter_and_residual_plots(self):

        for name, model in self.optimized_models.items():
            y_pred_test = model.predict(self.X_test_scaled)


            fig = plt.figure(figsize=(16, 14))
            gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1],
                                  hspace=0.3, wspace=0.3)


            ax_main = fig.add_subplot(gs[0, 0])
            scatter = ax_main.scatter(self.y_test, y_pred_test,
                                      s=100,
                                      c=np.abs(self.y_test - y_pred_test),
                                      cmap='coolwarm',
                                      alpha=0.7,
                                      edgecolors='w',
                                      linewidth=1)

            cbar = plt.colorbar(scatter, ax=ax_main)
            cbar.set_label('Absolute Residual', fontsize=12)


            ax_main.plot([self.y_test.min(), self.y_test.max()],
                         [self.y_test.min(), self.y_test.max()],
                         '--', color='red', lw=2)


            z = np.polyfit(self.y_test, y_pred_test, 1)
            p = np.poly1d(z)
            ax_main.plot(self.y_test, p(self.y_test), color='blue', lw=2)

            ax_main.set_xlabel('True Values', fontsize=18)
            ax_main.set_ylabel('Predicted Values', fontsize=18)


            ax_res_dist = fig.add_subplot(gs[0, 1])
            residuals = self.y_test - y_pred_test
            ax_res_dist.hist(residuals, bins=30, orientation='horizontal',
                             color='lightblue', edgecolor='black')
            ax_res_dist.set_title('Residual Distribution', fontsize=12)


            ax_res_trend = fig.add_subplot(gs[1, 0])
            ax_res_trend.scatter(y_pred_test, residuals, s=50, alpha=0.7)
            ax_res_trend.axhline(0, color='red', linestyle='--', lw=2)
            ax_res_trend.set_xlabel('Predicted Values', fontsize=12)
            ax_res_trend.set_ylabel('Residuals', fontsize=12)

            plt.suptitle(f'{name} - Analysis', fontsize=18, y=0.95)
            plt.savefig(os.path.join(self.config['save_dir'], f'enhanced_{name}_scatter.png'),
                        dpi=300, facecolor='white')
            plt.close()

    def generate_radar_chart(self):

        # Model Comparison Radar Chart
        metrics = ['R2_test', 'RMSE_test', 'MAE_test', 'MRE_test']
        labels = ['R²', 'RMSE', 'MAE', 'MRE']

        normalized_data = {
            'R2_test': self.results_df['R2_test'] / self.results_df['R2_test'].max(),
            'RMSE_test': 1 - (self.results_df['RMSE_test'] - self.results_df['RMSE_test'].min()) /
                         (self.results_df['RMSE_test'].max() - self.results_df['RMSE_test'].min()),
            'MAE_test': 1 - (self.results_df['MAE_test'] - self.results_df['MAE_test'].min()) /
                        (self.results_df['MAE_test'].max() - self.results_df['MAE_test'].min()),
            'MRE_test': 1 - (self.results_df['MRE_test'] - self.results_df['MRE_test'].min()) /
                        (self.results_df['MRE_test'].max() - self.results_df['MRE_test'].min())
        }


        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, polar=True)


        for idx, model in enumerate(self.results_df.index):
            values = [
                normalized_data['R2_test'][model],
                normalized_data['RMSE_test'][model],
                normalized_data['MAE_test'][model],
                normalized_data['MRE_test'][model]
            ]
            values += values[:1]
            ax.plot(angles + angles[:1], values, linewidth=2,
                    label=model, linestyle='--' if 'Stacking' in model else '-')
            ax.fill(angles + angles[:1], values, alpha=0.1)


        ax.set_thetagrids(np.degrees(angles), labels)
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        plt.title('Model Performance Radar Chart', y=1.15)
        plt.savefig(os.path.join(self.config['save_dir'], '../model_radar_comparison.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

    def generate_shap_analysis(self):

        shap_results_dir = os.path.join(self.config['save_dir'], "../shap_analysis")
        os.makedirs(shap_results_dir, exist_ok=True)

        for name, model in self.optimized_models.items():
            try:

                if name in ["Xgboost", "CatBoost", "GBDT", "Random Forest", "Decision Tree"]:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(self.X_test_scaled)


                    plt.figure(figsize=(16, 6))
                    shap.summary_plot(shap_values, self.X_test_scaled,
                                      feature_names=self.feature_names,
                                      plot_type="bar",
                                      show=False)
                    plt.title(f"{name}-Feature Importance")
                    plt.tight_layout()
                    plt.savefig(os.path.join(shap_results_dir, f"{name}_shap_importance.png"),
                                dpi=300, facecolor='white')
                    plt.close()

            except Exception as e:
                print(f" SHAP analysis failed{name}: {str(e)}")

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

    def predict_with_model(self, model_name, X_data):

        if model_name in self.optimized_models:

            if hasattr(self, 'scaler'):
                X_data_scaled = self.scaler.transform(X_data)
                return self.optimized_models[model_name].predict(X_data_scaled)
            else:
                return self.optimized_models[model_name].predict(X_data)
        else:
            print(f"Model {model_name} does not exist.")
            return None


def main():
    # Create a Training Pipeline Instance
    pipeline = ModelTrainingPipeline()
    pipeline.config['save_dir'] = 'model_results_v2'
    pipeline.config['force_retrain'] = True
    pipeline.config['n_iter'] = 30
    pipeline.config['generate_visualizations'] = True


    success = pipeline.run_training()

    if success:

        performance = pipeline.get_model_performance()
        print("Model Performance Comparison:")
        print(performance)

        # Obtain the best model
        best_model, best_score = pipeline.get_best_model('R2_test')
        print(f"\nOptimal Model (R²): {best_model}")
        print(f"Best Score: {best_score['R2_test']:.4f}")

        print(f"\nAll results are saved in this path: {pipeline.config['save_dir']}")
        print("Next time you run the settings with `force_retrain=False`, you can directly use the saved results.")


if __name__ == "__main__":
    main()