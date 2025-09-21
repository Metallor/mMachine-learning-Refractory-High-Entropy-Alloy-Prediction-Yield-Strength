import pandas as pd
import numpy as np
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
import warnings
import matplotlib.pyplot as plt
from matplotlib import cm
import shap
from sklearn.ensemble import StackingRegressor
from sklearn.model_selection import cross_val_score
shap.initjs()
import os

warnings.filterwarnings('ignore')

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

# 1. Data Loading and Checking
try:
    file_path = r'F:\TraeProjects\ForChild\dataset.xlsx'
    if not os.path.exists(file_path):
        raise FileNotFoundError("The file path does not exist, please check the path.")
    data = pd.read_excel(file_path)
except Exception as e:
    print(f"Data loading failed: {e}")
    raise

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
X = data.iloc[:360, 1:11]

# Dataset partitioning
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Data standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

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

print("\nBuilding Stacking Fusion Model...")

# Obtain optimized base learners
base_learners = [
    ('xgb', optimized_models['Xgboost']),
    ('GBDT', optimized_models['GBDT'])
]

# Create Stacking Model
stacking_model = StackingRegressor(
    estimators=base_learners,
    final_estimator=RandomForestRegressor(
        n_estimators=500,
        max_depth=10,
        n_jobs=-1,
        random_state=42
    )
)

stacking_model.fit(X_train_scaled, y_train)
optimized_models["Stacking (XGB+GBDT)"] = stacking_model

# 5. Model Evaluation
scores = {}
for name, model in optimized_models.items():
    y_pred_test = model.predict(X_test_scaled)

    mre_test = np.mean(np.abs((y_test - y_pred_test) / y_pred_test))

    scores[name] = {
        "R2_test": r2_score(y_test, y_pred_test),
        "MSE_test": mean_squared_error(y_test, y_pred_test),
        "RMSE_test": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "MAE_test": mean_absolute_error(y_test, y_pred_test),
        "MRE_test": mre_test
    }

results_df = pd.DataFrame(scores).T
results_df.to_csv('model_performance.csv')
print(results_df)


# 6. Evaluation Metrics Plots
all_models = results_df.index.unique().tolist()
n_models = len(all_models)

start_color = np.array([0.12, 0.47, 0.71, 1.0]) 
end_color = np.array([1.0, 0.5, 0.05, 1.0]) 
colors_palette = [start_color + (end_color - start_color) * i / (n_models - 1) for i in range(n_models)]
model_colors = {model: colors_palette[i] for i, model in enumerate(all_models)}
metrics = ['R2_test', 'RMSE_test', 'MAE_test', 'MRE_test']
for metric in metrics:
    plt.figure(figsize=(16, 8))

    ascending = False if metric == "R2_test" else True
    sorted_models = results_df.sort_values(metric, ascending=ascending).index


    colors = [model_colors[model] for model in sorted_models]

    values = results_df.loc[sorted_models, metric].values
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
        plt.ylim(0, 1)

    plt.xticks(range(len(sorted_models)), sorted_models, rotation=40, ha='right')
    plt.ylabel(metric.replace('_test', ''), fontsize=18)
    plt.title(f'Model Performance - {metric.split("_")[0]}', pad=20)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'enhanced_{metric}_comparison.png', dpi=300, facecolor='white')
    plt.show()

# 7. Scatter Plots
special_models = list(optimized_models.keys()) + ["Stacking (XGB+GBDT)"]

for name, model in optimized_models.items():
    y_pred_test = model.predict(X_test_scaled)
    if name not in optimized_models:
        continue

    fig = plt.figure(figsize=(16, 14))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[3, 1],
                          hspace=0.3, wspace=0.3)
    z = np.polyfit(y_test, y_pred_test, 1)
    p = np.poly1d(z)

    ax_main = fig.add_subplot(gs[0, 0])
    scatter = ax_main.scatter(y_test, y_pred_test,
                              s=400,
                              c=np.abs(y_test - y_pred_test),
                              cmap='coolwarm',
                              alpha=0.9,
                              edgecolors='w',
                              linewidth=1,
                              zorder=3)

    cbar = plt.colorbar(scatter, ax=ax_main)
    cbar.set_label('Absolute Residual', fontsize=12, color=colors[1])
    cbar.ax.yaxis.set_tick_params(color=colors[1])
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=colors[1])

    main_line_color = '#2E86C1'
    residual_color = '#E74C3C'

    ax_main.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
                 '--', color=main_line_color, lw=3, zorder=2)

    ax_main.plot(y_test, p(y_test), color=main_line_color, lw=2, zorder=1)

    z = np.polyfit(y_test, y_pred_test, 1)
    p = np.poly1d(z)
    ax_main.plot(y_test, p(y_test), color=colors[0], lw=2, zorder=1)

    ax_main.set_xlabel('True Values', fontsize=18, color=colors[1])
    ax_main.set_ylabel('Predicted Values', fontsize=18, color=colors[1])
    ax_main.tick_params(axis='both', colors=colors[1])

    ax_res_dist = fig.add_subplot(gs[0, 1])
    ax_res_dist.hist(y_test - y_pred_test, bins=30,
                     orientation='horizontal',
                     color='#85C1E9',
                     edgecolor=main_line_color)
    ax_res_dist.set_title('Residual Distribution',
                          fontsize=12, color=colors[1])

    ax_res_trend = fig.add_subplot(gs[1, 0])
    ax_res_trend.scatter(y_pred_test, y_test - y_pred_test,
                         s=300,
                         color=colors[0],
                         edgecolor=colors[1],
                         alpha=0.7)
    ax_res_trend.axhline(0, color=colors[1], linestyle='--', lw=2)
    ax_res_trend.set_xlabel('Predicted Values',
                            fontsize=12, color=colors[1])
    ax_res_trend.set_ylabel('Residuals',
                            fontsize=12, color=colors[1])

    plt.suptitle(f'{name} - Analysis with Enhanced Visualization',
                 fontsize=18, color=colors[1], y=0.95)
    plt.savefig(f'enhanced_{name}_scatter.png', dpi=300,
                facecolor='#f5f5f5')
    plt.show()

# SHAP Analysis
shap_results_dir = "shap_analysis"
os.makedirs(shap_results_dir, exist_ok=True)

for name, model in optimized_models.items():
    try:
        if name == "Stacking (XGB+GBDT)":
            print(f"Analyze the SHAP value of the Stacking model...")
            explainer = shap.KernelExplainer(model.predict, X_train_scaled)
            shap_values = explainer.shap_values(X_test_scaled)

            plt.figure(figsize=(16, 6))
            shap.summary_plot(shap_values, X_test_scaled,
                              feature_names=X.columns,
                              show=False)
            plt.title("Stacking Model SHAP Summary", color='black')
            plt.tight_layout()
            plt.savefig(f"{shap_results_dir}/stacking_shap.png",
                        dpi=300, facecolor='white')
            plt.close()

        elif name in ["Xgboost", "CatBoost", "GBDT", "Random Forest", "Decision Tree"]:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test_scaled)

            plt.figure(figsize=(16, 6))
            plt.subplot(1, 2, 1)
            shap.summary_plot(shap_values, X_test_scaled,
                              feature_names=X.columns,
                              plot_type="bar",
                              show=False)
            plt.title(f"{name}-Feature Importance", color='black')

            plt.subplot(1, 2, 2)
            shap.summary_plot(shap_values, X_test_scaled,
                              feature_names=X.columns,
                              plot_type="dot",
                              show=False)
            plt.title(f"{name}-Feature Effects", color='black')

            plt.tight_layout()
            plt.savefig(f"{shap_results_dir}/combined_{name}_shap.png",
                        dpi=300, facecolor='white')
            plt.close()

    except Exception as e:
        print(f"SHAP analysis failed: {str(e)}")
# Model Comparison Radar Chart
print("\nGenerate model comparison radar image...")

metrics = ['R2_test', 'RMSE_test', 'MAE_test', 'MRE_test']
labels = ['R²', 'RMSE', 'MAE', 'MRE']

max_r2 = results_df['R2_test'].max()
min_rmse = results_df['RMSE_test'].min()
min_mae = results_df['MAE_test'].min()
min_mre = results_df['MRE_test'].min()

normalized_data = {
    'R2_test': results_df['R2_test'] / max_r2,
    'RMSE_test': 1 - (results_df['RMSE_test'] - results_df['RMSE_test'].min()) /
                (results_df['RMSE_test'].max() - results_df['RMSE_test'].min()),
    'MAE_test': 1 - (results_df['MAE_test'] - results_df['MAE_test'].min()) /
                (results_df['MAE_test'].max() - results_df['MAE_test'].min()),
    'MRE_test': 1 - (results_df['MRE_test'] - results_df['MRE_test'].min()) /
                (results_df['MRE_test'].max() - results_df['MRE_test'].min())
}

angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, polar=True)

for idx, (model, row) in enumerate(results_df.iterrows()):
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
ax.set_theta_offset(np.pi/2)
ax.set_theta_direction(-1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
plt.title('Model Performance Radar Chart', y=1.15)
plt.savefig('model_radar_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nAll model evaluations completed!")