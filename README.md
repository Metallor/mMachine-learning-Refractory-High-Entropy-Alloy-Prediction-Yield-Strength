# mMachine-learning-Refractory-High-Entropy-Alloy-Prediction-Yield-Strength

## Project Structure

The repository has the following structure:

- `/dataset/` - Contains datasets *F*<sub>1</sub>, *F*<sub>2</sub>, and *F*<sub>3</sub>.
- `/model_results/` - Includes the developed model.
- `/code/` - Python scripts used for model training, hyperparameter optimization, visualization processing, etc.

## How to run the code

Download the dataset, replace the absolute paths in line 43 of `/code/RHEA_YS_CODE.py`.

Modify the range of retrieved data features ( *F*<sub>1</sub>: 1-11; *F*<sub>2</sub>: 11-24; *F*<sub>3</sub>: 1-24 ) in 157 of `/code/RHEA_YS_CODE.py`

The pre-optimization results(`/model_results/model_results.pkl/`) and post-optimization results(`/model_results/optimized_models.pkl/`) can be accessed by running `/code/display-results.py`.

Attention, please replace the absolute paths in line 44 of `/code/display_result.py`.
