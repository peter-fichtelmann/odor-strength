#!/usr/bin/env python
# coding: utf-8

# # Odor Strength Prediction: Model Training, Validation & Insights
# 
# This notebook provides a the training, hyperparameter optimization, validation and interpretation of machine learning models for predicting odor strength from molecular structures. It includes:
# 
# 1. **Load Dataset**: Examination of the curated odor strength dataset
# 2. **Model Training**: Hyperparameter optimization for various encoder-predictor combinations
# 3. **Performance Evaluation**: Comparison of direct (directly predicting odor strength) vs. indirect (first predicting presence of odor, then strength for odorous molecules) prediction approaches and several combinations of different molecular representations and predictive algorithms
# 4. **Feature Importance**: SHAP analysis to understand model predictions
# 5. **External Validation**: Testing on independent Keller 2016 dataset

# In[ ]:


# In[ ]:


# tested with python==3.12.2


# In[ ]:


import pandas as pd


# ## Load Dataset

# Run the notebook dataset_curation.ipynb or python script dataset_curation.py first

# In[ ]:


df_odor_strength = pd.read_csv('data/df_odor_strength.csv')
groups = pd.read_csv('data/odor_strength_groups.csv', index_col=0).values.flatten()


# ### Dataset Analysis

# #### see additional notebook dataset_analysis.ipynb

# ## Model Training with Hyperparameter optimization

# In[ ]:


from typing import Callable, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from optuna.study import MaxTrialsCallback
from optuna.trial import TrialState
import os


from models.odor_strength_module import OdorStrengthModule, OdorStrengthModuleHyperparameterOptimizationWrapper
from models.predictors import Average, LogisticRegressionPredictor, RandomForestPredictor, XGBoostPredictor, MLP_Predictor, ChemPropMPNNPredictor, ChemeleonPredictor, CoralPredictor
from models.molecule_encoder import NativeEncoder, MorganFp, RDKitFp, TopologicalTorsionFp, AtomPairFp, MACCSKeysFp, RDKitDescriptors, ChemBerta

from hyperparameter_optimization.hyperparameter_optimizer import HyperparameterOptimizer
from hyperparameter_optimization.hyperparameter_spaces import HyperparameterSpaces
from utility.metrics import Metrics
from utility.stratified_group_split import stratified_group_train_test_split

hyperparameter_spaces = HyperparameterSpaces()

# STORAGE_FOLDER = 'sqlite:///hyperparameter_optimization/hp_opt_dbs/'
from pathlib import Path

BASE_DIR = Path.cwd()  # or set to your project root explicitly
STORAGE_DIR = BASE_DIR / "hyperparameter_optimization" / "hp_opt_dbs"
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

STORAGE_FOLDER = f"sqlite:///{STORAGE_DIR.as_posix()}/"
print('Storage folder for loading studies:', STORAGE_FOLDER)
# if not os.path.exists('hyperparameter_optimization/hp_opt_dbs/'):
#     os.makedirs('hyperparameter_optimization/hp_opt_dbs/')


# In[ ]:


X = df_odor_strength['canonical_smiles'].values
y = df_odor_strength['numerical_strength'].values
X_train, _, y_train, _, groups_train, _ = stratified_group_train_test_split(X, y, groups, random_state=42)
X_train = X_train.tolist()


# ### Example Model Cross Validation

# In[ ]:


odor_strength_module = OdorStrengthModule(
    molecule_encoder=MorganFp(),
    odor_strength_predictor=RandomForestPredictor()
)

odor_strength_module.evaluate_kFold(X_train, y_train, groups=groups_train, n_splits=5, metric=Metrics().calculate_mse_macro, plot=True, show_wrong_pred=False)


# ### Hyperparameter Optimization

# In[ ]:


N_REPEATS = 10
N_FOLDS = 10
N_TRIALS = 100
LIMIT = 100

callback = MaxTrialsCallback(
    n_trials=LIMIT,
    states=(TrialState.COMPLETE, TrialState.PRUNED),
)

class OdorStrengthHyperparameterOptimizer:
    """
    A hyperparameter optimizer specifically designed for odor strength prediction models.
    
    This class provides a structured approach to hyperparameter optimization for combinations
    of molecular encoders and odor strength predictors using Optuna optimization framework.
    
    Attributes:
        encoder (object): Molecular encoder class to use for molecule representation
        predictor (object): Predictor class to use for odor strength prediction
        n_trials (int): Number of optimization trials to run
        evaluation_metric (Callable): Metric function to optimize
        groups (np.ndarray): Group labels for stratified cross-validation
        n_cv_splits (int): Number of cross-validation splits
        n_cv_repeats (int): Number of cross-validation repeats
        load_if_exists (bool): Whether to load existing study if available
        predictor_hyperparameter_space_name_prefix (str): Prefix for hyperparameter space function names
    """
    
    def __init__(
            self,
            encoder: object,
            predictor: object,
            n_trials: int,
            evaluation_metric: Callable[[np.ndarray, np.ndarray], float],
            groups: np.ndarray = groups,
            n_cv_splits: int = 5,
            n_cv_repeats: int = 5,
            load_if_exists: bool = True,
            predictor_hyperparameter_space_name_prefix: str = 'hyperparameter_space_'
            ) -> None:
        """
        Initialize the hyperparameter optimizer.
        
        Args:
            encoder: Molecular encoder class for molecular representation
            predictor: Predictor class for odor strength prediction
            n_trials: Number of optimization trials to perform
            evaluation_metric: Function to evaluate model performance
            groups: Group labels for stratified cross-validation splits
            n_cv_splits: Number of cross-validation folds (default: 5)
            n_cv_repeats: Number of cross-validation repetitions (default: 5)
            load_if_exists: Whether to resume existing optimization study (default: True)
            predictor_hyperparameter_space_name_prefix: Prefix for hyperparameter space function names
        """
        self.encoder = encoder
        self.predictor = predictor
        self.n_trials = n_trials
        self.load_if_exists = load_if_exists
        self.evaluation_metric = evaluation_metric
        self.groups = groups
        self.n_cv_splits = n_cv_splits
        self.n_cv_repeats = n_cv_repeats
        self.predictor_hyperparameter_space_name_prefix = predictor_hyperparameter_space_name_prefix

    def evaluation_function(self, odor_strength_module: object, X: list[str], y: np.ndarray) -> tuple[float, dict]:
        """
        Evaluate the performance of an odor strength module using cross-validation.
        
        Args:
            odor_strength_module: Configured odor strength prediction module
            X: List of molecular SMILES strings
            y: Target values (numeric odor strength labels)
            
        Returns:
            tuple: Mean performance score and detailed evaluation results
        """
        evaluation_result = odor_strength_module.evaluate_kFold(X, y, metric=self.evaluation_metric, n_splits=self.n_cv_splits, n_repeats=self.n_cv_repeats, real_time_evaluation=True, groups=self.groups)
        return evaluation_result[0], evaluation_result[3]


    def hyperparameter_space(self, trial: optuna.trial.Trial) -> dict[str, Any]:
        """
        Define the hyperparameter search space for encoder-predictor combinations.
        
        Args:
            trial: Optuna trial object for hyperparameter sampling
            
        Returns:
            dict: Dictionary containing encoder and predictor hyperparameters
        """
        hyperparameters = {
            'encoder_name': self.encoder.__name__,
            'predictor_name': self.predictor.__name__,
        }
        encoder_function = getattr(hyperparameter_spaces, 'hyperparameter_space_' + self.encoder.__name__)
        hyperparameters['hp_molecule_encoder'] = encoder_function(trial)
        predictor_function = getattr(hyperparameter_spaces, self.predictor_hyperparameter_space_name_prefix + self.predictor.__name__)
        hyperparameters['hp_odor_strength_predictor'] = predictor_function(trial)
        return hyperparameters

    def optimize_study(
            self,
            X: list[str],
            y: np.ndarray,
            direction: str,
            study_name: str,
            storage_path: str,
            pruner: optuna.pruners.BasePruner,
            pruner_tolerance: float,
            n_repeats: int,
            **optimize_kwargs
            ) -> optuna.Study:
        """
        Run hyperparameter optimization study.
        
        Args:
            X: List of molecular SMILES strings for training
            y: Target values for training
            direction: Optimization direction ('minimize' or 'maximize')
            study_name: Name for the optimization study
            storage_path: Path to store optimization results
            pruner: Optuna pruner for early stopping of unpromising trials
            pruner_tolerance: Tolerance threshold for not pruning trials to the best-performing trial
            n_repeats: Number of evaluation repeats for robustness
            **optimize_kwargs: Additional arguments for optimization
            
        Returns:
            optuna.Study: Completed optimization study object
        """
        study = optuna.create_study(
            direction=direction,
            study_name=study_name,
            storage=storage_path,
            load_if_exists=self.load_if_exists,
            pruner=pruner
        )
        hp_opt = HyperparameterOptimizer(OdorStrengthModuleHyperparameterOptimizationWrapper, study)
        hp_opt.optimize(
            X,
            y,
            hyperparameter_space=self.hyperparameter_space,
            evaluation_function=self.evaluation_function,
            n_trials=self.n_trials,
            n_repeats=n_repeats,
            pruner_tolerance=pruner_tolerance,
            **optimize_kwargs
        )
        return study

encoders = [
    NativeEncoder,
    RDKitDescriptors,
    MorganFp,
    RDKitFp,
    TopologicalTorsionFp,
    AtomPairFp,
    MACCSKeysFp,
    ChemBerta
]
predictors = [
    RandomForestPredictor,
    XGBoostPredictor,
    MLP_Predictor,
    LogisticRegressionPredictor,
    CoralPredictor,
    ChemPropMPNNPredictor,
    ChemeleonPredictor,
    ]


# ### Hyperparameter Optimization Functions
# 
# Models for the following steps were optimized:
# 
# 1. **Direct Approach**: Directly predict odor strength values (0-3 scale)
# 2. **Indirect First Step**: Predict odor presence/absence (binary classification)
# 3. **Indirect Second Step**: Predict strength for odorous molecules only (regression on subset (1-3 scale))

# In[ ]:


def compatibility_check(encoder_name: str, predictor_name: str, w_regression: bool = True) -> bool:
    """
    Check compatibility between molecular encoders and predictors.
    
    Some predictors require specific types of input features. This function ensures
    that incompatible combinations are skipped during optimization.
    
    Args:
        encoder_name (str): Name of the molecular encoder
        predictor_name (str): Name of the predictor model
        w_regression (bool): Whether regression predictors should be included
        
    Returns:
        bool: True if the combination is compatible, False otherwise
    """
    check_list = ['Average', 'ChemPropMPNNPredictor', 'ChemeleonPredictor']
    check_1 = (encoder_name == 'NativeEncoder' and predictor_name not in check_list)
    check_2 = (encoder_name != 'NativeEncoder' and predictor_name in check_list)
    if  check_1 or check_2:
        print(f'Skipping {encoder_name} with {predictor_name} due to incompatibility.')
        return False
    elif not w_regression and predictor_name == 'CoralPredictor':
        return False
    return True

def optimize_hyperparameters_direct_approach(encoder: object, predictor: object) -> None:
    """
    Optimize hyperparameters using direct odor strength prediction approach.
    
    The encoder-predictor models directly predicts odor strength values (0-3 scale) from SMILES strings.
    
    Args:
        encoder (object): Molecular encoder class for molecular representation
        predictor (object): Predictor class for odor strength prediction
    """
    X = df_odor_strength['canonical_smiles'].values
    y = df_odor_strength['numerical_strength'].values
    X_train, _, y_train, _, groups_train, _ = stratified_group_train_test_split(X, y, groups, random_state=42)
    X_train = X_train.tolist()
    try:
        encoder_name = encoder.__name__
        predictor_name = predictor.__name__
        if compatibility_check(encoder_name, predictor_name, w_regression=True):
            print(f'Optimizing hyperparameters: {encoder_name} with {predictor_name}')
            OdorStrengthHyperparameterOptimizer(
                encoder,
                predictor,
                n_trials=N_TRIALS,
                evaluation_metric=Metrics().calculate_mse_macro,
                groups=groups_train,
                n_cv_splits=N_FOLDS,
                n_cv_repeats=1,
                load_if_exists=True,
                predictor_hyperparameter_space_name_prefix='hyperparameter_space_'
                ).optimize_study(
                    X=X_train,
                    y=y_train,
                    direction='minimize',
                    study_name=f'{encoder.__name__}_{predictor.__name__}_direct',
                    storage_path=STORAGE_FOLDER + f'{encoder.__name__}_{predictor.__name__}_direct.db',
                    pruner=optuna.pruners.PercentilePruner(25, n_startup_trials=3),
                    pruner_tolerance=0.02,
                    n_repeats=N_REPEATS,
                    callbacks=[callback]
                    )
    except Exception as e:
        print(f'Error with {encoder_name} and {predictor_name}. Error {e} Skipping...')


def optimize_hyperparameters_indirect_approach_first_step(encoder: object, predictor: object) -> None:
    """
    Optimize hyperparameters for the first step of indirect approach: odor detection.
    
    This function optimizes encoder-predictor models for binary classification to predict whether
    a molecule has any detectable odor (has_odor: 0 or 1).
    
    Args:
        encoder (object): Molecular encoder class for molecular representation
        predictor (object): Binary classifier for odor detection
    """
    X = df_odor_strength['canonical_smiles'].values
    y = df_odor_strength['has_odor'].values
    stratify_data = df_odor_strength['numerical_strength'].values
    X_train, _, y_train, _, groups_train, _ = stratified_group_train_test_split(X, y, groups, random_state=42, stratify_data=stratify_data)
    X_train = X_train.tolist()
    try:
        encoder_name = encoder.__name__
        predictor_name = predictor.__name__
        if compatibility_check(encoder_name, predictor_name, w_regression=False):
            print(f'Optimizing hyperparameters: {encoder_name} with {predictor_name}')
            OdorStrengthHyperparameterOptimizer(
                encoder,
                predictor,
                n_trials=N_TRIALS,
                evaluation_metric=Metrics().calculate_f1_score,
                groups=groups_train,
                n_cv_splits=N_FOLDS,
                n_cv_repeats=1,
                load_if_exists=True,
                predictor_hyperparameter_space_name_prefix='hyperparameter_space_binary_'
                ).optimize_study(
                    X=X_train,
                    y=y_train,
                    direction='maximize',
                    study_name=f'{encoder.__name__}_{predictor.__name__}_indirect_1',
                    storage_path=STORAGE_FOLDER + f'{encoder.__name__}_{predictor.__name__}_indirect_1.db',
                    pruner=optuna.pruners.PercentilePruner(25, n_startup_trials=3),
                    pruner_tolerance=0.015,
                    n_repeats=N_REPEATS,
                    callbacks=[callback]
                    )
    except Exception as e:
        print(f'Error with {encoder_name} and {predictor_name}. Error {e} Skipping...')


def optimize_hyperparameters_indirect_approach_second_step(encoder: object, predictor: object) -> None:
    """
    Optimize hyperparameters for the second step of indirect approach: odor strength prediction.
    
    This function optimizes encoder-predictor models for regression to predict odor strength (1-3 scale)
    for molecules that are odorous.
    
    Args:
        encoder (object): Molecular encoder class for molecular representation
        predictor (object): Regression model for odor strength prediction
    """
    X = df_odor_strength['canonical_smiles'].values
    y = df_odor_strength['numerical_strength'].values
    X_train, _, y_train, _, groups_train, _ = stratified_group_train_test_split(X, y, groups, random_state=42)
    X_train = X_train[y_train >= 1]
    groups_train = groups_train[y_train >= 1]
    y_train = y_train[y_train >= 1]
    X_train = X_train.tolist()
    try:
        encoder_name = encoder.__name__
        predictor_name = predictor.__name__
        if compatibility_check(encoder_name, predictor_name, w_regression=True):
            print(f'Optimizing hyperparameters: {encoder_name} with {predictor_name}')
            OdorStrengthHyperparameterOptimizer(
                encoder,
                predictor,
                n_trials=N_TRIALS,
                evaluation_metric=Metrics().calculate_mse_macro,
                groups=groups_train,
                n_cv_splits=N_FOLDS,
                n_cv_repeats=1,
                load_if_exists=True,
                predictor_hyperparameter_space_name_prefix='hyperparameter_space_'
                ).optimize_study(
                    X=X_train,
                    y=y_train,
                    direction='minimize',
                    study_name=f'{encoder.__name__}_{predictor.__name__}_indirect_2',
                    storage_path=STORAGE_FOLDER + f'{encoder.__name__}_{predictor.__name__}_indirect_2.db',
                    pruner=optuna.pruners.PercentilePruner(25, n_startup_trials=3),
                    pruner_tolerance=0.02,
                    n_repeats=N_REPEATS,
                    callbacks=[callback]
                    )
    except Exception as e:
        print(f'Error with {encoder_name} and {predictor_name}. Error {e} Skipping...')
              
for encoder in encoders:
    for predictor in predictors:
        optimize_hyperparameters_direct_approach(encoder, predictor)
        optimize_hyperparameters_indirect_approach_first_step(encoder, predictor)
        optimize_hyperparameters_indirect_approach_second_step(encoder, predictor)
print('Hyperparameter optimization completed for all combinations of encoders and predictors.')


# ## Model Evaluation

# In[ ]:


from typing import Tuple
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import optuna

from utility.colors import okabe_ito
from models.predictors import Average, LogisticRegressionPredictor, RandomForestPredictor, XGBoostPredictor, MLP_Predictor, ChemPropMPNNPredictor, ChemeleonPredictor, CoralPredictor
from models.molecule_encoder import NativeEncoder, MorganFp, RDKitFp, TopologicalTorsionFp, AtomPairFp, MACCSKeysFp, RDKitDescriptors, ChemBerta
from hyperparameter_optimization.hyperparameter_optimizer import HyperparameterOptimizer
from models.odor_strength_module import OdorStrengthModule, OdorStrengthModuleHyperparameterOptimizationWrapper


DPI = 600
FONTSIZE = 8
LABELSIZE = 7
LABELPAD = 4
FIGURE_WIDTH = 8.3 / 2.54

encoders = [
    NativeEncoder,
    RDKitDescriptors,
    MorganFp,
    RDKitFp,
    TopologicalTorsionFp,
    AtomPairFp,
    MACCSKeysFp,
    ChemBerta
]
predictors = [
    RandomForestPredictor,
    XGBoostPredictor,
    MLP_Predictor,
    LogisticRegressionPredictor,
    CoralPredictor,
    ChemPropMPNNPredictor,
    ChemeleonPredictor,
    ]

X = df_odor_strength['canonical_smiles'].values
y = df_odor_strength['numerical_strength'].values
X_train, X_test, y_train, y_test, groups_train, groups_test = stratified_group_train_test_split(X, y, groups, random_state=42)
X_train = X_train.tolist()
X_test = X_test.tolist()


# ### Performance Evaluation Functions
# 
# This section contains functions for evaluating the best-performing models from hyperparameter optimization and generating performance visualizations.

# In[ ]:


def predict_on_test_set(
        model_class: OdorStrengthModuleHyperparameterOptimizationWrapper,
        hyperparameters: dict[str, Any],
        X_train: list[str],
        y_train: np.ndarray,
        X_test: list[str],
        y_test: np.ndarray,
        repeats: int = 10
        ) -> pd.DataFrame:
    """
    Generate predictions on the test set with multiple random initializations.
    
    This function trains the same model multiple times with different random seeds
    to assess prediction robustness and provide uncertainty estimates.
    
    Args:
        model_class: Class of the model to instantiate
        hyperparameters (dict): Hyperparameters for model initialization
        X_train (list): Training molecular SMILES strings
        y_train (np.ndarray): Training target values
        X_test (list): Test molecular SMILES strings
        y_test (np.ndarray): Test target values
        repeats (int): Number of training/prediction repeats (default: 10)
        
    Returns:
        pd.DataFrame: DataFrame with repeated predictions and corresponding true values
    """
    preds_list = []
    for i in range(repeats):
        model = model_class(**hyperparameters)
        # print('model initiated for prediction run', i+1)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        preds_list.append(preds)
    preds = np.hstack(preds_list)
    y_test_stacked = np.hstack([y_test]*repeats)
    df = pd.DataFrame({'y_test': y_test_stacked, 'preds': preds})
    return df


def get_best_model_performances(
        encoders: list[object],
        predictors: list[object],
        study_name_suffix: str,
        storage_path_suffix: str,
        direction: str,
        metric: Callable[[np.ndarray, np.ndarray], float],
        X_train: list[str],
        y_train: np.ndarray,
        X_test: list[str],
        y_test: np.ndarray,
        repeats: int = 10,
        save_path_addition: str = '',
        additional_test_set: tuple[list[str], np.ndarray] | None = None
        ) -> tuple[pd.DataFrame, type, dict[str, Any]]:
    """
    Evaluate the best models from hyperparameter optimization studies.
    
    This function loads optimized hyperparameter configurations from storage, evaluates them on test data,
    and creates a performance matrix comparing all encoder-predictor combinations.
    
    Args:
        encoders (list): List of molecular encoder classes
        predictors (list): List of predictor classes
        study_name_suffix (str): Suffix for study names to load
        storage_path_suffix (str): Suffix for storage paths to load
        direction (str): Optimization direction ('minimize' or 'maximize')
        metric (callable): Evaluation metric function
        X_train (list): Training molecular SMILES strings
        y_train (np.ndarray): Training target values
        X_test (list): Test molecular SMILES strings  
        y_test (np.ndarray): Test target values
        repeats (int): Number of evaluation repeats (default: 10)
        save_path_addition (str): Additional path component for saving predictions
        additional_test_set (tuple, optional): Additional test set (X, y) for evaluation
        
    Returns:
        tuple: Performance DataFrame, best model class, best hyperparameters
    """
    encoder_name_list = [encoder.__name__ for encoder in encoders]
    predictor_name_list = [predictor.__name__ for predictor in predictors]
    df = pd.DataFrame(columns=encoder_name_list, index=predictor_name_list)
    models = {}
    hyperparameters = {}
    print('Storage folder for loading studies:', STORAGE_FOLDER)
    for encoder in encoders:
        for predictor in predictors:
            study_name=f'{encoder.__name__}_{predictor.__name__}_' + study_name_suffix
            storage_path=f'{STORAGE_FOLDER}{encoder.__name__}_{predictor.__name__}_' + storage_path_suffix + '.db'
            try:
                if not os.path.exists(storage_path.replace('sqlite:///', '')):
                    storage_path = optuna.storages.JournalStorage(
                        optuna.storages.journal.JournalFileBackend(
                            STORAGE_FOLDER.replace('sqlite:///', '') + f'{encoder.__name__}_{predictor.__name__}_' + storage_path_suffix + '.log'
                        )
                    )   
                study = optuna.load_study(
                    study_name=study_name,
                    storage=storage_path
                )
                complete_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
                print(f'{storage_path_suffix} {encoder.__name__} {predictor.__name__} Number of complete trials: {len(complete_trials)}')
                hp_opt = HyperparameterOptimizer(OdorStrengthModuleHyperparameterOptimizationWrapper, study)
                if encoder.__name__ not in models:
                    models[encoder.__name__] = {}
                models[encoder.__name__][predictor.__name__] = hp_opt.model
                if encoder.__name__ not in hyperparameters:
                    hyperparameters[encoder.__name__] = {}
                hyperparameters[encoder.__name__][predictor.__name__] = hp_opt.get_best_hyperparameters()
                if not os.path.exists('test_predictions/'):
                    os.makedirs('test_predictions/')
                if not os.path.exists('test_predictions/' + save_path_addition + encoder.__name__ + predictor.__name__ + '_predictions.csv'):
                    print('path not found, predicting on test set:', 'test_predictions/' + encoder.__name__ + predictor.__name__ + '_predictions.csv')
                    df_pred = predict_on_test_set(hp_opt.model, hp_opt.get_best_hyperparameters(), X_train, y_train, X_test, y_test, repeats=repeats)
                    df_pred.to_csv('test_predictions/' + save_path_addition + encoder.__name__ + predictor.__name__ + '_predictions.csv')
                    print('saved successfully')
                    if additional_test_set is not None:
                        X_test_additional, y_test_additional = additional_test_set
                        df_pred_additional = predict_on_test_set(hp_opt.model, hp_opt.get_best_hyperparameters(), X_train, y_train, X_test_additional, y_test_additional, repeats=repeats)
                        df_pred_additional.to_csv('test_predictions/' + save_path_addition + encoder.__name__ + predictor.__name__ + '_predictions_additional.csv')
                        print('saved additional test set successfully')
                else:
                    df_pred = pd.read_csv('test_predictions/' + save_path_addition + encoder.__name__ + predictor.__name__ + '_predictions.csv', index_col=0)
                best_value_runs = []
                for i in range(repeats):
                    df_pred_subset = df_pred.iloc[i*len(y_test):(i+1)*len(y_test)]
                    best_values = metric(df_pred_subset['y_test'].values, df_pred_subset['preds'].values)
                    if isinstance(best_values, (list, tuple)):
                        best_value = best_values[0]
                    else:
                        best_value = best_values
                    best_value_runs.append(best_value)
                best_value = np.mean(best_value_runs)
                print('Standard deviation over runs for ' + encoder.__name__ + predictor.__name__, np.std(best_value_runs))
                print(f'Best value for {study_name}: {best_value}')
                df.loc[predictor.__name__, encoder.__name__] = best_value
            except Exception as e:
                print(f"Error loading study {study_name}: {e}")
                best_value = None
    df = df.astype(float)
    df[df<0] = np.nan
    if direction == 'maximize':
        best_position = np.unravel_index(df.fillna(-np.inf).values.argmax(), df.shape)
    elif direction == 'minimize':
        best_position = np.unravel_index(df.fillna(np.inf).values.argmin(), df.shape)
    else:
        raise ValueError("Direction must be 'maximize' or 'minimize'")
    row_label = df.index[best_position[0]]
    col_label = df.columns[best_position[1]]
    best_model = models[col_label][row_label]
    best_hyperparameters = hyperparameters[col_label][row_label]
    return df, best_model, best_hyperparameters, hyperparameters, models


encoders = [
    NativeEncoder,
    RDKitDescriptors,
    MorganFp,
    RDKitFp,
    TopologicalTorsionFp,
    AtomPairFp,
    MACCSKeysFp,
    ChemBerta
]
predictors = [
    RandomForestPredictor,
    XGBoostPredictor,
    MLP_Predictor,
    LogisticRegressionPredictor,
    CoralPredictor,
    ChemPropMPNNPredictor,
    ChemeleonPredictor,
    ]


y_train_has_odor = y_train.copy()
y_train_has_odor[y_train_has_odor>0] = 1
y_test_has_odor = y_test.copy()
y_test_has_odor[y_test_has_odor>0] = 1
X_train_wo_odorless = np.array(X_train)[np.array(y_train)>0].tolist()
y_train_wo_odorless = y_train.copy()[np.array(y_train)>0]
X_test_wo_odorless = np.array(X_test)[np.array(y_test)>0].tolist()
y_test_wo_odorless = y_test.copy()[np.array(y_test)>0]

df_direct, best_model_direct_class, best_hyperparameters_direct, hyperparameters_direct, models_direct = get_best_model_performances(
    encoders,
    predictors,
    # 'direct',
    'odor_strength_w_odorless',
    'odor_strength_w_odorless',
    # 'direct',
    direction='minimize',
    metric=Metrics().calculate_mse_macro,
    X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
    save_path_addition='direct',
    repeats=N_REPEATS
    )
df_indirect_1, best_model_indirect_first_step_class, best_hyperparameters_indirect_first_step, hyperparameters_first_step, models_first_step = get_best_model_performances(
    encoders,
    predictors,
    # 'indirect_1',
    'has_odor',
    'has_odor',
    # 'indirect_1',
    direction='maximize',
    metric=Metrics().calculate_f1_score,
    X_train=X_train, y_train=y_train_has_odor, X_test=X_test, y_test=y_test_has_odor,
    save_path_addition='indirect_1',
    repeats=N_REPEATS
    )
df_indirect_2, best_model_indirect_second_step_class, best_hyperparameters_indirect_second_step, hyperparameters_second_step, models_second_step = get_best_model_performances(
    encoders,
    predictors, 
    # 'indirect_2',
    'odor_strength_wo_odorless',
    'odor_strength_wo_odorless',
    # 'indirect_2',
     direction='minimize',
     metric=Metrics().calculate_mse_macro,
     X_train=X_train_wo_odorless, y_train=y_train_wo_odorless, X_test=X_test_wo_odorless, y_test=y_test_wo_odorless,
     save_path_addition='indirect_2',
     additional_test_set=(X_test, y_test),
     repeats=N_REPEATS
     )


# ### Metrics Table Per Model Combination

# In[ ]:


metrics = Metrics()

encoder_old_names = [
    'RDKitDescriptors',
    'MorganFp',
    'RDKitFp',
    'TopologicalTorsionFp',
    'AtomPairFp',
    'MACCSKeysFp',
    'ChemBerta',
    'NativeEncoder'
]
encoder_new_names = [
    'RDKit Descriptors',
    'Morgan FP',
    'RDKit FP',
    'Topological Torsion FP',
    'Atom Pair FP',
    'MACCCS Keys FP',
    'ChemBERTa',
    'None'
]
predictor_old_names = [
    'LogisticRegressionPredictor',
    'RandomForestPredictor',
    'XGBoostPredictor',
    'MLP_Predictor',
    'CoralPredictor',
    'ChemPropMPNNPredictor',
    'ChemeleonPredictor'

]
predictor_new_names = [
    'Logistic Regression',
    'Random Forest',
    'XGBoost',
    'MLP',
    'CORAL',
    'ChemProp',
    'CheMeleon'
]
abbreviation_dict = {
    **dict(zip(encoder_old_names, encoder_new_names)),
    **dict(zip(predictor_old_names, predictor_new_names)),
}

def evaluate_prediction_df(df_pred: pd.DataFrame, y_test: np.ndarray) -> dict[str, float]:
    if df_pred is None or len(df_pred) == 0:
        return {}
    repeats = len(df_pred) // len(y_test) if len(y_test) > 0 else 1
    repeats = max(1, repeats)
    values = []
    for i in range(repeats):
        df_pred_subset = df_pred.iloc[i*len(y_test):(i+1)*len(y_test)]
        y_true = df_pred_subset['y_test'].values
        y_pred = df_pred_subset['preds'].values
        metrics_row = {
            'MSE macro': metrics.calculate_mse_macro(y_true, y_pred)[0],
            'MSE micro': metrics.calculate_mse(y_true, y_pred),
            'F1 macro': metrics.calculate_f1_macro(y_true, y_pred)[0],
            # 'f1_micro': metrics.calculate_f1_micro(y_true, y_pred),
            'Accuracy/F1 micro': metrics.calculate_accuracy(y_true, y_pred),
            'ROC AUC': metrics.calculate_roc_auc(y_true, y_pred)[0],
        }
        values.append(metrics_row)
    df_values = pd.DataFrame(values)
    return df_values.mean().to_dict()

def build_metrics_table(preds_path_template: str, y_test: np.ndarray) -> pd.DataFrame:
    metrics_rows = []
    for encoder in encoders:
        for predictor in predictors:
            preds_path = preds_path_template.format(
                encoder=encoder.__name__,
                predictor=predictor.__name__,
            )
            if not os.path.exists(preds_path):
                continue
            df_pred = pd.read_csv(preds_path, index_col=0)
            metrics_result = evaluate_prediction_df(df_pred, y_test)
            if not metrics_result:
                continue
            metrics_rows.append({
                'encoder': encoder.__name__,
                'predictor': predictor.__name__,
                **metrics_result
            })
    df_metrics = pd.DataFrame(metrics_rows)
    if df_metrics.empty:
        return df_metrics
    df_metrics = df_metrics.sort_values(['encoder', 'predictor']).reset_index(drop=True)
    df_metrics['encoder'] = df_metrics['encoder'].map(abbreviation_dict).fillna(df_metrics['encoder'])
    df_metrics['predictor'] = df_metrics['predictor'].map(abbreviation_dict).fillna(df_metrics['predictor'])
    df_metrics.rename(columns={'encoder': 'Descriptor', 'predictor': 'Predictor'}, inplace=True)
    return df_metrics

def build_metrics_table_combined(
    preds_path_step1_template: str,
    preds_path_step2_template: str,
    y_test: np.ndarray,
    decision_threshold: float = 0.5,
) -> pd.DataFrame:
    metrics_rows = []
    for encoder in encoders:
        for predictor in predictors:
            preds_path_1 = preds_path_step1_template.format(
                encoder=encoder.__name__,
                predictor=predictor.__name__,
            )
            preds_path_2 = preds_path_step2_template.format(
                encoder=encoder.__name__,
                predictor=predictor.__name__,
            )
            if not os.path.exists(preds_path_1) or not os.path.exists(preds_path_2):
                continue
            df_pred_1 = pd.read_csv(preds_path_1, index_col=0)
            df_pred_2 = pd.read_csv(preds_path_2, index_col=0)
            if len(df_pred_1) != len(df_pred_2):
                continue
            df_pred_combined = df_pred_2.copy()
            df_pred_combined.loc[df_pred_1['preds'] < decision_threshold, 'preds'] = df_pred_1.loc[df_pred_1['preds'] < decision_threshold, 'preds']
            metrics_result = evaluate_prediction_df(df_pred_combined, y_test)
            if not metrics_result:
                continue
            metrics_rows.append({
                'encoder': encoder.__name__,
                'predictor': predictor.__name__,
                **metrics_result
            })
    df_metrics = pd.DataFrame(metrics_rows)
    if df_metrics.empty:
        return df_metrics
    df_metrics = df_metrics.sort_values(['encoder', 'predictor']).reset_index(drop=True)
    df_metrics['encoder'] = df_metrics['encoder'].map(abbreviation_dict).fillna(df_metrics['encoder'])
    df_metrics['predictor'] = df_metrics['predictor'].map(abbreviation_dict).fillna(df_metrics['predictor'])
    df_metrics.rename(columns={'encoder': 'Descriptor', 'predictor': 'Predictor'}, inplace=True)
    return df_metrics

df_metrics_direct = build_metrics_table(
    'test_predictions/direct{encoder}{predictor}_predictions.csv',
    y_test,
 )
df_metrics_indirect_1 = build_metrics_table(
    'test_predictions/indirect_1{encoder}{predictor}_predictions.csv',
    y_test_has_odor,
 )
df_metrics_indirect_2 = build_metrics_table(
    'test_predictions/indirect_2{encoder}{predictor}_predictions_additional.csv',
    y_test,
 )
df_metrics_indirect_combined = build_metrics_table_combined(
    'test_predictions/indirect_1{encoder}{predictor}_predictions.csv',
    'test_predictions/indirect_2{encoder}{predictor}_predictions_additional.csv',
    y_test,
 )


# In[ ]:


print(df_metrics_direct.to_latex(index=False, float_format="%.2f"))


# In[ ]:


print(df_metrics_indirect_combined.to_latex(index=False, float_format="%.2f"))


# ### Paired t-tests

# In[ ]:


from scipy import stats

class Scorer:
    def __init__(self, metric: Callable[[np.ndarray, np.ndarray], float]):
        self.metric = metric

    def __call__(self, estimator, X, y):
        preds = estimator.predict(X)
        score = self.metric(y, preds)
        if isinstance(score, (list, tuple)):
            score = score[0]
        return score

def paired_ttest_5x2cv(estimator1, estimator2, X, y, scorer, groups, random_seed=None):
    """
    Implements the 5x2cv paired t test proposed
    by Dieterrich (1998)
    to compare the performance of two models.
    
    This is an adaption of the implementation provided by mlxtend:
    https://rasbt.github.io/mlxtend/user_guide/evaluate/paired_ttest

    Parameters
    ----------

    """
    rng = np.random.RandomState(random_seed)

    # if scoring is None:
    #     est_type = _infer_estimator_type(estimator1)
    #     if est_type == "classifier":
    #         scoring = "accuracy"
    #     elif est_type == "regressor":
    #         scoring = "r2"
    #     else:
    #         raise AttributeError("Estimator must be a Classifier or Regressor.")
    # if isinstance(scoring, str):
    #     scorer = get_scorer(scoring)
    # else:
    #     scorer = scoring
    variance_sum = 0.0
    first_diff = None

    def score_diff(X_1, X_2, y_1, y_2):
        estimator1.fit(X_1, y_1)
        estimator2.fit(X_1, y_1)
        est1_score = scorer(estimator1, X_2, y_2)
        est2_score = scorer(estimator2, X_2, y_2)
        score_diff = est1_score - est2_score
        return score_diff

    for i in range(5):
        X_1, X_2, y_1, y_2, _, _ = stratified_group_train_test_split(X, y, groups, test_size=0.5, random_state=random_seed)
        score_diff_1 = score_diff(X_1, X_2, y_1, y_2)
        score_diff_2 = score_diff(X_2, X_1, y_2, y_1)
        score_mean = (score_diff_1 + score_diff_2) / 2.0
        print('score_mean', score_mean)
        score_var = (score_diff_1 - score_mean) ** 2 + (score_diff_2 - score_mean) ** 2
        print('score_var', score_var)
        variance_sum += score_var
        if first_diff is None:
            first_diff = score_diff_1
            print('first_diff', first_diff)

    numerator = first_diff
    denominator = np.sqrt(1 / 5.0 * variance_sum)
    t_stat = numerator / denominator

    pvalue = stats.t.sf(np.abs(t_stat), 5) * 2.0
    return float(t_stat), float(pvalue)


# In[ ]:


class IndirectOdorStrengthPredictor:
    def __init__(self, odor_detection_model, odor_strength_model):
        self.odor_detection_model = odor_detection_model
        self.odor_strength_model = odor_strength_model

    def fit(self, X, y):
        y_has_odor = y.copy()
        y_has_odor[y_has_odor>0] = 1
        self.odor_detection_model.fit(X, y_has_odor)
        X_wo_odorless = np.array(X)[np.array(y)>0].tolist()
        y_wo_odorless = y.copy()[np.array(y)>0]
        self.odor_strength_model.fit(X_wo_odorless, y_wo_odorless)

    def predict(self, X):
        has_odor_preds = self.odor_detection_model.predict(X)
        strength_preds = self.odor_strength_model.predict(X)
        final_preds = []
        for has_odor_pred, strength_pred in zip(has_odor_preds, strength_preds):
            if has_odor_pred == 0:
                final_preds.append(0)
            else:
                final_preds.append(strength_pred)
        return np.array(final_preds)


# In[ ]:


if os.path.exists('paired_ttests/direct_vs_indirect_paired_ttest_results.csv'):
    df_results = pd.read_csv('paired_ttests/direct_vs_indirect_paired_ttest_results.csv')
else:
    # loop over all encoder-predictor combinations and perform paired t-test between direct and indirect approach
    results = []
    scorer = Scorer(Metrics().calculate_mse_macro)
    for encoder in encoders:
        for predictor in predictors:
            try:
                print('encoder', encoder.__name__, 'predictor', predictor.__name__)
                # create model encoder and predictors and combine to model
                t_stat, p_value = paired_ttest_5x2cv(
                    estimator1=models_direct[encoder.__name__][predictor.__name__](**hyperparameters_direct[encoder.__name__][predictor.__name__]),
                    estimator2=IndirectOdorStrengthPredictor(
                        odor_detection_model=models_first_step[encoder.__name__][predictor.__name__](**hyperparameters_first_step[encoder.__name__][predictor.__name__]),
                        odor_strength_model=models_second_step[encoder.__name__][predictor.__name__](**hyperparameters_second_step[encoder.__name__][predictor.__name__])
                    ),
                    X=np.array(X_test),
                    y=y_test,
                    scorer=scorer,
                    groups=groups_test,
                    random_seed=0
                )
                results.append({
                    'encoder': encoder.__name__,
                    'predictor': predictor.__name__,
                    't_statistic': t_stat,
                    'p_value': p_value
                })
            except Exception as e:
                print(f'Error processing {encoder.__name__} with {predictor.__name__}: {e}')
    df_results = pd.DataFrame(results)
    if not os.path.isdir('paired_ttests'):
        os.makedirs('paired_ttests/')
    df_results.to_csv('paired_ttests/direct_vs_indirect_paired_ttest_results.csv', index=False)


# In[ ]:


df_results.rename(columns={'encoder': 'Descriptor', 'predictor': 'Predictor', 't_stat': 't value', 'p_value': 'p value'}, inplace=True)
df_results['Descriptor'] = df_results['Descriptor'].map(abbreviation_dict).fillna(df_results['Descriptor'])
df_results['Predictor'] = df_results['Predictor'].map(abbreviation_dict).fillna(df_results['Predictor'])
print(df_results.to_latex(index=False, float_format="%.2f"))


# In[ ]:


# Pairwise 5x2cv t-tests among the four top-performing direct models (lowest Macro MSE in df_direct)
df_direct_long = (
    df_direct.stack()
    .reset_index()
    .rename(columns={'level_0': 'predictor', 'level_1': 'encoder', 0: 'score'})
)
df_direct_long = df_direct_long.dropna(subset=['score']).sort_values('score', ascending=True).reset_index(drop=True)

top4_direct = df_direct_long.head(4).copy()
top4_direct['rank'] = np.arange(1, len(top4_direct) + 1)
print(top4_direct[['rank', 'encoder', 'predictor', 'score']])

# Recreate the canonical split to avoid mutated kernel variables (e.g., y_test stacked in later cells)
X_all = df_odor_strength['canonical_smiles'].values
y_all = df_odor_strength['numerical_strength'].values
_, X_pair, _, y_pair, _, groups_pair = stratified_group_train_test_split(
    X_all, y_all, groups, random_state=42
)
X_pair = np.array(X_pair)
y_pair = np.array(y_pair)
groups_pair = np.array(groups_pair)

def align_predictions_with_targets(preds: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    preds = np.asarray(preds).reshape(-1)
    y_true = np.asarray(y_true).reshape(-1)
    if preds.shape[0] == y_true.shape[0]:
        return preds
    if preds.shape[0] % y_true.shape[0] == 0:
        repeats = preds.shape[0] // y_true.shape[0]
        preds = preds.reshape(repeats, y_true.shape[0]).mean(axis=0)
        return preds
    raise ValueError(
        f'Incompatible prediction/target lengths: preds={preds.shape[0]}, y={y_true.shape[0]}'
    )

class RobustScorer:
    def __init__(self, metric: Callable[[np.ndarray, np.ndarray], float]):
        self.metric = metric

    def __call__(self, estimator, X, y):
        preds = estimator.predict(X)
        preds = align_predictions_with_targets(preds, y)
        score = self.metric(y, preds)
        if isinstance(score, (list, tuple)):
            score = score[0]
        return score

results_pairwise = []
scorer = RobustScorer(Metrics().calculate_mse_macro)

for i in range(len(top4_direct)):
    for j in range(i + 1, len(top4_direct)):
        row_i = top4_direct.iloc[i]
        row_j = top4_direct.iloc[j]

        encoder_i, predictor_i = row_i['encoder'], row_i['predictor']
        encoder_j, predictor_j = row_j['encoder'], row_j['predictor']

        print(f"Comparing ({i+1}) {encoder_i} + {predictor_i} vs ({j+1}) {encoder_j} + {predictor_j}")
        try:
            estimator_i = models_direct[encoder_i][predictor_i](**hyperparameters_direct[encoder_i][predictor_i])
            estimator_j = models_direct[encoder_j][predictor_j](**hyperparameters_direct[encoder_j][predictor_j])

            t_stat, p_value = paired_ttest_5x2cv(
                estimator1=estimator_i,
                estimator2=estimator_j,
                X=X_pair,
                y=y_pair,
                scorer=scorer,
                groups=groups_pair,
                random_seed=0,
            )

            results_pairwise.append({
                'model_a_rank': int(row_i['rank']),
                'model_a_encoder': encoder_i,
                'model_a_predictor': predictor_i,
                'model_a_score': float(row_i['score']),
                'model_b_rank': int(row_j['rank']),
                'model_b_encoder': encoder_j,
                'model_b_predictor': predictor_j,
                'model_b_score': float(row_j['score']),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
            })
        except Exception as e:
            print(f"Error comparing {encoder_i}/{predictor_i} vs {encoder_j}/{predictor_j}: {e}")

df_pairwise_ttests_top4_direct = pd.DataFrame(results_pairwise)

if not os.path.isdir('paired_ttests'):
    os.makedirs('paired_ttests/')

save_path = 'paired_ttests/top4_direct_pairwise_5x2cv_ttests.csv'
df_pairwise_ttests_top4_direct.to_csv(save_path, index=False)
print(f'Saved pairwise 5x2cv results to: {save_path}')

df_pairwise_ttests_top4_direct


# In[ ]:


# P-value matrix: rows = encoder/predictor A, columns = encoder/predictor B (mapped with abbreviation_dict)
df_pairwise = df_pairwise_ttests_top4_direct.copy()


def map_model_name(encoder_name: str, predictor_name: str) -> str:
    encoder_mapped = abbreviation_dict.get(encoder_name, encoder_name)
    predictor_mapped = abbreviation_dict.get(predictor_name, predictor_name)
    return f"{encoder_mapped} | {predictor_mapped}"

df_pairwise['model_a'] = df_pairwise.apply(
    lambda row: map_model_name(row['model_a_encoder'], row['model_a_predictor']),
    axis=1,
    )
df_pairwise['model_b'] = df_pairwise.apply(
    lambda row: map_model_name(row['model_b_encoder'], row['model_b_predictor']),
    axis=1,
    )

# Keep ordering by rank from top4_direct and map names
ordered_models = [
    map_model_name(row['encoder'], row['predictor'])
    for _, row in top4_direct.iterrows()
    ]

# Upper-triangular pivot
df_pvalues_top4_direct = df_pairwise.pivot(index='model_a', columns='model_b', values='p_value')

# Reindex for consistent model order
df_pvalues_top4_direct = df_pvalues_top4_direct.reindex(index=ordered_models, columns=ordered_models)

# Fill diagonal and mirror lower triangle for a full symmetric matrix
for model_name in ordered_models:
    df_pvalues_top4_direct.loc[model_name, model_name] = np.nan
for i, row_name in enumerate(ordered_models):
    for j, col_name in enumerate(ordered_models):
        if i > j and pd.isna(df_pvalues_top4_direct.loc[row_name, col_name]):
            df_pvalues_top4_direct.loc[row_name, col_name] = df_pvalues_top4_direct.loc[col_name, row_name]

df_pvalues_top4_direct


# In[ ]:


print(df_pvalues_top4_direct.to_latex(float_format="%.2f", na_rep="--"))


# ### Indirect Approach: Combining Two-Step Predictions
# 
# This section combines the predictions from the two-step indirect approach:
# 1. First model predicts if a molecule has any odor (binary classification)
# 2. Second model predicts odor strength for molecules classified as odorous (regression)

# In[ ]:


y_test_stack = np.hstack([y_test]*N_REPEATS)


# In[ ]:


save_path_addition_1 = 'indirect_1'
save_path_addition_2 = 'indirect_2'
mse_macros = {encoder.__name__: {} for encoder in encoders}

for encoder in encoders:
    for predictor in predictors:
        try:
            df_pred_1 = pd.read_csv('test_predictions/' + save_path_addition_1 + encoder.__name__ + predictor.__name__ + '_predictions.csv', index_col=0)
            df_pred_2 = pd.read_csv('test_predictions/' + save_path_addition_2 + encoder.__name__ + predictor.__name__ + '_predictions_additional.csv', index_col=0)
            df_pred_combined = df_pred_2.copy()
            df_pred_combined.loc[df_pred_1['preds'] < 0.5, 'preds'] = df_pred_1.loc[df_pred_1['preds'] < 0.5, 'preds']
            mse_macros[encoder.__name__][predictor.__name__] = Metrics().calculate_mse_macro(y_test_stack, df_pred_combined['preds'].values)[0]
        except Exception as e:
            print(f"Error processing predictions for {encoder.__name__} {predictor.__name__}: {e}")
            continue
df_odor_strength_indirect = pd.DataFrame(mse_macros)


# ### Heatmap Visualization
# 

# In[ ]:


from typing import Mapping, Tuple
DPI = 600
colors = [okabe_ito[1], okabe_ito[3]]
mycmap = LinearSegmentedColormap.from_list("mycmap", colors, N=256)
width_factor = np.sqrt(2.45/2.2)

encoder_label_map = {
    'NativeEncoder': 'None',
    'RDKitDescriptors': 'RDKit Descriptors',
    'MorganFp': 'Morgan FP',
    'RDKitFp': 'RDKit FP',
    'TopologicalTorsionFp': 'Topological Torsion FP',
    'AtomPairFp': 'Atom Pair FP',
    'MACCSKeysFp': 'MACCCS Keys FP',
    'ChemBerta': 'ChemBERTa',
}

predictor_label_map = {
    'LogisticRegressionPredictor': 'Logistic Regression',
    'RandomForestPredictor': 'Random Forest',
    'XGBoostPredictor': 'XGBoost',
    'MLP_Predictor': 'MLP',
    'CoralPredictor': 'CORAL',
    'ChemPropMPNNPredictor': 'ChemProp',
    'ChemeleonPredictor': 'CheMeleon',
}


def plot_heatmap(
        df: pd.DataFrame,
        colorbar_label: str,
        figsize: Tuple[int, int] = (10, 10),
        fontsize: int = 12,
        labelsize: int = 12,
        labelpad: int = 10,
        dpi: int = 100,
        save_path: list[str] | None = None,
        custom_x_tick_mapping: Mapping[str, str] | None = None,
        custom_y_tick_mapping: Mapping[str, str] | None = None,
) -> None:
    """
    Create a performance heatmap for encoder-predictor combinations.

    This function visualizes model performance across different combinations of
    molecular encoders and predictors using a color-coded heatmap.

    Args:
        df (pd.DataFrame): Performance matrix with predictors as rows and encoders as columns
        colorbar_label (str): Label for the colorbar indicating the performance metric
        figsize (Tuple[int, int]): Figure size in inches (width, height)
        fontsize (int): Font size for labels and titles
        labelsize (int): Font size for tick labels
        labelpad (int): Padding for axis labels
        dpi (int): Resolution for saved figures
        save_path (list[str], optional): List of file paths to save the figure
        custom_x_tick_mapping (Mapping[str, str], optional): Mapping of encoder names to display labels
        custom_y_tick_mapping (Mapping[str, str], optional): Mapping of predictor names to display labels
    """
    plt.figure(figsize=figsize)
    plt.imshow(df, cmap=mycmap, aspect='auto')
    colorbar = plt.colorbar()
    colorbar.set_label(colorbar_label, fontsize=fontsize, labelpad=labelpad+labelpad*0.2)
    colorbar.ax.tick_params(labelsize=labelsize)

    if custom_x_tick_mapping is None:
        x_labels = [label.replace('Encoder', '') for label in df.columns]
    else:
        x_labels = [custom_x_tick_mapping.get(label, label.replace('Encoder', '')) for label in df.columns]

    if custom_y_tick_mapping is None:
        y_labels = [label.split('Predictor')[0] for label in df.index]
    else:
        y_labels = [custom_y_tick_mapping.get(label, label.split('Predictor')[0]) for label in df.index]

    plt.xticks(ticks=range(len(df.columns)), labels=x_labels, rotation=45, ha='right')
    plt.yticks(ticks=range(len(df.index)), labels=y_labels)
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    plt.tight_layout()
    if save_path is not None:
        for path in save_path:
            plt.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.show()


plot_heatmap(
    df_indirect_1.dropna(how='all').iloc[:4, 1:],
    colorbar_label='F1 Score',
    figsize=(FIGURE_WIDTH*1.2, 2*1.2),
    dpi=DPI,
    fontsize=FONTSIZE,
    labelsize=LABELSIZE,
    labelpad=LABELPAD,
    save_path=['figures/hp_opt_heatmap_has_odor.pdf'],
    custom_x_tick_mapping=encoder_label_map,
    custom_y_tick_mapping=predictor_label_map,
)
plot_heatmap(
    - df_indirect_2.dropna(how='all').iloc[:5, 1:],
    colorbar_label='Negative Macro MSE',
    figsize=(FIGURE_WIDTH*1.2, 2.2*1.2),
    dpi=DPI,
    fontsize=FONTSIZE,
    labelsize=LABELSIZE,
    labelpad=LABELPAD,
    save_path=['figures/hp_opt_heatmap_wo_odorless.pdf'],
    custom_x_tick_mapping=encoder_label_map,
    custom_y_tick_mapping=predictor_label_map,
)
plot_heatmap(
    - df_odor_strength_indirect.dropna(how='all').iloc[2:, 1:],
    colorbar_label='Negative Macro MSE',
    figsize=(FIGURE_WIDTH*1.2, 2.2*1.2),
    dpi=DPI,
    fontsize=FONTSIZE,
    labelsize=LABELSIZE,
    labelpad=LABELPAD,
    save_path=['figures/hp_opt_heatmap_indirect.pdf'],
    custom_x_tick_mapping=encoder_label_map,
    custom_y_tick_mapping=predictor_label_map,
)
#
plot_heatmap(
    - df_direct.dropna(how='all').iloc[:5, 1:],
    colorbar_label=' Negative Macro MSE',
    figsize=(FIGURE_WIDTH*width_factor, 2.2*width_factor),
    dpi=DPI,
    fontsize=FONTSIZE,
    labelsize=LABELSIZE,
    labelpad=LABELPAD,
    save_path=['figures/hp_opt_heatmap_w_odorless.pdf'],
    custom_x_tick_mapping=encoder_label_map,
    custom_y_tick_mapping=predictor_label_map,
)
# difference heatmap
plot_heatmap(
    df_odor_strength_indirect.iloc[2:, 1:] - df_direct.dropna(how='all').iloc[:4, 1:],
    colorbar_label='Difference Macro MSE',
    figsize=(FIGURE_WIDTH*1.2, 2*1.2),
    dpi=DPI,
    fontsize=FONTSIZE,
    labelsize=LABELSIZE,
    labelpad=LABELPAD,
    save_path=['figures/hp_opt_heatmap_difference.pdf', 'figures/hp_opt_heatmap_difference.png'],
    custom_x_tick_mapping=encoder_label_map,
    custom_y_tick_mapping=predictor_label_map,
)


# ### Model Comparison: Direct vs. Indirect Approaches
# 
# This section compares the best-performing models from both approaches on the test set to determine which strategy works better for odor strength prediction.

# In[ ]:
from data.molecules.smiles_converter import SmilesCanonicalizer

class Ensemble:
    def __init__(self, models: list):
        self.models = models
        self.smiles_canonicalizer = SmilesCanonicalizer()

    def fit(self, X, y):
        X = self.smiles_canonicalizer.canonicalize_smiles(X)
        for model in self.models:
            model.fit(X, y)

    def predict(self, X):
        X = self.smiles_canonicalizer.canonicalize_smiles(X)
        preds = np.array([model.predict(X) for model in self.models])
        return np.mean(preds, axis=0)
    
    def save(self, encoder_paths: list[str], predictor_paths: list[str], predictor_hyperparameters_paths: list[str]):
        for model, encoder_path, predictor_path, hyperparameters_path in zip(self.models, encoder_paths, predictor_paths, predictor_hyperparameters_paths):
            model.save(encoder_path, predictor_path, hyperparameters_path)

    def load(self, encoder_paths: list[str], predictor_paths: list[str], predictor_hyperparameters_paths: list[str]):
        for i, (encoder_path, predictor_path, hyperparameters_path) in enumerate(zip(encoder_paths, predictor_paths, predictor_hyperparameters_paths)):
            self.models[i].load(encoder_path, predictor_path, hyperparameters_path)
    

class IndirectEnsemble:
    def __init__(self, odor_detection_models, odor_strength_models):
        self.odor_detection_models = odor_detection_models
        self.odor_strength_models = odor_strength_models

    def fit(self, X, y):
        y_has_odor = y.copy()
        y_has_odor[y_has_odor>0] = 1
        X_wo_odorless = np.array(X)[np.array(y)>0].tolist()
        y_wo_odorless = y.copy()[np.array(y)>0]
        for odor_detection_model in self.odor_detection_models:
            odor_detection_model.fit(X, y_has_odor)
        for odor_strength_model in self.odor_strength_models:
            odor_strength_model.fit(X_wo_odorless, y_wo_odorless)

    def predict(self, X):
        has_odor_preds = [model.predict(X) for model in self.odor_detection_models]
        has_odor_preds = np.mean(has_odor_preds, axis=0)
        strength_preds = [model.predict(X) for model in self.odor_strength_models]
        strength_preds = np.mean(strength_preds, axis=0)
        final_preds = []
        for has_odor_pred, strength_pred in zip(has_odor_preds, strength_preds):
            if has_odor_pred == 0:
                final_preds.append(0)
            else:
                final_preds.append(strength_pred)
        return np.array(final_preds)


# In[ ]:


preds_list = []
preds_two_step_list = []
X = df_odor_strength['canonical_smiles'].values
y = df_odor_strength['has_odor'].values
stratify_data = df_odor_strength['numerical_strength'].values
X_train, X_test, y_train, y_test, groups_train, groups_test = stratified_group_train_test_split(X, stratify_data, groups, random_state=42)
X_train = X_train.tolist()
X_test = X_test.tolist()


direct_model = Ensemble([
    models_direct['RDKitDescriptors']['RandomForestPredictor'](**hyperparameters_direct['RDKitDescriptors']['RandomForestPredictor']),
    models_direct['RDKitDescriptors']['XGBoostPredictor'](**hyperparameters_direct['RDKitDescriptors']['XGBoostPredictor']),
    models_direct['RDKitDescriptors']['MLP_Predictor'](**hyperparameters_direct['RDKitDescriptors']['MLP_Predictor'])
])

for i in range(N_REPEATS):
    direct_model.fit(X_train, y_train)
#     model.save(
#     encoder_path='encoder.gz',
#     predictor_path='predictor.pth',
#     predictor_hyperparameter_path='predictor_hyperparameters.json',
#     config_path='model_config.txt',
# )
    preds = direct_model.predict(X_test)
    preds_list.append(preds)


    # indirect_first_step_model = best_model_indirect_first_step_class(**best_hyperparameters_indirect_first_step)
    # indirect_second_step_model = best_model_indirect_second_step_class(**best_hyperparameters_indirect_second_step)
    

    # y_train_has_odor = y_train.copy()
    # y_train_has_odor[y_train_has_odor>0] = 1
    # indirect_first_step_model.fit(X_train, y_train_has_odor)

    # y_train_wo_odorless = y_train.copy()
    # X_train_wo_odorless = np.array(X_train)[y_train_wo_odorless >= 1].tolist()
    # y_train_wo_odorless = y_train_wo_odorless[y_train_wo_odorless >= 1]
    # indirect_second_step_model.fit(X_train_wo_odorless, y_train_wo_odorless)

    # preds_two_step = indirect_first_step_model.predict(X_test)
    # preds_two_step[preds_two_step>0.5] = indirect_second_step_model.predict(np.array(X_test)[preds_two_step>0.5].tolist())
    # preds_two_step_list.append(preds_two_step)

preds = np.hstack(preds_list)
# preds_two_step = np.hstack(preds_two_step_list)
y_test_stack = np.hstack([y_test]*N_REPEATS)


# In[ ]:


preds_two_step_list = []
X = df_odor_strength['canonical_smiles'].values
y = df_odor_strength['has_odor'].values
stratify_data = df_odor_strength['numerical_strength'].values
X_train, X_test, y_train, y_test, groups_train, groups_test = stratified_group_train_test_split(X, stratify_data, groups, random_state=42)
X_train = X_train.tolist()
X_test = X_test.tolist()


indirect_model = IndirectEnsemble([
    models_direct['RDKitDescriptors']['RandomForestPredictor'](**hyperparameters_direct['RDKitDescriptors']['RandomForestPredictor']),
    models_direct['RDKitDescriptors']['XGBoostPredictor'](**hyperparameters_direct['RDKitDescriptors']['XGBoostPredictor']),
    models_direct['RDKitDescriptors']['MLP_Predictor'](**hyperparameters_direct['RDKitDescriptors']['MLP_Predictor'])
])

for i in range(N_REPEATS):
    indirect_model.fit(X_train, y_train)
#     model.save(
#     encoder_path='encoder.gz',
#     predictor_path='predictor.pth',
#     predictor_hyperparameter_path='predictor_hyperparameters.json',
#     config_path='model_config.txt',
# )
    preds_two_step = indirect_model.predict(X_test)
    preds_two_step_list.append(preds_two_step)


    # indirect_first_step_model = best_model_indirect_first_step_class(**best_hyperparameters_indirect_first_step)
    # indirect_second_step_model = best_model_indirect_second_step_class(**best_hyperparameters_indirect_second_step)
    

    # y_train_has_odor = y_train.copy()
    # y_train_has_odor[y_train_has_odor>0] = 1
    # indirect_first_step_model.fit(X_train, y_train_has_odor)

    # y_train_wo_odorless = y_train.copy()
    # X_train_wo_odorless = np.array(X_train)[y_train_wo_odorless >= 1].tolist()
    # y_train_wo_odorless = y_train_wo_odorless[y_train_wo_odorless >= 1]
    # indirect_second_step_model.fit(X_train_wo_odorless, y_train_wo_odorless)

    # preds_two_step = indirect_first_step_model.predict(X_test)
    # preds_two_step[preds_two_step>0.5] = indirect_second_step_model.predict(np.array(X_test)[preds_two_step>0.5].tolist())
    # preds_two_step_list.append(preds_two_step)

preds_two_step = np.hstack(preds_two_step_list)
y_test_stack = np.hstack([y_test]*N_REPEATS)


# In[ ]:


from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from utility.colors import okabe_ito

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_2: np.ndarray|None = None, # for difference plotting
    figsize: tuple = (10, 10),
    dpi: int = 100,
    tick_labels: dict = {0: "Odorless", 1: "Low", 2: "Medium", 3: "High"},
    cmap: str|LinearSegmentedColormap = 'viridis',
    text_annotation: bool = True,
    fontsize: int = 20,
    labelsize: int = 18,
    labelpad: int = 10,
    save_path: str | None = None,
    center_value: float | None = None,
    ) -> None:
    """
    Plots a normalized (by true values) confusion matrix for classification results.

    Parameters
    ----------
    y_true : np.ndarray
        Array of true class labels.
    y_pred : np.ndarray
        Array of predicted class labels.
    figsize : tuple, optional
        Size of the figure in inches (width, height). Default is (10, 10).
    dpi : int, optional
        Dots per inch for the figure resolution. Default is 100.
    tick_labels : dict, optional
        Dictionary mapping class indices to label names for axis ticks. If prediction or true values are higher or lower than the keys in this dictionary, they will be set to the maximum or minimum key value, respectively.
        Default is {0: "None", 1: "Low", 2: "Medium", 3: "High"}.

    Returns
    -------
    None
        Displays the confusion matrix plot.
    """
    y_pred = y_pred.round().astype(int)
    y_true = y_true.round().astype(int)
    max_tick_value = max(tick_labels.keys())
    min_tick_value = min(tick_labels.keys())
    y_pred[y_pred > max_tick_value], y_pred[y_pred < min_tick_value] = max_tick_value, min_tick_value
    y_true[y_true > max_tick_value], y_true[y_true < min_tick_value] = max_tick_value, min_tick_value
    confusion = confusion_matrix(y_true, y_pred)
    confusion = confusion.T
    normed_confusion_matrix = confusion / np.sum(confusion, axis=0)
    if y_pred_2 is not None:
        y_pred_2 = y_pred_2.round().astype(int)
        y_pred_2[y_pred_2 > max_tick_value], y_pred_2[y_pred_2 < min_tick_value] = max_tick_value, min_tick_value
        confusion_2 = confusion_matrix(y_true, y_pred_2)
        confusion_2 = confusion_2.T
        normed_confusion_matrix_2 = confusion_2 / np.sum(confusion_2, axis=0)
        normed_confusion_matrix = normed_confusion_matrix_2 - normed_confusion_matrix
    fig, ax = plt.subplots(figsize=figsize)
    # Calculate vmin and vmax for centering colorbar
    if center_value is not None:
        data_min = normed_confusion_matrix.min()
        data_max = normed_confusion_matrix.max()
        max_abs_diff = max(abs(data_max - center_value), abs(data_min - center_value))
        vmin = center_value - max_abs_diff
        vmax = center_value + max_abs_diff
    else:
        vmin = None
        vmax = None
    im = ax.imshow(normed_confusion_matrix, cmap=cmap, aspect='equal', origin='lower', vmin=vmin, vmax=vmax)
    if text_annotation:
        for i in range(normed_confusion_matrix.shape[0]):
            for j in range(normed_confusion_matrix.shape[1]):
                text = ax.text(
                    j, i, f"{normed_confusion_matrix[i, j]:.2f}",
                    ha='center', va='center',
                    color='white' # if normed_confusion_matrix[i, j] < 0.5 else 'black'  # adjust for contrast
                )
    colorbar = plt.colorbar(im)
    colorbar.set_label('Ratio of test values', fontsize=fontsize, labelpad=labelpad+labelpad*0.2)
    colorbar.ax.tick_params(labelsize=labelsize)
    plt.xticks(list(tick_labels.keys()), list(tick_labels.values()))
    plt.yticks(list(tick_labels.keys()), list(tick_labels.values()))
    plt.tick_params(axis='both', which='major', labelsize=labelsize)
    plt.xlabel('Test Set', fontsize=fontsize, labelpad=labelpad)
    plt.ylabel('Prediction', fontsize=fontsize, labelpad=labelpad)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.show()

colors = [okabe_ito[1], okabe_ito[3]]
mycmap = LinearSegmentedColormap.from_list("mycmap", colors, N=256)

df_metrics_direct_ensemble = pd.Series({
    'MSE Macro': Metrics().calculate_mse_macro(y_test_stack, preds)[0],
    'MSE Odorless': Metrics().calculate_mse_macro(y_test_stack, preds)[1][0],
    'MSE Low': Metrics().calculate_mse_macro(y_test_stack, preds)[1][1],
    'MSE Medium': Metrics().calculate_mse_macro(y_test_stack, preds)[1][2],
    'MSE High': Metrics().calculate_mse_macro(y_test_stack, preds)[1][3],
    'MSE Micro': Metrics().calculate_mse(y_test_stack, preds),
    'F1 Macro': Metrics().calculate_f1_macro(y_test_stack, preds)[0],
    'F1 Odorless': Metrics().calculate_f1_macro(y_test_stack, preds)[1][0],
    'F1 Low': Metrics().calculate_f1_macro(y_test_stack, preds)[1][1],
    'F1 Medium': Metrics().calculate_f1_macro(y_test_stack, preds)[1][2],
    'F1 High': Metrics().calculate_f1_macro(y_test_stack, preds)[1][3],
    'Accuracy/F1 Micro': Metrics().calculate_accuracy(y_test_stack, preds),
    'ROC AUC': Metrics().calculate_roc_auc(y_test_stack, preds)[0],
    'ROC AUC Low Threshold': Metrics().calculate_roc_auc(y_test_stack, preds)[1][0],
    'ROC AUC Medium Threshold': Metrics().calculate_roc_auc(y_test_stack, preds)[1][1],
    'ROC AUC High Threshold': Metrics().calculate_roc_auc(y_test_stack, preds)[1][2],
})

# print(Metrics().calculate_mse_macro(y_test, preds), Metrics().calculate_mse(y_test, preds))
# samples_per_repeat = int(y_test.shape[0]/N_REPEATS)
# mse_macros = [Metrics().calculate_mse_macro(y_test[i*samples_per_repeat:(i+1)*samples_per_repeat], preds[i*samples_per_repeat:(i+1)*samples_per_repeat])[0] for i in range(N_REPEATS)]
# print('Standard MSE Macro Deviation per repeat', np.std(mse_macros))
# print(mse_macros)
# print(Metrics().calculate_r2(y_test, preds))
# r2s = [Metrics().calculate_r2(y_test[i*samples_per_repeat:(i+1)*samples_per_repeat], preds[i*samples_per_repeat:(i+1)*samples_per_repeat]) for i in range(N_REPEATS)]
# print('Standard R2 Deviation per repeat', np.std(r2s))
plot_confusion_matrix(
    y_test_stack,
    preds,
    cmap=mycmap,
    figsize=(FIGURE_WIDTH, 2.45),
    dpi=DPI,
    text_annotation=False,
    fontsize=FONTSIZE,
    labelsize=LABELSIZE,
    labelpad=LABELPAD,
    )

# print(Metrics().calculate_mse_macro(y_test, preds_two_step), Metrics().calculate_mse(y_test, preds_two_step))
# mse_macros_two_step = [Metrics().calculate_mse_macro(y_test[i*samples_per_repeat:(i+1)*samples_per_repeat], preds_two_step[i*samples_per_repeat:(i+1)*samples_per_repeat])[0] for i in range(N_REPEATS)]
# print('Standard MSE Macro Deviation per repeat', np.std(mse_macros_two_step))
# print(mse_macros_two_step)
# print(Metrics().calculate_r2(y_test, preds_two_step))
# r2s_two_step = [Metrics().calculate_r2(y_test[i*samples_per_repeat:(i+1)*samples_per_repeat], preds_two_step[i*samples_per_repeat:(i+1)*samples_per_repeat]) for i in range(N_REPEATS)]
# print('Standard R2 Deviation per repeat', np.std(r2s_two_step))
# plot_confusion_matrix(
#     y_test_stack,
#     preds_two_step,
#     cmap=mycmap,
#     figsize=(FIGURE_WIDTH, 3),
#     dpi=DPI,
#     text_annotation=False,
#     fontsize=FONTSIZE,
#     labelsize=LABELSIZE,
#     labelpad=LABELPAD,
# )

# # difference plot
# colors = [okabe_ito[1], '#808080', okabe_ito[3]]
# mycmap3 = LinearSegmentedColormap.from_list("mycmap", colors, N=256)
# plot_confusion_matrix(
#     y_test_stack,
#     preds,
#     y_pred_2=preds_two_step,
#     cmap=mycmap3,
#     figsize=(FIGURE_WIDTH, 3),
#     dpi=DPI,
#     text_annotation=False,
#     fontsize=FONTSIZE,
#     labelsize=LABELSIZE,
#     labelpad=LABELPAD,
#     center_value=0.0
# )


# In[ ]:


print(df_metrics_direct_ensemble.to_latex(float_format="%.2f"))


# ## External Validation on Keller 2016 Dataset
# 
# This section validates the best-performing model on an independent dataset from Keller et al. (2016) to assess generalization performance.

# In[ ]:


from data.data_cleaner import GoodScentsDataCleaner
import pandas as pd
import os

cleaned_keller_path = 'data/keller_2016/cleaned_keller_2016.csv'
if os.path.exists(cleaned_keller_path):
    keller_2016 = pd.read_csv(cleaned_keller_path)
else:
    keller_2016 = pd.read_excel('data/keller_2016/12868_2016_287_MOESM1_ESM.xlsx', header=2)
    keller_2016.rename(columns={'C.A.S.': 'cas'}, inplace=True)
    data_cleaner = GoodScentsDataCleaner(data=keller_2016)
    data_cleaner.clean_molecules()
    keller_2016 = data_cleaner.data
    keller_2016.to_csv(cleaned_keller_path, index=False)


# In[ ]:


keller_2016 = keller_2016[keller_2016['canonical_smiles'].notna()]
keller_2016['predicted_intensity'] = direct_model.predict(keller_2016['canonical_smiles'].tolist())
keller_2016['predicted_intensity_rounded'] = keller_2016['predicted_intensity'].round().astype(float)


# In[ ]:


keller_2016_test  = keller_2016[~keller_2016['canonical_smiles'].isin(df_odor_strength['canonical_smiles'])]
from tqdm import tqdm
def calculate_morgan_tanimoto_similarity(smiles_list_1: list[str], smiles_list_2: list[str]) -> pd.DataFrame:
    """
    Check for molecular similarity between training and external test sets.
    
    This function computes Tanimoto similarity coefficients using Morgan fingerprints
    to identify molecules in the external test set that are too similar to training
    molecules, which could lead to data leakage.
    
    Args:
        smiles_list_1 (list): SMILES strings from training dataset
        smiles_list_2 (list): SMILES strings from external test dataset
        
    Returns:
        pd.DataFrame: Similarity matrix with external molecules as rows and 
                     training molecules as columns
    """
    morgan_fingerprint = MorganFp(radius=3, fpSize=2048)
    morgan_fingerprints_1 = morgan_fingerprint.encode(smiles_list_1)
    morgan_fingerprints_2 = morgan_fingerprint.encode(smiles_list_2)
    # morgan_similarities = {smiles_1: np.sum(np.logical_and(np.array(morgan_fingerprints_2), fp1), axis=1) / len(fp1) for smiles_1, fp1 in tqdm(zip(smiles_list_1, morgan_fingerprints_1))}
    morgan_similarities = {smiles_1: np.sum(np.logical_and(np.array(morgan_fingerprints_2), fp1), axis=1) / np.sum(np.logical_or(np.array(morgan_fingerprints_2), fp1), axis=1) for smiles_1, fp1 in tqdm(zip(smiles_list_1, morgan_fingerprints_1))}

    # morgan_similarities = {smiles_2: [jaccard_score(fp1, fp2) for smiles_1, fp1 in zip(smiles_list_1, morgan_fingerprints_1)] for smiles_2, fp2 in tqdm(zip(smiles_list_2, morgan_fingerprints_2))}
    df_morgan_similarities = pd.DataFrame(morgan_similarities, index=smiles_list_2)
    return df_morgan_similarities
df_morgan_similarities = calculate_morgan_tanimoto_similarity(df_odor_strength['canonical_smiles'].tolist(), keller_2016_test['canonical_smiles'].unique().tolist())


# In[ ]:


smiles_to_remove = df_morgan_similarities[df_morgan_similarities.max(axis=1) > 0.8].index
keller_2016_test = keller_2016_test[~keller_2016_test['canonical_smiles'].isin(smiles_to_remove)]
keller_2016_test['canonical_smiles'].nunique()


# In[ ]:


keller_2016_test['Odor dilution'] = keller_2016_test['Odor dilution'].map({  
    '1/10': 0.1,
    '1/1,000': 0.001,
    '1/100,000': 0.00001,
    '1/10,000,000': 1e-07
}).astype(float)


# In[ ]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from utility.colors import okabe_ito

FIGURE_WIDTH_LONG = 17.1 / 2.54

def plot_combined_figure_with_violins(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    df: pd.DataFrame,
    keller_data: pd.DataFrame,
    colorbar_label_1: str,
    colorbar_label_2: str,
    tick_labels: dict = {0: "Odorless", 1: "Low", 2: "Medium", 3: "High"},
    text_annotation: bool = True,
    abbreviation_dict: dict | None = None,
    custom_x_ticks: list[str] | None = None,
    custom_y_ticks: list[str] | None = None,
    figsize: tuple = (15, 18),
    dpi: int = 100,
    fontsize: int = 12,
    labelsize: int = 12,
    labelpad: int = 10,
    cmap: str|LinearSegmentedColormap = 'viridis',
    save_path: list[str] | None = None,
    width_ratios: list[float] = [1, 1, 1, 1, 1, 1],
    height_ratios: list[float] = [1, 1],
    hspace: float = 1
    ) -> plt.Figure:
    """
    Create a comprehensive multi-panel figure combining performance visualization.
    
    This function creates a complex figure with multiple subplots:
    - Top left: Performance heatmap across encoder-predictor combinations
    - Top right: Confusion matrix for the best model
    - Bottom: Violin plots showing external validation results
    
    Args:
        y_true (np.ndarray): True labels for confusion matrix
        y_pred (np.ndarray): Predicted labels for confusion matrix
        df (pd.DataFrame): Performance matrix for heatmap
        keller_data (pd.DataFrame): External validation dataset
        colorbar_label_1 (str): Label for heatmap colorbar
        colorbar_label_2 (str): Label for confusion matrix colorbar
        tick_labels (dict): Mapping of numeric labels to descriptive names
        text_annotation (bool): Whether to add text annotations to confusion matrix
        custom_x_ticks (list[str], optional): Custom x-axis labels for heatmap
        custom_y_ticks (list[str], optional): Custom y-axis labels for heatmap
        figsize (tuple): Figure size in inches
        dpi (int): Resolution for saved figures
        fontsize (int): Font size for labels
        labelsize (int): Font size for tick labels
        labelpad (int): Padding for axis labels
        cmap: Colormap for visualizations
        save_path (list[str], optional): Paths to save the figure
        width_ratios (list[float]): Relative widths of subplot columns
        height_ratios (list[float]): Relative heights of subplot rows
        hspace (float): Height spacing between subplots
        
    Returns:
        tuple: Matplotlib figure object
    """

    # Create figure and grid
    fig = plt.figure(figsize=figsize)

    # Create a more structured grid: 2 rows, 6 columns with different width ratios for top row
    gs = fig.add_gridspec(2, 6, height_ratios=height_ratios, 
                         width_ratios=width_ratios,
                         hspace=hspace, wspace=0.7)
    
    # Top row: Heatmap (left 3 cols) and Confusion matrix (right 2 cols)
    ax_heatmap = fig.add_subplot(gs[0, :2])
    ax_confusion = fig.add_subplot(gs[0, 3:])
    
    # --- HEATMAP ---
    im1 = ax_heatmap.imshow(df, cmap=cmap)
    
    if custom_x_ticks is None:
        base_x_ticks = list(df.columns)
        if abbreviation_dict is not None:
            custom_x_ticks = [abbreviation_dict.get(label, label) for label in base_x_ticks]
        else:
            custom_x_ticks = [label.replace('Encoder', '') for label in base_x_ticks]
    ax_heatmap.set_xticks(range(len(custom_x_ticks)))
    ax_heatmap.set_xticklabels(custom_x_ticks, rotation=45, ha='right')
    if custom_y_ticks is None:
        base_y_ticks = list(df.index)
        if abbreviation_dict is not None:
            custom_y_ticks = [abbreviation_dict.get(label, label) for label in base_y_ticks]
        else:
            custom_y_ticks = [label.split('Predictor')[0] for label in base_y_ticks]
    ax_heatmap.set_yticks(range(len(df.index)))
    ax_heatmap.set_yticklabels(custom_y_ticks)
    ax_heatmap.tick_params(axis='both', which='major', labelsize=labelsize)
    
    # --- CONFUSION MATRIX ---
    y_pred_rounded = y_pred.round().astype(int)
    y_true_rounded = y_true.round().astype(int)
    max_tick_value = max(tick_labels.keys())
    min_tick_value = min(tick_labels.keys())
    y_pred_rounded[y_pred_rounded > max_tick_value], y_pred_rounded[y_pred_rounded < min_tick_value] = max_tick_value, min_tick_value
    y_true_rounded[y_true_rounded > max_tick_value], y_true_rounded[y_true_rounded < min_tick_value] = max_tick_value, min_tick_value
    confusion = confusion_matrix(y_true_rounded, y_pred_rounded)
    confusion = confusion.T
    normed_confusion_matrix = confusion / np.sum(confusion, axis=0)
    
    im2 = ax_confusion.imshow(normed_confusion_matrix, cmap=cmap, origin='lower')
    if text_annotation:
        for i in range(normed_confusion_matrix.shape[0]):
            for j in range(normed_confusion_matrix.shape[1]):
                text = ax_confusion.text(
                    j, i, f"{normed_confusion_matrix[i, j]:.2f}",
                    ha='center', va='center',
                    color='white'
                )
    
    ax_confusion.set_xticks(list(tick_labels.keys()))
    ax_confusion.set_xticklabels(list(tick_labels.values()))
    ax_confusion.set_yticks(list(tick_labels.keys()))
    ax_confusion.set_yticklabels(list(tick_labels.values()))
    ax_confusion.tick_params(axis='both', which='major', labelsize=labelsize)
    ax_confusion.set_xlabel('Test Set', fontsize=fontsize, labelpad=labelpad)
    ax_confusion.set_ylabel('Prediction', fontsize=fontsize, labelpad=labelpad)
    
    cax1 = ax_heatmap.inset_axes([1.05, 0, 0.04, 1])  # relative to the parent Axes
    colorbar1 = plt.colorbar(im1, cax=cax1) 
    colorbar1.set_label(colorbar_label_1, fontsize=fontsize, labelpad=labelpad+labelpad*0.2)    # Manually adjust subplot positions to ensure equal visual height
    colorbar1.ax.tick_params(labelsize=labelsize)

    cax2 = ax_confusion.inset_axes([1.05, 0, 0.05, 1])  # relative to the parent Axes
    colorbar2 = plt.colorbar(im2, cax=cax2) 
    colorbar2.set_label(colorbar_label_2, fontsize=fontsize, labelpad=labelpad+labelpad*0.2)    # Manually adjust subplot positions to ensure equal visual height
    colorbar2.ax.tick_params(labelsize=labelsize)

    pos1 = ax_heatmap.get_position()
    pos2 = ax_confusion.get_position()
    
    ax_heatmap.set_position([pos1.x0, 1 - pos1.height, pos1.width, pos1.height])
    ax_confusion.set_position([pos2.x0, 1 - pos2.height, pos2.width, pos2.height])
    
    ax_heatmap.text(-0.55, 1.08, 'a', transform=ax_heatmap.transAxes, 
            fontsize=fontsize+2, fontweight='bold', ha='right', va='top')
    ax_confusion.text(-0.35, 1.06, 'b', transform=ax_confusion.transAxes, 
        fontsize=fontsize+2, fontweight='bold', ha='right', va='top')

    violin_axes = [
        fig.add_subplot(gs[1, :3]),  # Row 1, cols 0-2
        fig.add_subplot(gs[1, 3:]),  # Row 1, cols 3-5
    ]
    
    violin_labels = ['c', 'd']
    dilutions = [1.e-03, 1.e-05]
    
    colors_4 = {
        0.0: okabe_ito[2],
        1.0: okabe_ito[3],
        2.0: okabe_ito[1],
        3.0: okabe_ito[7]
        }
    
    for i, ax_violin in enumerate(violin_axes):
        if i == 0:        
            subset = keller_data[keller_data['Odor dilution'] == dilutions[i]]
            print(f'Number of compounds:', subset['canonical_smiles'].nunique())
            sns.violinplot(data=subset, x='predicted_intensity_rounded', y='HOW STRONG IS THE SMELL?', 
                        ax=ax_violin, hue='predicted_intensity_rounded', legend=False, palette=colors_4,
                        cut=0,
                        density_norm='area',
                        common_norm=True,
                        order=[0.0, 1.0, 2.0, 3.0]
                        )
            
            
            ax_violin.set_xlim(-0.5, 3.5)
            if i % 2 == 0:
                ax_violin.set_ylabel('Rated Intensity', fontsize=fontsize)
            else:
                ax_violin.set_ylabel('')
                ax_violin.tick_params(labelleft=False)
            ax_violin.set_xlabel('Predicted Odor Strength', fontsize=fontsize)
            ax_violin.tick_params(axis='both', which='major', labelsize=labelsize)
            ax_violin.set_xticks(list(tick_labels.keys()))
            ax_violin.set_xticklabels(list(tick_labels.values()))
        if i % 2 == 0:
            text_x = -0.16
        else:
            text_x = -0.03
        ax_violin.text(text_x, 1.1, violin_labels[i], transform=ax_violin.transAxes, 
                      fontsize=fontsize+2, fontweight='bold', ha='right', va='top')
    violin_positions = [ax.get_position() for ax in violin_axes]
    y_shift = 0.16 
    for i, ax_violin in enumerate(violin_axes):
        pos = violin_positions[i]
        if i % 2 == 0:
            new_width = pos.width * 1.15  
            new_x = pos.x0 - 0.07 
        else:
            new_width = pos.width * 1.15
            new_x = pos.x0 -0.02 
        
        if i < 2: 
            new_pos = [new_x, pos.y0 + y_shift, new_width, pos.height]
        else: 
            new_pos = [new_x, pos.y0 + 1.37*y_shift, new_width, pos.height]
        ax_violin.set_position(new_pos)  
    plt.tight_layout()
    if save_path is not None:
        for path in save_path:
            plt.savefig(path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    return fig

colors = [okabe_ito[1], okabe_ito[3]]
mycmap = LinearSegmentedColormap.from_list("mycmap", colors, N=256)

combined_fig = plot_combined_figure_with_violins(
    y_true=y_test_stack,
    y_pred=preds,
    df=-df_direct.dropna(how='all').iloc[:5, 1:],
    keller_data=keller_2016_test,
    colorbar_label_1=' Negative Macro MSE',
    colorbar_label_2='Ratio of Test Categories',
    tick_labels={0: "Odorless", 1: "Low", 2: "Medium", 3: "High"},
    text_annotation=False,
    abbreviation_dict=abbreviation_dict,
    figsize=(FIGURE_WIDTH_LONG*1.07, FIGURE_WIDTH_LONG * 0.78*1.07),
    dpi=DPI,
    fontsize=FONTSIZE,
    labelsize=LABELSIZE,
    labelpad=LABELPAD,
    cmap=mycmap,
    save_path=['figures/combined_performance_plots.pdf', 'figures/combined_performance_plots.svg', 'figures/combined_performance_plots.png'],
    width_ratios=[1, 1, 0.4, 1, 1, 0.4],
    height_ratios=[1, 0.85],
    hspace=0.4
)


# In[ ]:


# Violin plot combining Keller ratings across selected dilution levels in one 1x2 figure
colors_4 = {
    0.0: okabe_ito[2],
    1.0: okabe_ito[3],
    2.0: okabe_ito[1],
    3.0: okabe_ito[7],
}

dilution_sets = [
    [1.e-03, 1.e-05, 1.e-07],
    [1.e-01],
]
panel_labels = ['a', 'b']

fig, axes = plt.subplots(
    1, 2,
    figsize=(FIGURE_WIDTH_LONG, FIGURE_WIDTH_LONG * 0.38),
    sharex=True,
    sharey=True,
    constrained_layout=True,
 )

for i, (ax, dilutions) in enumerate(zip(axes, dilution_sets)):
    keller_dilutions = keller_2016_test.dropna(subset=[
        'Odor dilution',
        'HOW STRONG IS THE SMELL?',
        'predicted_intensity_rounded',
    ])
    keller_dilutions = keller_dilutions[keller_dilutions['Odor dilution'].isin(dilutions)]

    sns.violinplot(
        data=keller_dilutions,
        x='predicted_intensity_rounded',
        y='HOW STRONG IS THE SMELL?',
        ax=ax,
        hue='predicted_intensity_rounded',
        legend=False,
        palette=colors_4,
        cut=0,
        density_norm='area',
        common_norm=True,
        order=[0.0, 1.0, 2.0, 3.0],
    )

    ax.set_xlim(-0.5, 3.5)
    ax.set_ylabel('Rated Intensity', fontsize=FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=LABELSIZE)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(['Odorless', 'Low', 'Medium', 'High'])
    ax.set_xlabel('Predicted Odor Strength', fontsize=FONTSIZE)
    if i % 2 == 0:
        text_x = -0.16
    else:
        text_x = -0.03
    ax.text(
        text_x, 1.1, panel_labels[i],
        transform=ax.transAxes,
        ha='left',
        va='top',
        fontsize=FONTSIZE,
        fontweight='bold',
    )

plt.savefig('figures/keller_violin_dilutions_combined.pdf', dpi=DPI)
plt.savefig('figures/keller_violin_dilutions_combined.png', dpi=DPI)
plt.show()


# ## SHAP Feature Importance Analysis

# In[ ]:


import shap
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from utility.colors import okabe_ito
import matplotlib.pyplot as plt


colors = [okabe_ito[1], okabe_ito[0], okabe_ito[3]]
mycmap = LinearSegmentedColormap.from_list("mycmap", colors, N=256)


# ### SHAP Analysis Setup
# 
# Setting up SHAP (SHapley Additive exPlanations) analysis to understand feature importance and model interpretability for the best-performing odor strength prediction model.

# In[ ]:


if best_hyperparameters_direct['predictor_name'] in ['ChemPropPredictor', 'ChemeleonPredictor']:
    encoded_X_train = direct_model.models[0].odor_strength_module.odor_strength_predictor.encode(X_train)
    encoded_X_test = direct_model.models[0].odor_strength_module.odor_strength_predictor.encode(X_test)
    encoded_X = direct_model.models[0].odor_strength_module.odor_strength_predictor.encode(X)
else:
    encoded_X_train = direct_model.models[0].odor_strength_module.molecule_encoder.encode(X_train)
    encoded_X_test = direct_model.models[0].odor_strength_module.molecule_encoder.encode(X_test)
    encoded_X = direct_model.models[0].odor_strength_module.molecule_encoder.encode(X)
feature_names = encoded_X_test.columns if hasattr(encoded_X_test, 'columns') else [f'Feature {i}' for i in range(encoded_X_test.shape[1])]


# In[ ]:


# Check the shape and type of encoded data
print("Encoded X_train shape:", encoded_X_train.shape)
print("Encoded X_train type:", type(encoded_X_train))
print("Encoded X_test shape:", encoded_X_test.shape)

# Convert to numpy arrays if they're pandas DataFrames

if hasattr(encoded_X_train, 'values'):
    encoded_X_train_np = encoded_X_train.values
    encoded_X_test_np = encoded_X_test.values
else:
    encoded_X_train_np = encoded_X_train
    encoded_X_test_np = encoded_X_test

print("Final shapes - Train:", encoded_X_train_np.shape, "Test:", encoded_X_test_np.shape)


# In[ ]:


# Create a wrapper function for the predictor that works with encoded features
def predictor_wrapper(encoded_features: np.ndarray | pd.DataFrame) -> np.ndarray:
    """
    Wrapper function for SHAP analysis that takes encoded molecular features as input.
    
    This wrapper is necessary for SHAP to properly interface with the predictor
    component of the odor strength module, bypassing the encoding step.
    
    Args:
        encoded_features (np.ndarray or pd.DataFrame): Pre-encoded molecular features
        
    Returns:
        np.ndarray: Predicted odor strength values
    """
    predictions_list = []
    if hasattr(direct_model, 'models'):
        for model in direct_model.models:
            if model.predictor.__name__ in ['ChemPropPredictor', 'ChemeleonPredictor']:
                predictions = model.odor_strength_module.odor_strength_predictor.predict_from_fingerprint(encoded_features.values if hasattr(encoded_features, 'values') else encoded_features)
            else:
                predictions = model.odor_strength_module.odor_strength_predictor.predict(encoded_features.values if hasattr(encoded_features, 'values') else encoded_features)
            predictions_list.append(predictions)
        predictions = np.mean(predictions_list, axis=0)
    else:
        if direct_model.predictor.__name__ in ['ChemPropPredictor', 'ChemeleonPredictor']:
            predictions = direct_model.odor_strength_module.odor_strength_predictor.predict_from_fingerprint(encoded_features.values if hasattr(encoded_features, 'values') else encoded_features)
        else:
            predictions = direct_model.odor_strength_module.odor_strength_predictor.predict(encoded_features.values if hasattr(encoded_features, 'values') else encoded_features)
    return predictions

# Test the wrapper
test_pred = predictor_wrapper(encoded_X_train_np[:5])
print("Test prediction shape:", test_pred.shape)
print("Test predictions:", test_pred.flatten())


# In[ ]:


# Model-agnostic SHAP analysis using encoded features

# Use a subset of training data as background for faster computation
background_sample_size = encoded_X_train.shape[0]
test_sample_size = encoded_X_test.shape[0]

# Create the SHAP explainer using the predictor wrapper
explainer = shap.Explainer(predictor_wrapper, encoded_X_train_np[:background_sample_size], feature_names=feature_names, max_evals=4100)

shap_values = explainer(encoded_X_test_np[:test_sample_size])


# ### Dealing with Feature Correlation

# In[ ]:


# correlation matrix
FIGURE_WIDTH_LONG = 17.1 / 2.54

colors = [okabe_ito[1], okabe_ito[0], okabe_ito[3]]
mycmap_black = LinearSegmentedColormap.from_list("mycmap", colors, N=256)

encoded_X_df = pd.DataFrame(encoded_X, columns=feature_names)
correlation_matrix = encoded_X_df.corr().fillna(0)

fig = plt.figure(figsize=(FIGURE_WIDTH_LONG, FIGURE_WIDTH_LONG))
plt.imshow(correlation_matrix, cmap=mycmap_black, vmin=-1, vmax=1)
plt.colorbar(label='Correlation Coefficient', shrink=0.73)
plt.xticks(ticks=np.arange(len(feature_names)), labels=feature_names, rotation=90, fontsize=2)
plt.yticks(ticks=np.arange(len(feature_names)), labels=feature_names, fontsize=2)
# plt.title('Feature Correlation Matrix', fontsize=10)
plt.tight_layout()
print(fig.get_size_inches()*2.54)
plt.savefig('figures/feature_correlation_matrix.pdf', dpi=DPI)
plt.savefig('figures/feature_correlation_matrix.svg')
# plt.tick_params(labelbottom=False, labelleft=False)
# plt.xticks(ticks=[])
# plt.yticks(ticks=[])
plt.savefig('figures/feature_correlation_matrix.png', dpi=DPI)
plt.show()


# #### Agglomerative Clustering

# In[ ]:


# Improved feature grouping using agglomerative clustering
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt


# In[ ]:


# Sweep correlation thresholds and compute silhouette scores
from sklearn.metrics import silhouette_score

thresholds = np.round(np.arange(0.65, 1, 0.01), 2)
silhouette_scores = []

correlation_matrix = pd.DataFrame(encoded_X_train_np).corr().fillna(0)
distance_base = 1 - np.abs(correlation_matrix.values)
np.fill_diagonal(distance_base, 0)

for correlation_threshold in thresholds:
    distance_threshold = 1 - correlation_threshold
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        linkage='average',
        metric='precomputed',
    )
    cluster_labels = clustering.fit_predict(distance_base)
    n_clusters = len(np.unique(cluster_labels))
    if n_clusters < 2:
        silhouette_scores.append(np.nan)
        continue
    try:
        score = silhouette_score(distance_base, cluster_labels, metric='precomputed')
    except:
        print(correlation_threshold, n_clusters)
    silhouette_scores.append(score)

df_silhouette = pd.DataFrame({
    'correlation_threshold': thresholds,
    'silhouette_score': silhouette_scores,
})
df_silhouette

plt.figure(figsize=(FIGURE_WIDTH, FIGURE_WIDTH * 0.6))
plt.plot(df_silhouette['correlation_threshold'], df_silhouette['silhouette_score'], c=okabe_ito[1])
plt.xlabel('Correlation Threshold', fontsize=FONTSIZE, labelpad=LABELPAD)
plt.ylabel('Silhouette Score', fontsize=FONTSIZE, labelpad=LABELPAD)
# plt.xticks(df_silhouette['correlation_threshold'], rotation=45)
plt.tick_params(axis='both', which='major', labelsize=LABELSIZE)
plt.tight_layout()
plt.savefig('figures/rdkit_descriptors_agglomerative_thresholds.png', dpi=DPI)
plt.savefig('figures/rdkit_descriptors_agglomerative_thresholds.pdf', dpi=DPI)
plt.show()

max_silhouette = df_silhouette['silhouette_score'].max()
max_silhouette_idx = df_silhouette['silhouette_score'].idxmax()
max_threshold = df_silhouette.loc[max_silhouette_idx, 'correlation_threshold']
print(f'Max silhouette score {max_silhouette} at correlation threshold {max_threshold}')
df_silhouette.sort_values('silhouette_score', ascending=False).head()


# In[ ]:


# Agglomerative clustering

correlation_matrix = pd.DataFrame(encoded_X_train_np).corr()
feature_variances = np.var(encoded_X_train_np, axis=0)
distance_matrix = 1 - np.abs(correlation_matrix.fillna(0))
correlation_threshold = 0.75
distance_threshold = 1 - correlation_threshold

clustering = AgglomerativeClustering(
    n_clusters=None, 
    distance_threshold=distance_threshold,
    linkage='average', 
    metric='precomputed'
)

# Fit the clustering
cluster_labels = clustering.fit_predict(distance_matrix)

print(f"Agglomerative clustering created {clustering.n_clusters_} clusters")

# Create feature groups based on cluster labels
feature_groups = []
for cluster_id in range(clustering.n_clusters_):
    cluster_features = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
    feature_groups.append(cluster_features)

# Select representative features for each group (highest variance)
selected_features_idx = []
selected_features_names = []
cluster_info = []

for group_id, feature_indices in enumerate(feature_groups):
    if len(feature_indices) > 1:
        # Multiple features in group - select the one with highest variance
        group_variances = [(idx, feature_names[idx], feature_variances[idx]) for idx in feature_indices]
        best_feature = max(group_variances, key=lambda x: x[2])
        selected_features_idx.append(best_feature[0])
        selected_features_names.append(best_feature[1])
        cluster_info.append({
            'cluster_id': group_id,
            'representative': best_feature[1],
            'members': [feature_names[idx] for idx in feature_indices],
            'size': len(feature_indices)
        })
    else:
        idx = feature_indices[0]
        name = feature_names[idx]
        selected_features_idx.append(idx)
        selected_features_names.append(name)
        cluster_info.append({
            'cluster_id': group_id,
            'representative': name,
            'members': [name],
            'size': 1
        })

print(f"Agglomerative clustering: Selected {len(selected_features_idx)} representative features")
print(f"Reduced from {len(feature_names)} to {len(selected_features_idx)} features ({100*(1-len(selected_features_idx)/len(feature_names)):.1f}% reduction)")

multi_member_groups = [c for c in cluster_info if c['size'] > 1]
print(f"\nAgglomerative clustering - Groups with multiple correlated features ({len(multi_member_groups)} groups):")
for group in sorted(multi_member_groups, key=lambda x: x['size'], reverse=True)[:15]:
    print(f"  Cluster {group['cluster_id']} ({group['size']} features): {group['representative']} represents {group['members']}{'...' if len(group['members']) > 5 else ''}")

print(f"\nComparison:")
print(f"Agglomerative: {len(selected_features_idx)} representative features, {len(multi_member_groups)} multi-member groups")

# Show cluster size distributions
agg_sizes = [c['size'] for c in cluster_info if c['size'] > 1]

print(f"\nCluster size statistics:")
print(f"Agglomerative - Max: {max(agg_sizes) if agg_sizes else 0}, Mean: {np.mean(agg_sizes) if agg_sizes else 0:.1f}, Median: {np.median(agg_sizes) if agg_sizes else 0:.1f}")


# In[ ]:


groups_dict = { # representative: group
    'TPSA': 'Polarity',
    'BertzCT': 'Weight and Shape',
    'Ipc': 'Nitrogen-Polarity',
    'VSA_EState3': 'Alcohol Groups',
    'FpDensityMorgan3': 'Morgan FP Density'
    }


# ### Global SHAP feature importance

# In[ ]:


import matplotlib.pyplot as plt
from utility.colors import okabe_ito
import matplotlib as mpl

bar_color = okabe_ito[3]

clustered_shap_values = np.zeros((shap_values.values.shape[0], len(feature_groups)))
for cluster_id, feature_indices in enumerate(feature_groups):
    clustered_shap_values[:, cluster_id] = np.sum(np.abs(shap_values.values[:, feature_indices]), axis=1)
clustered_shap = shap.Explanation(values=clustered_shap_values,
                                #  base_values=clustered_shap_values.mean(axis=0),
                                base_values=np.array([0.0]*len(feature_groups)),
                                 data=None,
                                 feature_names=selected_features_names)

fig = plt.figure(figsize=(FIGURE_WIDTH*1.1, FIGURE_WIDTH*1.2))
shap.plots.bar(clustered_shap,
                    show=False,
                    max_display=6,
                    )
ax = plt.gca()
for txt in list(ax.texts):
    try:
        txt.remove()
    except Exception:
        pass

patches = [p for p in ax.patches if isinstance(p, mpl.patches.Rectangle) and p.get_width() > 0]

num_bars = len(patches)
if num_bars > 0:
    for i, p in enumerate(patches):
        p.set_facecolor(bar_color)


y_tick_locs = ax.get_yticks()
y_tick_labels = [t.get_text() for t in ax.get_yticklabels()]
# if not any('feature' in name.lower() for name in y_tick_labels):
# For RDKit Descriptor names in study
new_y_tick_labels = [groups_dict.get(name, name) for name in y_tick_labels[:len(y_tick_labels) // 2] if not 'feature' in name.lower()] + [f'{131 - len(y_tick_labels)//2 + 1} other Groups']
if len(new_y_tick_labels) == len(y_tick_labels) // 2:
    ax.set_yticklabels(2*new_y_tick_labels)


current_size = fig.get_size_inches()
new_width = FIGURE_WIDTH
scale_factor = new_width / current_size[0]
new_height = current_size[1] * scale_factor * 1.25
print(fig.get_size_inches()*2.54)
fig.set_size_inches(new_width, new_height)
print(fig.get_size_inches()*2.54)
ax.set_xlabel('Sum absolute SHAP values', fontsize=FONTSIZE, labelpad=LABELPAD)
ax.tick_params(axis='both', which='major', labelsize=LABELSIZE)
ax.margins(y=0.05)  # Add some vertical margins for better spacing

# Add frame around the plot
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_linewidth(1)
    spine.set_edgecolor('black')


plt.tight_layout()
plt.savefig('figures/shap_feature_importance.pdf', dpi=DPI)
plt.savefig('figures/shap_feature_importance.svg')
plt.savefig('figures/shap_feature_importance.png', dpi=DPI)
plt.show()



df_clustered_shap_values = pd.DataFrame(clustered_shap_values, columns=selected_features_names)


# In[ ]:


# Import matplotlib
import numpy as np
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
colors = [okabe_ito[3], okabe_ito[1]]
mycmap = LinearSegmentedColormap.from_list("mycmap", colors, N=256)


def plot_summary_plot_feature_group(most_important_feature: str) -> None:
    """
    Create a SHAP summary plot for the most important feature group.
    
    This function generates a beeswarm plot showing SHAP values for all features
    within the most important feature groups, colored by feature values.
    
    Args:
        most_important_feature (str): Name of the representative feature for the cluster
        
    Returns:
        tuple: Representative feature name and list of cluster feature names
    """
    most_important_cluster = next(c for c in cluster_info if c['representative'] == most_important_feature)

    print(f"Most important feature group:")
    print(f"Representative feature: {most_important_cluster['representative']}")
    print(f"Cluster size: {most_important_cluster['size']} features")
    print(f"All features in this cluster: {most_important_cluster['members']}")
    most_important_cluster_indices = [i for i, name in enumerate(feature_names) if name in most_important_cluster['members']]
    cluster_shap_values = shap_values.values[:, most_important_cluster_indices]
    cluster_feature_values = encoded_X_test_np[:test_sample_size, most_important_cluster_indices]
    cluster_feature_names = [feature_names[i] for i in most_important_cluster_indices]
    cluster_shap_explanation = shap.Explanation(
        values=cluster_shap_values,
        base_values=shap_values.base_values,
        data=cluster_feature_values,
        feature_names=cluster_feature_names
    )

    fig =plt.figure(figsize=(FIGURE_WIDTH, FIGURE_WIDTH*0.75))
    shap.plots.beeswarm(cluster_shap_explanation,
                        max_display=25,
                        show=False)
    cmap = mpl.cm.get_cmap(mycmap)

    ax = plt.gca()
    for coll in ax.collections:
        try:
            coll.set_cmap(cmap)
        except Exception:
            pass

    # update any image artists / colorbars if present
    for im in ax.get_images():
        try:
            im.set_cmap(cmap)
        except Exception:
            pass

    # redraw figure
    plt.draw()
    updated_cbar = False
    for coll in ax.collections:
        try:
            arr = coll.get_array()
            if arr is None or len(arr) == 0:
                continue
            norm = getattr(coll, "norm", None)
            mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            mappable.set_array(arr)

            for a in fig.axes[:]:
                if a is not ax:
                    pos = a.get_position()
                    if pos.width < 0.2 or pos.height < 0.2:
                        try:
                            fig.delaxes(a)
                        except Exception:
                            pass
            cbar = fig.colorbar(mappable, ax=ax, orientation="vertical", pad=0.02)
            cbar.set_label("Feature Value", rotation=270, labelpad=12, fontsize=FONTSIZE)
            # remove colorbar tick labels
            try:
                cbar.set_ticks([])
                cbar.ax.set_yticklabels([])
            except Exception:
                pass
            updated_cbar = True
            break
        except Exception:
            continue

    ax.set_xlabel('SHAP Value (Impact on Model Output)', fontsize=FONTSIZE, labelpad=LABELPAD)
    ax.tick_params(axis='both', which='major', labelsize=LABELSIZE)
    max_height = 21 / 2.54
    current_width, current_height = fig.get_size_inches()
    if current_height > max_height:
        fig.set_size_inches(current_width, max_height, forward=True)
    plt.tight_layout()
    plt.savefig(f'figures/shap_summary_plot_{most_important_cluster["representative"]}_cluster.pdf', dpi=DPI)
    plt.show()
    return most_important_feature, cluster_feature_names

important_features = []
representative_features = []

for representative_feature in y_tick_labels[:5]:
    rep_feature, cluster_features = plot_summary_plot_feature_group(representative_feature)
    representative_features.append(rep_feature)
    important_features.extend(cluster_features)


# In[ ]:


df_clusterd_shap_values_renamed = df_clustered_shap_values.rename(columns=lambda x: groups_dict.get(x, x))

category_colors = [okabe_ito[2], okabe_ito[3], okabe_ito[1], okabe_ito[7]]

preds_rounded = np.round(direct_model.predict(X_test)).astype(int)


# In[ ]:


# Create 2x2 subplot for each category showing mean SHAP values
fig, axes = plt.subplots(2, 2, figsize=(FIGURE_WIDTH_LONG, FIGURE_WIDTH*1.5))
axes = axes.flatten()

subplot_labels = ['a', 'b', 'c', 'd']
for i, category in enumerate([0, 1, 2, 3]):
    category_mask = preds_rounded == category
    category_shap_data = df_clusterd_shap_values_renamed[category_mask]
    # Check if all values from groups_dict are present in the columns
    if all(col in category_shap_data.columns for col in list(groups_dict.values())):
        mean_shap_values = category_shap_data[list(groups_dict.values())].mean(axis=0)
        print('mean_shap_values:', mean_shap_values)
    else:
        mean_shap_values = category_shap_data.iloc[:,:5].mean(axis=0)
    bars = axes[i].barh(range(len(mean_shap_values)), mean_shap_values.values, color=category_colors[i], alpha=0.7)
    axes[i].set_xlim(0, 0.5)
    if i % 2 == 0:
        axes[i].set_yticks(range(len(mean_shap_values)))
        axes[i].set_yticklabels(mean_shap_values.index, fontsize=LABELSIZE)
    else:
        axes[i].set_yticks(range(len(mean_shap_values)))
        axes[i].set_yticklabels([])
    if i > 1:
        axes[i].set_xlabel('Mean Absolute SHAP Value', fontsize=FONTSIZE)
        axes[i].tick_params(axis='x', which='major', labelsize=LABELSIZE)
    else:
        axes[i].set_xticklabels([])
    axes[i].invert_yaxis()
    if i % 2 == 0:
        text_x = -0.4
    else:
        text_x = -0.03
    axes[i].text(text_x, 1.1, f'{subplot_labels[i]}', transform=axes[i].transAxes, 
        fontsize=FONTSIZE+2, fontweight='bold', ha='right', va='top')

plt.tight_layout()
plt.savefig('figures/shap_feature_importance_by_category.pdf', dpi=DPI)
plt.savefig('figures/shap_feature_importance_by_category.svg')
plt.savefig('figures/shap_feature_importance_by_category.png', dpi=DPI)
plt.show()


# ### Local Explanations

# In[ ]:


def local_explain(smiles: str, background: pd.DataFrame | np.ndarray, save_path: str | None = None, dpi: int = DPI) -> None:
    """
    Generate and visualize local SHAP explanations for individual molecules.
    
    This function creates a waterfall plot showing how each feature group contributes
    to the prediction for a specific molecule, starting from the base value.
    
    Args:
        smiles (str): SMILES string of the molecule to explain
        background (pd.DataFrame or np.ndarray): Background dataset for SHAP explainer
        save_path (str, optional): Path to save the visualization
        dpi (int): Resolution for saved figures
    """
    max_display = 6
    if best_hyperparameters_direct['predictor_name'] in ['ChemPropPredictor', 'ChemeleonPredictor']:
        encoded_mol = direct_model.models[0].odor_strength_module.odor_strength_predictor.encode(2*[smiles])[0:1]
        print(encoded_mol.shape)
    else:
        encoded_mol = direct_model.models[0].odor_strength_module.molecule_encoder.encode([smiles])
    encoded_mol_np = encoded_mol.values if hasattr(encoded_mol, 'values') else encoded_mol
    background_values = background.values if hasattr(background, 'values') else background

    explainer = shap.Explainer(predictor_wrapper, background_values, feature_names=background.columns.tolist() if hasattr(background, 'columns') else None, max_evals=4100)
    shap_values_local = explainer(encoded_mol_np)
    clustered_shap_values = np.zeros((shap_values_local.values.shape[0], len(feature_groups)))
    for cluster_id, feature_indices in enumerate(feature_groups):
        clustered_shap_values[:, cluster_id] = np.sum(shap_values_local.values[:, feature_indices], axis=1)
    selected_clusters = pd.DataFrame(clustered_shap_values, columns=selected_features_names)
    selected_clusters = selected_clusters[representative_features]
    selected_clusters.columns = [groups_dict.get(col, col) for col in selected_clusters.columns]
    other_clusters =  clustered_shap_values.sum() - selected_clusters.values.sum()
    clustered_shap_values = selected_clusters.values
    
    shap_values_clustered = shap.Explanation(values=clustered_shap_values,
                                    base_values=shap_values_local.base_values + other_clusters,
                                    data=None,
                                    feature_names=selected_clusters.columns.tolist())


    print(f"SHAP values for molecule: {smiles}")
    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_WIDTH*0.75))
    shap.plots.waterfall(shap_values_clustered[0], max_display=max_display, show=False)
    for ax in fig.get_axes():
        for txt in list(ax.texts):
            txt.set_visible(False)
        for child in ax.get_children():
            if hasattr(child, 'get_text'):
                try:
                    child.set_visible(False)
                except:
                    pass
            elif hasattr(child, '__class__') and 'Arrow' in str(child.__class__):
                color = child.get_edgecolor()
                if color[0] == 1:
                    child.set_color(okabe_ito[3])
                else:
                    child.set_color(okabe_ito[1])
                pass
        
        ax.set_title('')
        ax.set_xlabel('')
        ax.set_ylabel('')
    
    for ax in fig.get_axes():
        plt.setp(ax.get_xticklabels(), fontsize=LABELSIZE*2.25)
        plt.setp(ax.get_yticklabels(), fontsize=LABELSIZE*2.25)
        
        ax.tick_params(axis='both', which='major', labelsize=LABELSIZE*2.25)
    
    if save_path is not None:
        plt.savefig(save_path, dpi=DPI)
    
    factor = 2
    for ax in fig.get_axes():
        plt.setp(ax.get_xticklabels(), fontsize=factor*LABELSIZE)
        plt.setp(ax.get_yticklabels(), fontsize=factor*LABELSIZE)
        ax.tick_params(axis='both', which='major', labelsize=factor*LABELSIZE)
    plt.tight_layout()
    plt.savefig(save_path.split('.')[0] + '.png', dpi=DPI)
    plt.show()
    

local_explain('OCCO', background=encoded_X_train, save_path='figures/shap_local_explanation_ethylene_glycol_group.svg', dpi=None)
local_explain('CC(C)=CCC/C(C)=C\CC/C(C)=C/CO', background=encoded_X_train, save_path='figures/shap_local_explanation_ez_farnesol_group.svg', dpi=None)
local_explain('CC1COCc2cc3c(cc21)C(C)(C)C(C)C3(C)C	', background=encoded_X_train, save_path='figures/shap_local_explanation_galaxolide_group.svg', dpi=None)
local_explain('c1ccc2[nH]ccc2c1', background=encoded_X_train, save_path='figures/shap_local_explanation_indole_group.svg', dpi=None)

