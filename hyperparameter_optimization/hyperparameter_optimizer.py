from typing import Callable
import optuna
from optuna.exceptions import TrialPruned
import numpy as np

class HyperparameterOptimizer:
    """A class for optimizing hyperparameters using Optuna.
    
    Parameters:
        model: The machine learning model class to be optimized.
        study: An Optuna study object for managing the optimization process.
    
    Methods:
        objective(trial, X, y, hyperparameter_space, evaluation_function, n_repeats=1, pruner_tolerance=0):
            Defines the objective function for the optimization.
        optimize(X, y, hyperparameter_space, evaluation_function, n_trials=100, n_repeats=1, pruner_tolerance=0):
            Runs the optimization process.
        get_best_hyperparameters():
            Returns the best hyperparameters found during optimization.
        get_best_model_unfitted():
            Returns the best model with the best hyperparameters but not fitted to data.
        get_best_model(X, y):
            Returns the best model fitted to the provided data.
    
    Attributes:
        model: The machine learning model class to be optimized.
        study: An Optuna study object for managing the optimization process.

    """
    def __init__(self, model: object, study: optuna.study.Study):
        self.model = model
        self.study = study

    def objective(self, trial, X, y, hyperparameter_space: Callable, evaluation_function: Callable, n_repeats: int = 1, pruner_tolerance: float = 0):
        """Defines the objective function for the optimization.
        
        Parameters:
            trial: An Optuna trial object.
            X: Features for the model.
            y: Target variable for the model.
            hyperparameter_space: A Callable that defines the hyperparameter search space.
            evaluation_function: A Callable that evaluates the model.
            n_repeats: Number of repetitions for the evaluation.
            pruner_tolerance: Tolerance to the best model's performance metric for not pruning trials.
            """
        hyperparameters = hyperparameter_space(trial)
        early_stopping_overwrite = None
        if hyperparameters.get('early_stopping_overwrite', False):
            early_stopping_overwrite = hyperparameters.pop('early_stopping_overwrite')
        for key_1, value_1 in hyperparameters.items():
            if isinstance(value_1, dict):
                if value_1.get('early_stopping_overwrite', False):
                    early_stopping_overwrite = value_1.pop('early_stopping_overwrite')
        for hyperparameter, value in hyperparameters.items():
            trial.set_user_attr(hyperparameter, value)
        scores = []
        best_early_stopping_rounds = []
        for i in range(n_repeats):
            score, best_early_stopping_round = evaluation_function(self.model(**hyperparameters), X, y)
            scores.append(score)
            best_early_stopping_rounds.append(best_early_stopping_round)
            mean_score = np.mean(scores)
            trial.report(mean_score, step=i)
            if trial.should_prune():
                if trial.number != 0:
                    if np.abs(self.study.best_value - mean_score) > pruner_tolerance:
                        raise TrialPruned(f"Trial pruned at repetition {i} with mean score {mean_score} and parameters {hyperparameters}")
                else:
                    raise TrialPruned(f"Trial pruned at repetition {i} with mean score {mean_score} and parameters {hyperparameters}")
        if not all(round is None for round in best_early_stopping_rounds):
            valid_rounds = [r for r in best_early_stopping_rounds if r is not None]
            best_early_stopping_round = int(np.round(np.mean(valid_rounds)))
            if early_stopping_overwrite:
                if trial.user_attrs.get(early_stopping_overwrite, False):
                    trial.set_user_attr(early_stopping_overwrite, best_early_stopping_round)
                else:
                    for key, value in trial.user_attrs.items():
                        if isinstance(value, dict):
                            if value.get(early_stopping_overwrite, False):
                                value[early_stopping_overwrite] = best_early_stopping_round + 1
                                trial.set_user_attr(key, value)
            # implement overwritting hyperparameter
        mean_score = np.mean(scores)
        return mean_score

    def optimize(self, X, y, hyperparameter_space, evaluation_function, n_trials=100, n_repeats=1, pruner_tolerance=0, **optimize_kwargs):
        self.study.optimize(lambda trial: self.objective(trial,
                                                         X,
                                                         y,
                                                         hyperparameter_space,
                                                         evaluation_function,
                                                         n_repeats=n_repeats,
                                                         pruner_tolerance=pruner_tolerance
                                                         ), 
                                            n_trials=n_trials,
                                            **optimize_kwargs
                                            )
        print('Best hyperparameters:', self.study.best_trial.user_attrs)

    def get_best_hyperparameters(self):
        return self.study.best_trial.user_attrs

    def get_best_model_unfitted(self):
        return self.model(**self.study.best_trial.user_attrs)
    
    def get_best_model(self, X, y):
        model = self.model(**self.study.best_trial.user_attrs)
        model.fit(X, y)
        return model
    