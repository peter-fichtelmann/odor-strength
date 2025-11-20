from typing import Callable
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np
from utility.metrics import Metrics
from . import molecule_encoder
from . import predictors
import pandas as pd
import json

class OdorStrengthModule(Metrics):
    """A module for predicting odor strength of molecules.
    
    Parameters:
        molecule_encoder: An instance of a molecule encoder class, which encodes a list of SMILES strings into a numerical representation.
        odor_strength_predictor: An instance of a predictor class, which predicts the odor strength based on the encoded features.
        
    Methods:
        fit(X, y, X_val=None, y_val=None):
            Fits the odor strength predictor to the provided data.
        predict(X):
            Predicts the odor strength for the provided data.
        predict_kFold(X, y, metric, n_splits=5, random_state=42, real_time_evaluation=False, groups=None, **metric_kwargs):
            Performs stratified group k-fold cross-validation and returns predictions and metric values.
        evaluate_kFold(X, y, metric, n_splits=5, n_repeats=1, random_state=None, plot=False, plot_kwargs=None, show_wrong_pred=False, real_time_evaluation=False, groups=None, **metric_kwargs):
            Evaluates the model using k-fold cross-validation and returns the mean and standard deviation of the metric values.
        scatter_plot(y_true, y_pred, dodge_true=True, dodge_pred=True, categorical=False, dpi=100, figsize=(7, 6), color='#666666', tick_labels={0: "None", 1: "Low", 2: "Medium", 3: "High"}):
            Creates a matplotlib scatter plot of true values vs predicted values.
        scatter_plotly(y_true, y_pred, X, dodge_true=True, dodge_pred=True, categorical=False, color='#666666', tick_labels={0: "None", 1: "Low", 2: "Medium", 3: "High"}):
            Creates an interactive plotly scatter plot of true values vs predicted values.
        plot_confusion_matrix(y_true, y_pred, figsize=(10, 10), dpi=100, tick_labels={0: "None", 1: "Low", 2: "Medium", 3: "High"}):
            Plots a confusion matrix for the categorized predicted values.
        visualize_wrong_predictions(X, y, y_pred, figsize=(10, 5), dpi=100):
            Visualizes wrong predictions by displaying the SMILES strings of the molecules that were misclassified. 
    """
    def __init__(self, molecule_encoder, odor_strength_predictor):
        self.molecule_encoder = molecule_encoder
        self.odor_strength_predictor = odor_strength_predictor

    def save(self, encoder_path: str, predictor_path: str):
        self.molecule_encoder.save(encoder_path)
        self.odor_strength_predictor.save(predictor_path)

    def load(self, encoder_path: str, predictor_path: str):
        self.molecule_encoder.load(encoder_path)
        self.odor_strength_predictor.load(predictor_path)

    def fit(self, X: list[str], y: np.ndarray, X_val: list[str]|None = None, y_val: np.ndarray|None = None) -> int|None:
        """Fits the odor strength predictor to the provided data.
        Parameters:
            X: List of SMILES strings representing the molecules.
            y: Numpy array of numerical odor strength values.
            X_val: List of SMILES strings for validation (optional).
            y_val: Numpy array of numerical odor strength values for validation (optional).
        Returns:
            The best early stopping round if applicable, otherwise None."""
        X_encoded = self.molecule_encoder.encode(X)
        X_val_encoded = self.molecule_encoder.encode(X_val) if X_val is not None else None
        if isinstance(X_encoded, pd.DataFrame):
            X_encoded = X_encoded.values
        if isinstance(X_val_encoded, pd.DataFrame):
            X_val_encoded = X_val_encoded.values
        if y_val is not None and X_val_encoded is not None:
            best_early_stopping_round = self.odor_strength_predictor.fit(X_encoded, y, X_val=X_val_encoded, y_val=y_val)
        else:
            best_early_stopping_round = self.odor_strength_predictor.fit(X_encoded, y)
        return best_early_stopping_round
    
    def predict(self, X: list[str]) -> np.ndarray:
        """Predicts the odor strength for the provided data.
        Parameters:
            X: List of SMILES strings representing the molecules.
        Returns:
            Numpy array of predicted odor strength values."""
        X_encoded = self.molecule_encoder.encode(X)
        if isinstance(X_encoded, pd.DataFrame):
            X_encoded = X_encoded.values
        return self.odor_strength_predictor.predict(X_encoded)
    
    def predict_kFold(self,
                      X: list[str],
                      y: np.ndarray,
                      metric: Callable | list[Callable],
                      n_splits: int = 5,
                      random_state: int | None = None,
                      real_time_evaluation: bool = False,
                      groups: np.ndarray | None = None,
                      **metric_kwargs
                      ) -> tuple[np.ndarray, list, int | None]:
        kf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        metric_values = []
        predictions_array = np.zeros(len(y))
        best_early_stopping_rounds = []
        for train_index, test_index in kf.split(X, y, groups=groups):
            X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.odor_strength_predictor.reset()
            if real_time_evaluation:
                best_early_stopping_round = self.fit(X_train, y_train, X_val=X_test, y_val=y_test)
                best_early_stopping_rounds.append(best_early_stopping_round)
            else:
                self.fit(X_train, y_train)
                best_early_stopping_round = None
            predictions = self.predict(X_test)
            if isinstance(metric, list):
                metric_value = np.zeros(len(metric))
                for i, m in enumerate(metric):
                    metric_value_i = m(y_test, predictions, **metric_kwargs)
                    # print(m.__name__, metric_value_i)
                    if isinstance(metric_value_i, tuple):
                        metric_value[i] = metric_value_i[0]
                    else:
                        metric_value[i] = metric_value_i            
            else:
                metric_value = metric(y_test, predictions, **metric_kwargs)
            if isinstance(metric_value, tuple):
                metric_value = metric_value[0]
            metric_values.append(metric_value)
            # print('mse_micro', self.calculate_mse(y_test, predictions))
            predictions_array[test_index] = predictions
        if not all(round is None for round in best_early_stopping_rounds):
            valid_rounds = [r for r in best_early_stopping_rounds if r is not None]
            best_early_stopping_round = int(np.round(np.mean(valid_rounds)))
        return predictions_array, metric_values, best_early_stopping_round
    
    def evaluate_kFold(self, 
                       X: list[str], 
                       y: np.ndarray, 
                       metric: Callable | list[Callable], 
                       n_splits: int, 
                       n_repeats: int = 1, 
                       random_state: int | None = None, 
                       plot: bool | str = False, 
                       plot_kwargs: dict | None = None, 
                       show_wrong_pred: bool = False, 
                       real_time_evaluation: bool = False, 
                       groups: np.ndarray | None = None, 
                       **metric_kwargs
                       ) -> tuple[float, float, list, int | None]:
        """
        Evaluates the model using k-fold cross-validation, optionally repeating the process multiple times, 
        and computes the specified metric(s) for each fold. Supports plotting results and visualizing wrong predictions.
        Parameters
        ----------
        X : list[str]
            List of input features.
        y : np.ndarray
            Array of target values.
        metric : Callable or list of Callable
            Metric function(s) to evaluate model performance.
        n_splits : int
            Number of folds for k-fold cross-validation.
        n_repeats : int, optional
            Number of times to repeat the k-fold cross-validation (default is 1).
        random_state : int or None, optional
            Random seed for reproducibility (default is None).
        plot : bool or str, optional
            If True or 'matplotlib', plots results using matplotlib. If 'plotly', uses plotly. Default is False.
        plot_kwargs : dict or None, optional
            Additional keyword arguments for plotting functions (default is None).
        show_wrong_pred : bool, optional
            If True, visualizes wrong predictions (default is False).
        real_time_evaluation : bool, optional
            If True, evaluates metrics in real time during cross-validation (default is False).
        groups : np.ndarray or None, optional
            Group labels for the samples used while splitting the dataset into folds (default is None).
            Additional keyword arguments passed to the metric function(s).
        Returns
        -------
        mean_metric : float
            Mean value of the metric(s) across all folds and repeats.
        std_metric : float
            Standard deviation of the metric(s) across all folds and repeats.
        metric_values_list : list
            List of metric values for each fold and repeat.
        best_early_stopping_round : int or None
            Best early stopping round if applicable, otherwise None.
        """
        metric_values_list = []
        for i in range(n_repeats):
            predictions_array, metric_values, best_early_stopping_round = self.predict_kFold(X, y, metric, n_splits=n_splits, random_state=random_state, real_time_evaluation=real_time_evaluation, groups=groups, **metric_kwargs)
            y_check = y.copy()
            y_check[y_check > 0] = 1
            predictions_array_check = predictions_array.copy()
            predictions_array_check[predictions_array_check > 0] = 1
            if plot == True or plot == 'matplotlib':
                self.scatter_plot(y, predictions_array, **(plot_kwargs if plot_kwargs else {}))
            elif plot == 'plotly':
                self.scatter_plotly(y, predictions_array, X, **(plot_kwargs if plot_kwargs else {}))
            if show_wrong_pred:
                self.visualize_wrong_predictions(X, y_check, predictions_array_check)
            if isinstance(metric, list):
                for j, m in enumerate(metric):
                    print(f'Iteration {i+1}/{n_repeats} - {m.__name__}: {np.mean([metric_value[j] for metric_value in metric_values]):.3f}, Std {m.__name__}: {np.std([metric_value[j] for metric_value in metric_values]):.3f}')
                metric_values_list += [metric_value[0] for metric_value in metric_values]
            else:
                print(f'Iteration {i+1}/{n_repeats} - {metric.__name__}: {np.mean(metric_values):.3f}, Std {metric.__name__}: {np.std(metric_values):.3f}')
                metric_values_list += metric_values
        return np.mean(metric_values_list), np.std(metric_values_list), metric_values_list, best_early_stopping_round
    
    def scatter_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dodge_true: bool = True,
        dodge_pred: bool = True,
        categorical: bool = False,
        dpi: int = 100,
        figsize: tuple[int, int] = (7, 6),
        color: str = '#666666',
        tick_labels: dict[int, str] = {0: "None", 1: "Low", 2: "Medium", 3: "High"}
        ) -> None:
        """ Creates a matplotlib scatter plot comparing true values (`y_true`) to predicted values (`y_pred`).
        This visualization helps assess the performance of a prediction model by plotting each true value against its corresponding prediction. The plot can be customized for categorical data, and jitter ("dodge") can be applied to both axes to reduce overlap.

        Parameters
        ----------
        y_true : np.ndarray
            Array of true target values.
        y_pred : np.ndarray
            Array of predicted values.
        dodge_true : bool, optional
            If True, adds jitter to the true values to reduce overlap in the plot (default is True).
        dodge_pred : bool, optional
            If True, adds jitter to the predicted values to reduce overlap in the plot (default is True).
        categorical : bool, optional
            If True, rounds values to integers for categorical plotting (default is False).
        dpi : int, optional
            Dots per inch for the figure resolution (default is 100).
        figsize : tuple[int, int], optional
            Size of the figure in inches as (width, height) (default is (7, 6)).
        color : str, optional
            Color of the scatter points (default is '#666666').
        tick_labels : dict[int, str], optional
            Mapping of tick positions to labels for categorical axes (default is {0: "None", 1: "Low", 2: "Medium", 3: "High"})..
        """
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=figsize, dpi=dpi)
        if categorical:
            y_pred = y_pred.round().astype(int)
            y_true = y_true.round().astype(int)
        if dodge_true:
            y_true = y_true + np.random.rand(y_true.shape[0])*max(y_true.max(), y_pred.max())/8 - 0.5*max(y_true.max(), y_pred.max())/8
        if dodge_pred:
            y_pred = y_pred + np.random.rand(y_pred.shape[0])*max(y_true.max(), y_pred.max())/8 - 0.5*max(y_true.max(), y_pred.max())/8
        plt.scatter(y_true, y_pred, s=1, color=color, alpha=0.5)
        plt.xlim(-0.1*max(y_true.max(), y_pred.max()), max(y_true.max(), y_pred.max()) * 1.1)
        plt.ylim(-0.1*max(y_true.max(), y_pred.max()), max(y_true.max(), y_pred.max()) * 1.1)
        plt.xticks(list(tick_labels.keys()), list(tick_labels.values()))
        plt.yticks(list(tick_labels.keys()), list(tick_labels.values()))
        plt.tick_params(axis='both', which='major', labelsize=18)
        plt.xlabel('True Values', fontsize=20, labelpad=10)
        plt.ylabel('Prediction', fontsize=20, labelpad=10)
        plt.show()

    def scatter_plotly(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        X: list[str],
        dodge_true: bool = True,
        dodge_pred: bool = True,
        categorical: bool = False,
        color: str = '#666666',
        tick_labels: dict[int, str] = {0: "None", 1: "Low", 2: "Medium", 3: "High"}
        ) -> None:
        """
        Generates a scatter plot using Plotly to visualize the relationship between true and predicted values.

        Parameters
        ----------
        y_true : np.ndarray
            Array of true target values.
        y_pred : np.ndarray
            Array of predicted target values.
        X : list[str]
            List of SMILES strings or identifiers for hover information.
        dodge_true : bool, default=True
            If True, adds random jitter to the true values to reduce overlap in the plot.
        dodge_pred : bool, default=True
            If True, adds random jitter to the predicted values to reduce overlap in the plot.
        categorical : bool, default=False
            If True, rounds and casts values to integers for categorical plotting.
        color : str, default='#666666'
            Color code for the scatter plot points.
        tick_labels : dict[int, str], default={0: "None", 1: "Low", 2: "Medium", 3: "High"}
            Mapping of tick values to their display labels for axes.
        """
        import plotly.express as px
        if categorical:
            y_pred = y_pred.round().astype(int)
            y_true = y_true.round().astype(int)
        if dodge_true:
            y_true = y_true + np.random.rand(y_true.shape[0])*max(y_true.max(), y_pred.max())/8 - 0.5*max(y_true.max(), y_pred.max())/8
        if dodge_pred:
            y_pred = y_pred + np.random.rand(y_pred.shape[0])*max(y_true.max(), y_pred.max())/8 - 0.5*max(y_true.max(), y_pred.max())/8
        fig = px.scatter(x=y_true,
                         y=y_pred,
                         color_discrete_sequence=[color],
                         hover_data={'smiles': X},)
        fig.update_layout(
            xaxis_title='True Values',
            yaxis_title='Prediction',
            xaxis=dict(tickmode='array', tickvals=list(tick_labels.keys()), ticktext=list(tick_labels.values())),
            yaxis=dict(tickmode='array', tickvals=list(tick_labels.keys()), ticktext=list(tick_labels.values())),
            width=800,
            height=600,
        )
        fig.show()


    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        figsize: tuple = (10, 10),
        dpi: int = 100,
        tick_labels: dict = {0: "None", 1: "Low", 2: "Medium", 3: "High"},
        cmap: str|LinearSegmentedColormap = 'viridis',
        text_annotation: bool = True,
        fontsize: int = 20,
        labelsize: int = 18,
        labelpad: int = 10,
        save_path: str | None = None
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
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        y_pred = y_pred.round().astype(int)
        y_true = y_true.round().astype(int)
        max_tick_value = max(tick_labels.keys())
        min_tick_value = min(tick_labels.keys())
        y_pred[y_pred > max_tick_value], y_pred[y_pred < min_tick_value] = max_tick_value, min_tick_value
        y_true[y_true > max_tick_value], y_true[y_true < min_tick_value] = max_tick_value, min_tick_value
        confusion = confusion_matrix(y_true, y_pred)
        # confusion = np.flipud(confusion.T)
        confusion = confusion.T
        normed_confusion_matrix = confusion / np.sum(confusion, axis=0)
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        im = ax.imshow(normed_confusion_matrix, cmap=cmap, aspect='equal', origin='lower')
        if text_annotation:
            for i in range(normed_confusion_matrix.shape[0]):
                for j in range(normed_confusion_matrix.shape[1]):
                    text = ax.text(
                        j, i, f"{normed_confusion_matrix[i, j]:.2f}",
                        ha='center', va='center',
                        color='white' # if normed_confusion_matrix[i, j] < 0.5 else 'black'  # adjust for contrast
                    )
        colorbar = plt.colorbar(im)
        colorbar.set_label('Ratio of correct test values', fontsize=fontsize, labelpad=labelpad+labelpad*0.2)
        colorbar.ax.tick_params(labelsize=labelsize)
        # plt.xlim(-0.1*max(y_true.max(), y_pred.max()), max(y_true.max(), y_pred.max()) * 1.1)
        # plt.ylim(-0.1*max(y_true.max(), y_pred.max()), max(y_true.max(), y_pred.max()) * 1.1)
        plt.xticks(list(tick_labels.keys()), list(tick_labels.values()))
        plt.yticks(list(tick_labels.keys()), list(tick_labels.values()))
        plt.tick_params(axis='both', which='major', labelsize=labelsize)
        plt.xlabel('Test Set', fontsize=fontsize, labelpad=labelpad)
        plt.ylabel('Prediction', fontsize=fontsize, labelpad=labelpad)
        if save_path:
            plt.savefig(save_path, dpi=dpi)
        plt.show()


    def visualize_wrong_predictions(
        self,
        X: list[str],
        y: np.ndarray,
        y_pred: np.ndarray,
        figsize: tuple[int, int] = (10, 5),
        dpi: int = 100
    ) -> None:
        """
        Visualizes molecules for which the model made incorrect predictions.
        For each molecule where the predicted label does not match the true label,
        displays its structure in a grid along with its true and predicted values.
        Parameters
        ----------
        X : list of str
            List of SMILES strings representing the molecules.
        y : np.ndarray
            Array of true labels.
        y_pred : np.ndarray
            Array of predicted labels.
        figsize : tuple of int, optional
            Size of the matplotlib figure (default is (10, 5)).
        dpi : int, optional
            Dots per inch for the figure (default is 100).
        Returns
        -------
        None
            Displays the grid image of wrongly predicted molecules.
        """
        import matplotlib.pyplot as plt
        from rdkit import Chem
        from rdkit.Chem import Draw
        y_pred = y_pred.round().astype(int)
        wrong_indices = np.where(y != y_pred)[0]
        if len(wrong_indices) == 0:
            print("No wrong predictions found.")
            return
        selected_smiles = [X[i] for i in wrong_indices]
        selected_y = y[wrong_indices]
        selected_y_pred = y_pred[wrong_indices]
        
        mols = [Chem.MolFromSmiles(s) for s in selected_smiles]
        img = Draw.MolsToGridImage(mols, molsPerRow=5, maxMols=100, subImgSize=(400, 400),
                                   legends=[f"True: {true}, Pred: {pred}" for true, pred in zip(selected_y, selected_y_pred)])
        display(img)


class OdorStrengthModuleHyperparameterOptimizationWrapper:
    """
    Wrapper class for hyperparameter optimization of the OdorStrengthModule.
    This class facilitates the instantiation and usage of the OdorStrengthModule with
    specified molecule encoder and odor strength predictor configurations. It provides
    methods for model training, prediction, evaluation, and visualization.
    Parameters:
        encoder_name (str): Name of the molecule encoder class to use.
        predictor_name (str): Name of the odor strength predictor class to use.
        hp_molecule_encoder (dict): Hyperparameters for the molecule encoder.
        hp_odor_strength_predictor (dict): Hyperparameters for the odor strength predictor.
    Attributes:
        odor_strength_module (OdorStrengthModule): The underlying odor strength module instance.
        dealing_with_incosistency (str): Strategy for handling inconsistencies if binarized labels are used in an ordinal regression task, default is 'sum'.
    Methods:
        predict_kFold(X, y, metric, n_splits=5, random_state=42, real_time_evaluation=False, groups=None, **metric_kwargs):
            Perform k-fold cross-validation prediction using the odor strength module.
        evaluate_kFold(X, y, **kwargs):
            Evaluate the model using k-fold cross-validation.
        scatter_plot(y_true, y_pred, dodge_true=True, dodge_pred=True, categorical=False, dpi=100, figsize=(7, 6), color='#666666', tick_labels={0: "None", 1: "Low", 2: "Medium", 3: "High"}):
            Generate a scatter plot of true vs. predicted values.
        scatter_plotly(y_true, y_pred, X, dodge_true=True, dodge_pred=True, categorical=False, color='#666666', tick_labels={0: "None", 1: "Low", 2: "Medium", 3: "High"}):
            Generate an interactive scatter plot using Plotly.
        plot_confusion_matrix(y_true, y_pred, figsize=(10, 10), dpi=100, tick_labels={0: "None", 1: "Low", 2: "Medium", 3: "High"}):
            Plot the confusion matrix for true vs. predicted values.
        fit(X, y, X_val=None, y_val=None):
            Fit the odor strength module to the training data.
        predict(X):
            Predict odor strength values for the given input data.
    """
    def __init__(self, encoder_name: str, predictor_name: str, hp_molecule_encoder: dict = {}, hp_odor_strength_predictor: dict = {}):
        # self.molecule_encoder = hp_molecule_encoder
        self.odor_strength_predictor = hp_odor_strength_predictor
        encoder = getattr(molecule_encoder, encoder_name)
        predictor = getattr(predictors, predictor_name)
        self.odor_strength_module = OdorStrengthModule(
            encoder(**hp_molecule_encoder),
            predictor(**hp_odor_strength_predictor)
            )
        self.dealing_with_incosistency = hp_odor_strength_predictor.get('dealing_with_incosistency', 'sum')
        self.predictor = predictor
        self.hp_odor_strength_predictor = hp_odor_strength_predictor
    
    def predict_kFold(
        self,
        X: list[str],
        y: np.ndarray,
        metric: Callable | list[Callable],
        n_splits: int = 5,
        random_state: int = 42,
        real_time_evaluation: bool = False,
        groups: np.ndarray | None = None,
        **metric_kwargs
    ) -> tuple[np.ndarray, list, int | None]:
        return self.odor_strength_module.predict_kFold(
            X, y, metric, n_splits=n_splits, random_state=random_state,
            real_time_evaluation=real_time_evaluation, groups=groups, **metric_kwargs
        )

    def evaluate_kFold(
        self,
        X: list[str],
        y: np.ndarray,
        **kwargs
    ) -> tuple[float, float, list, int | None]:
        return self.odor_strength_module.evaluate_kFold(
            X, y, dealing_with_incosistency=self.dealing_with_incosistency, **kwargs
        )

    def scatter_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        dodge_true: bool = True,
        dodge_pred: bool = True,
        categorical: bool = False,
        dpi: int = 100,
        figsize: tuple[int, int] = (7, 6),
        color: str = '#666666',
        tick_labels: dict[int, str] = {0: "None", 1: "Low", 2: "Medium", 3: "High"}
    ) -> None:
        return self.odor_strength_module.scatter_plot(
            y_true, y_pred, dodge_true=dodge_true, dodge_pred=dodge_pred,
            categorical=categorical, dpi=dpi, figsize=figsize, color=color, tick_labels=tick_labels
        )
    
    def scatter_plotly(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        X: list[str],
        dodge_true: bool = True,
        dodge_pred: bool = True,
        categorical: bool = False,
        color: str = '#666666',
        tick_labels: dict[int, str] = {0: "None", 1: "Low", 2: "Medium", 3: "High"}
    ) -> None:
        return self.odor_strength_module.scatter_plotly(
            y_true, y_pred, X, dodge_true=dodge_true, dodge_pred=dodge_pred,
            categorical=categorical, color=color, tick_labels=tick_labels
        )
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        **plot_kwargs
    ) -> None:
        return self.odor_strength_module.plot_confusion_matrix(
            y_true, y_pred, **plot_kwargs
        )

    def fit(
        self,
        X: list[str],
        y: np.ndarray,
        X_val: list[str] | None = None,
        y_val: np.ndarray | None = None
    ) -> int | None:
        return self.odor_strength_module.fit(X, y, X_val=X_val, y_val=y_val)

    def predict(
        self,
        X: list[str]
    ) -> np.ndarray:
        return self.odor_strength_module.predict(X)
    
    def save(self, encoder_path: str, predictor_path: str, predictor_hyperparameter_path: str) -> None:
        """Saves the molecule encoder and odor strength predictor to specified paths."""
        self.odor_strength_module.save(encoder_path, predictor_path)
        with open(predictor_hyperparameter_path, 'w') as f:
            json.dump(self.hp_odor_strength_predictor, f, indent=4)

    
    def load(self, encoder_path: str, predictor_path: str, predictor_hyperparameter_path: str) -> None:
        """Loads the molecule encoder and odor strength predictor from specified paths."""
        with open(predictor_hyperparameter_path, 'r') as f:
            self.hp_odor_strength_predictor = json.load(f)
        self.odor_strength_module.odor_strength_predictor = self.predictor(**self.hp_odor_strength_predictor)
        self.odor_strength_module.load(encoder_path, predictor_path)
    

class OdorStrengthModuleWrapperWithOdorlessFilter(OdorStrengthModule):
    """Wrapper class for OdorStrengthModule that includes an odorless filter (first model predicts if molecule has odor. Second model predicts how strong odor if not odorless.
    Parameters:
        filter_model: An instance of a binary classifier that predicts if a molecule has odor (1) or is odorless (0).
        model: An instance of a predictor that predicts the odor strength of molecules that are not odorless.
    Methods:
        fit(X, y, X_val=None, y_val=None):
            Fits the odorless filter model and the odor strength predictor model to the provided data.
        predict(X):
            Predicts the odor strength for the provided data, filtering out odorless molecules.
    """
    def __init__(self, filter_model: object, model: object):
        self.filter_model = filter_model
        self.model = model

    def fit(
        self,
        X: list,
        y: np.ndarray,
        X_val: list = None,
        y_val: np.ndarray = None
        ) -> None:
        """
        Fits the filter model and the main model using the provided training and optional validation data.

        The method first trains a filter model to perform binary classification on the target labels,
        filtering out samples where the filter predicts a positive class. The filtered data is then used
        to train the main model. If validation data is provided, it is also filtered and used for validation
        during training.

        Parameters:
            X (list): Training feature data.
            y (np.ndarray): Training target labels.
            X_val (list, optional): Validation feature data. Defaults to None.
            y_val (np.ndarray, optional): Validation target labels. Defaults to None.

        Returns:
            None
        """
        y_to_filter = y.copy()
        y_to_filter[y_to_filter > 0] = 1  # Convert to binary classification for filtering
        y_val_to_filter = y_val.copy() if y_val is not None else None
        y_val_to_filter[y_val_to_filter > 0] = 1 if y_val_to_filter is not None else None
        self.filter_model.fit(X, y_to_filter, X_val=X_val, y_val=y_val_to_filter)
        y_filter_train = self.filter_model.predict(X)
        y_filter_val = self.filter_model.predict(X_val) if X_val is not None else None
        y_filter_train = y_filter_train.round().astype(int)  # Ensure binary output
        y_filter_val = y_filter_val.round().astype(int) if y_filter_val is not None else None
        # print(Metrics().calculate_accuracy(y_val_to_filter, y_filter_val), Metrics().calculate_f1_score(y_val_to_filter, y_filter_val))
        X_filtered = [x for x, f in zip(X, y_filter_train) if f == 1]
        y_filtered = y[y_filter_train == 1]
        y_filtered[y_filtered == 0] = 1
        X_val_filtered = [x for x, f in zip(X_val, y_filter_val) if f == 1] if X_val is not None else None
        y_val_filtered = y_val[y_filter_val == 1] if y_val is not None else None
        y_val_filtered[y_val_filtered == 0] = 1 if y_val_filtered is not None else None
        self.model.fit(X_filtered, y_filtered, X_val=X_val_filtered, y_val=y_val_filtered)

    def predict(self, X: list[str]) -> np.ndarray:
        """
        Predicts output values for the given input data X using a two-step process:
        1. Filters the input data using a filter model to select relevant samples.
        2. Applies the main prediction model to the filtered samples.
        3. Returns an array of predictions aligned with the original input, with zeros for filtered-out samples.
        Parameters:
            X (list[str]): Input data samples to predict on.
        Returns:
            np.ndarray: Array of predictions, with zeros for samples filtered out by the filter model.
        """
        y_filter_train = self.filter_model.predict(X)
        y_filter_train = y_filter_train.round().astype(int)  # Ensure binary output
        X_filtered = [x for x, f in zip(X, y_filter_train) if f == 1]
        predictions = self.model.predict(X_filtered)
        predictions_full = np.zeros(len(X))
        predictions_full[y_filter_train == 1] = predictions
        return predictions_full
    
