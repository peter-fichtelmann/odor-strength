import numpy as np
import torch
from .label_binarizer import LabelBinarizer

class Metrics:
    def calculate_mse(self, y_true, y_pred):
        if len(y_pred) == 0:
            raise ValueError("y_pred is empty. Cannot calculate RMSE.")
        return np.mean((y_true - y_pred) ** 2)

    def calculate_mse_macro(self, y_true, y_pred, dealing_with_incosistency='sum'):
        mses = {}
        y_pred_rounded = np.round(y_pred.copy()).astype(int)
        if len(np.squeeze(y_pred_rounded).shape) > 1:
            y_pred_rounded = LabelBinarizer().inverse_binarize_labels(y_pred_rounded, dealing_with_incosistency=dealing_with_incosistency)
        if len(np.squeeze(y_true).shape) > 1:
            y_true = LabelBinarizer().inverse_binarize_labels(y_true, dealing_with_incosistency=dealing_with_incosistency)
        for category in np.unique(y_true):
            mses[category] = self.calculate_mse(y_true[y_true == category], y_pred_rounded[y_true == category])
        return np.mean(list(mses.values())), mses
    
    def calculate_f1_score(self, y_true:np.ndarray, y_pred:np.ndarray, beta: float=1, minority_class:str|int='auto', **kwargs):
        if len(y_pred) == 0:
            raise ValueError("y_pred is empty. Cannot calculate F1 Score.")
        if minority_class == 'auto':
            count = np.unique(y_true, return_counts=True)
            majority_class = count[0][np.argmax(count[1])]
            minority_class = count[0][np.argmin(count[1])]
        elif minority_class == 0:
            majority_class = 1
        elif minority_class == 1:
            majority_class = 0
        else:
            raise ValueError("Invalid minority_class. Use 'auto', 0, or 1.")
        y_pred_rounded = np.round(y_pred.copy()).astype(int)
        true_positives = np.sum((y_true == minority_class) & (y_pred_rounded == minority_class))
        false_positives = np.sum((y_true == majority_class) & (y_pred_rounded == minority_class))
        false_negatives = np.sum((y_true == minority_class) & (y_pred_rounded == majority_class))
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (beta**2 * precision + recall) > 0 else 0
        return f_beta, precision, recall
    
    def calculate_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        if len(y_pred) == 0:
            raise ValueError("y_pred is empty. Cannot calculate accuracy.")
        y_pred_rounded = np.round(y_pred.copy()).astype(int)
        return np.mean(y_true == y_pred_rounded)
    
    def calculate_r2(self, y_true: np.ndarray, y_pred: np.ndarray, **kwargs) -> float:
        if len(y_pred) == 0:
            raise ValueError("y_pred is empty. Cannot calculate R2.")
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0


class TorchMetrics:
    def calculate_mse(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        if y_pred.numel() == 0:
            raise ValueError("y_pred is empty. Cannot calculate MSE.")
        return torch.mean((y_true - y_pred) ** 2)

    def calculate_mse_macro(self, y_true: torch.Tensor, y_pred: torch.Tensor, dealing_with_incosistency: str='sum') -> tuple[torch.Tensor, dict]:
        if y_pred.numel() == 0:
            raise ValueError("y_pred is empty. Cannot calculate MSE Macro.")
        y_pred_rounded = torch.round(y_pred).to(torch.int)
        if len(torch.squeeze(y_pred_rounded).shape) > 1:
            y_pred_rounded = LabelBinarizer().torch_inverse_binarize_labels(y_pred_rounded, dealing_with_incosistency=dealing_with_incosistency)
        if len(torch.squeeze(y_true).shape) > 1:
            y_true = LabelBinarizer().torch_inverse_binarize_labels(y_true, dealing_with_incosistency=dealing_with_incosistency)
        categories = torch.unique(y_true)
        
        mses = {}
        mse_values = []

        for category in categories:
            mask = y_true == category
            mse_cat = torch.mean((y_true[mask] - y_pred_rounded[mask]) ** 2)
            mses[int(category.item())] = mse_cat.item()
            mse_values.append(mse_cat)
        
        macro_mse = torch.mean(torch.stack(mse_values))
        return macro_mse, mses
    
    def calculate_f1_score(self, y_true: torch.Tensor, y_pred: torch.Tensor, beta: float = 1.0, minority_class: str|int='auto', **kwargs) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if y_pred.numel() == 0:
            raise ValueError("y_pred is empty. Cannot calculate F1 Score.")
        if minority_class == 'auto':
            count = torch.unique(y_true, return_counts=True)
            majority_class = count[0][torch.argmax(count[1])]
            minority_class = count[0][torch.argmin(count[1])]
        elif minority_class == 0:
            majority_class = 1
        elif minority_class == 1:
            majority_class = 0
        else:
            raise ValueError("Invalid minority_class. Use 'auto', 0, or 1.")
        y_pred_rounded = torch.round(y_pred).to(torch.int)
        true_positives = torch.sum((y_true == minority_class) & (y_pred_rounded == minority_class)).float()
        false_positives = torch.sum((y_true == majority_class) & (y_pred_rounded == minority_class)).float()
        false_negatives = torch.sum((y_true == minority_class) & (y_pred_rounded == majority_class)).float()
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else torch.tensor(0.0)
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else torch.tensor(0.0)
        f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (beta**2 * precision + recall) > 0 else torch.tensor(0.0)
        
        return f_beta, precision, recall


# # test for same results:
# from FragranceFinder.general.metrics import Metrics, TorchMetrics
# import torch

# metrics = Metrics()
# metrics_torch = TorchMetrics()

# y_true_array = np.array([0, 1, 2, 0, 2, 3, 1, 2])  # Example true labels
# y_pred_array = np.array([0, 2, 1, 3, 2, 2, 1, 2])  # Example predicted values
# y_true = torch.tensor(y_true_array, dtype=torch.float64)
# y_pred = torch.tensor(y_pred_array, dtype=torch.float64)

# mse = metrics.calculate_mse(y_true_array, y_pred_array)
# macro_mse, per_class = metrics.calculate_mse_macro(y_true_array, y_pred_array)

# print("MSE:", mse.item())
# print("Macro MSE:", macro_mse.item())
# print("Per class MSE:", per_class)


# mse = metrics_torch.calculate_mse(y_true, y_pred)
# macro_mse, per_class = metrics_torch.calculate_mse_macro(y_true, y_pred)

# print("MSE:", mse.item())
# print("Macro MSE:", macro_mse.item())
# print("Per class MSE:", per_class)