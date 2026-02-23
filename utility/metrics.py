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

    def calculate_f1_micro(self, y_true: np.ndarray, y_pred: np.ndarray, beta: float = 1.0, **kwargs) -> float:
        """identical to accuracy in multiclass"""
        if len(y_pred) == 0:
            raise ValueError("y_pred is empty. Cannot calculate F1 micro.")
        y_pred_rounded = np.round(y_pred.copy()).astype(int)
        labels = np.unique(np.concatenate([y_true, y_pred_rounded]))
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        true_negatives = 0
        for label in labels:
            true_positives += np.sum((y_true == label) & (y_pred_rounded == label))
            false_positives += np.sum((y_true != label) & (y_pred_rounded == label))
            false_negatives += np.sum((y_true == label) & (y_pred_rounded != label))
            true_negatives += np.sum((y_true != label) & (y_pred_rounded != label))
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (beta**2 * precision + recall) > 0 else 0
        return f_beta

    def calculate_f1_macro(self, y_true: np.ndarray, y_pred: np.ndarray, beta: float = 1.0, **kwargs) -> float:
        if len(y_pred) == 0:
            raise ValueError("y_pred is empty. Cannot calculate F1 macro.")
        y_pred_rounded = np.round(y_pred.copy()).astype(int)
        y_pred_rounded = np.round(y_pred.copy()).astype(int)
        labels = np.unique(np.concatenate([y_true, y_pred_rounded]))
        f_scores = []
        for label in labels:
            true_positives = np.sum((y_true == label) & (y_pred_rounded == label))
            false_positives = np.sum((y_true != label) & (y_pred_rounded == label))
            false_negatives = np.sum((y_true == label) & (y_pred_rounded != label))
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
            f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (beta**2 * precision + recall) > 0 else 0
            f_scores.append(f_beta)
        return float(np.mean(f_scores)) if len(f_scores) > 0 else 0.0, f_scores

    # def calculate_roc_curve(self, y_true: np.ndarray, y_score: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    #     if len(y_score) == 0:
    #         raise ValueError("y_score is empty. Cannot calculate ROC.")
    #     y_true = y_true.astype(int)
    #     thresholds = np.unique(y_score)
    #     thresholds = np.sort(thresholds)[::-1]
    #     tpr_list = []
    #     fpr_list = []
    #     for threshold in thresholds:
    #         y_pred = (y_score >= threshold).astype(int)
    #         true_positives = np.sum((y_true == 1) & (y_pred == 1))
    #         false_positives = np.sum((y_true == 0) & (y_pred == 1))
    #         false_negatives = np.sum((y_true == 1) & (y_pred == 0))
    #         true_negatives = np.sum((y_true == 0) & (y_pred == 0))
    #         tpr = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0.0
    #         fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else 0.0
    #         tpr_list.append(tpr)
    #         fpr_list.append(fpr)
    #     return np.array(fpr_list), np.array(tpr_list), thresholds

    # def calculate_auc(self, fpr: np.ndarray, tpr: np.ndarray) -> float:
    #     if len(fpr) == 0 or len(tpr) == 0:
    #         raise ValueError("fpr/tpr is empty. Cannot calculate AUC.")
    #     order = np.argsort(fpr)
    #     return float(np.trapz(tpr[order], fpr[order]))

    def calculate_roc_auc(self, y_true: np.ndarray, y_score: np.ndarray) -> float:
        from sklearn.metrics import roc_auc_score
        if len(y_score) == 0:
            raise ValueError("y_score is empty. Cannot calculate AUC.")
        
        aucs = []
        for k in range(1, int(y_true.max()) + 1):
            y_bin = (y_true >= k).astype(int)
            aucs.append(roc_auc_score(y_bin, y_score))
        return np.mean(aucs), aucs

    # def calculate_ordinal_roc_auc(self, y_true: np.ndarray, y_score: np.ndarray, average: str = 'macro') -> tuple[dict, float]:
    #     if len(y_score) == 0:
    #         raise ValueError("y_score is empty. Cannot calculate ordinal AUC.")
    #     y_true = y_true.astype(int)
    #     y_score = y_score.astype(float)
    #     classes = np.unique(y_true)
    #     if len(classes) < 2:
    #         raise ValueError("Need at least two ordinal classes to compute ordinal AUC.")
    #     thresholds = classes[:-1]
    #     aucs = {}
    #     auc_values = []
    #     for threshold in thresholds:
    #         y_true_bin = (y_true > threshold).astype(int)
    #         fpr, tpr, _ = self.calculate_roc_curve(y_true_bin, y_score)
    #         auc_value = self.calculate_auc(fpr, tpr)
    #         aucs[int(threshold)] = auc_value
    #         auc_values.append(auc_value)
    #     if average == 'macro':
    #         return aucs, float(np.mean(auc_values)) if len(auc_values) > 0 else 0.0
    #     if average == 'micro':
    #         y_true_bins = []
    #         y_score_bins = []
    #         for threshold in thresholds:
    #             y_true_bins.append((y_true > threshold).astype(int))
    #             y_score_bins.append(y_score.copy())
    #         y_true_all = np.concatenate(y_true_bins) if len(y_true_bins) > 0 else np.array([])
    #         y_score_all = np.concatenate(y_score_bins) if len(y_score_bins) > 0 else np.array([])
    #         if y_true_all.size == 0 or np.unique(y_true_all).size < 2:
    #             return aucs, 0.0
    #         fpr, tpr, _ = self.calculate_roc_curve(y_true_all, y_score_all)
    #         return aucs, self.calculate_auc(fpr, tpr)
    #     if average == 'weighted':
    #         weights = []
    #         for threshold in thresholds:
    #             weights.append(np.sum(y_true > threshold))
    #         total_weight = np.sum(weights)
    #         weighted_auc = np.sum(np.array(auc_values) * np.array(weights)) / total_weight if total_weight > 0 else 0.0
    #         return aucs, float(weighted_auc)
    #     raise ValueError("Invalid average. Use 'macro', 'micro', or 'weighted'.")
    
    def calculate_ordinal_roc_auc(self, y_true: np.ndarray, y_score: np.ndarray, average: str = 'macro') -> tuple[dict, float]:
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(
                y_true,
                y_score,
                multi_class="ovr",
                average=average,
        )
        return auc
    
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

    def calculate_f1_micro(self, y_true: torch.Tensor, y_pred: torch.Tensor, beta: float = 1.0, **kwargs) -> torch.Tensor:
        if y_pred.numel() == 0:
            raise ValueError("y_pred is empty. Cannot calculate F1 micro.")
        y_pred_rounded = torch.round(y_pred).to(torch.int)
        labels = torch.unique(torch.cat([y_true, y_pred_rounded]))
        true_positives = torch.tensor(0.0)
        false_positives = torch.tensor(0.0)
        false_negatives = torch.tensor(0.0)
        for label in labels:
            true_positives += torch.sum((y_true == label) & (y_pred_rounded == label)).float()
            false_positives += torch.sum((y_true != label) & (y_pred_rounded == label)).float()
            false_negatives += torch.sum((y_true == label) & (y_pred_rounded != label)).float()
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else torch.tensor(0.0)
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else torch.tensor(0.0)
        f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (beta**2 * precision + recall) > 0 else torch.tensor(0.0)
        return f_beta

    def calculate_f1_macro(self, y_true: torch.Tensor, y_pred: torch.Tensor, beta: float = 1.0, **kwargs) -> torch.Tensor:
        if y_pred.numel() == 0:
            raise ValueError("y_pred is empty. Cannot calculate F1 macro.")
        y_pred_rounded = torch.round(y_pred).to(torch.int)
        labels = torch.unique(torch.cat([y_true, y_pred_rounded]))
        f_scores = []
        for label in labels:
            true_positives = torch.sum((y_true == label) & (y_pred_rounded == label)).float()
            false_positives = torch.sum((y_true != label) & (y_pred_rounded == label)).float()
            false_negatives = torch.sum((y_true == label) & (y_pred_rounded != label)).float()
            precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else torch.tensor(0.0)
            recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else torch.tensor(0.0)
            f_beta = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall) if (beta**2 * precision + recall) > 0 else torch.tensor(0.0)
            f_scores.append(f_beta)
        if len(f_scores) == 0:
            return torch.tensor(0.0)
        return torch.mean(torch.stack(f_scores))

    def calculate_roc_curve(self, y_true: torch.Tensor, y_score: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if y_score.numel() == 0:
            raise ValueError("y_score is empty. Cannot calculate ROC.")
        y_true = y_true.to(torch.int)
        thresholds = torch.unique(y_score)
        thresholds = torch.sort(thresholds, descending=True).values
        tpr_list = []
        fpr_list = []
        for threshold in thresholds:
            y_pred = (y_score >= threshold).to(torch.int)
            true_positives = torch.sum((y_true == 1) & (y_pred == 1)).float()
            false_positives = torch.sum((y_true == 0) & (y_pred == 1)).float()
            false_negatives = torch.sum((y_true == 1) & (y_pred == 0)).float()
            true_negatives = torch.sum((y_true == 0) & (y_pred == 0)).float()
            tpr = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else torch.tensor(0.0)
            fpr = false_positives / (false_positives + true_negatives) if (false_positives + true_negatives) > 0 else torch.tensor(0.0)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        return torch.stack(fpr_list), torch.stack(tpr_list), thresholds

    def calculate_auc(self, fpr: torch.Tensor, tpr: torch.Tensor) -> torch.Tensor:
        if fpr.numel() == 0 or tpr.numel() == 0:
            raise ValueError("fpr/tpr is empty. Cannot calculate AUC.")
        order = torch.argsort(fpr)
        fpr_sorted = fpr[order]
        tpr_sorted = tpr[order]
        return torch.trapz(tpr_sorted, fpr_sorted)

    def calculate_ordinal_roc_auc(self, y_true: torch.Tensor, y_score: torch.Tensor, average: str = 'macro') -> tuple[dict, torch.Tensor]:
        if y_score.numel() == 0:
            raise ValueError("y_score is empty. Cannot calculate ordinal AUC.")
        y_true = y_true.to(torch.int)
        y_score = y_score.to(torch.float)
        classes = torch.unique(y_true)
        if classes.numel() < 2:
            raise ValueError("Need at least two ordinal classes to compute ordinal AUC.")
        thresholds = classes[:-1]
        aucs = {}
        auc_values = []
        weights = []
        for threshold in thresholds:
            y_true_bin = (y_true > threshold).to(torch.int)
            fpr, tpr, _ = self.calculate_roc_curve(y_true_bin, y_score)
            auc_value = self.calculate_auc(fpr, tpr)
            aucs[int(threshold.item())] = float(auc_value.item())
            auc_values.append(auc_value)
            weights.append(torch.sum(y_true > threshold).float())
        if len(auc_values) == 0:
            return aucs, torch.tensor(0.0)
        if average == 'macro':
            return aucs, torch.mean(torch.stack(auc_values))
        if average == 'micro':
            y_true_bins = []
            y_score_bins = []
            for threshold in thresholds:
                y_true_bins.append((y_true > threshold).to(torch.int))
                y_score_bins.append(y_score.clone())
            y_true_all = torch.cat(y_true_bins) if len(y_true_bins) > 0 else torch.tensor([])
            y_score_all = torch.cat(y_score_bins) if len(y_score_bins) > 0 else torch.tensor([])
            if y_true_all.numel() == 0 or torch.unique(y_true_all).numel() < 2:
                return aucs, torch.tensor(0.0)
            fpr, tpr, _ = self.calculate_roc_curve(y_true_all, y_score_all)
            return aucs, self.calculate_auc(fpr, tpr)
        if average == 'weighted':
            weight_tensor = torch.stack(weights)
            total_weight = torch.sum(weight_tensor)
            weighted_auc = torch.sum(torch.stack(auc_values) * weight_tensor) / total_weight if total_weight > 0 else torch.tensor(0.0)
            return aucs, weighted_auc
        raise ValueError("Invalid average. Use 'macro', 'micro', or 'weighted'.")


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