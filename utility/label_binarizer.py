import numpy as np
import torch

class LabelBinarizer:
    def binarize_labels(self, y):
        if len(y) == 0:
            return y
        unique_labels = np.unique(y)
        if len(unique_labels) <= 2:
            return y
        binarized_labels = np.zeros((len(y), len(unique_labels) - 1), dtype=int)
        update_vector = np.zeros(len(unique_labels) - 1)
        for i, unique_label in enumerate(unique_labels):
            # print(update_vector)
            binarized_labels[y == unique_label] = update_vector
            if i < len(unique_labels) - 1:
                update_vector[i] = 1
        return binarized_labels
    
    def inverse_binarize_labels(self, binarized_labels, dealing_with_incosistency='sum', one_sample=False):
        if len(np.squeeze(binarized_labels).shape) <= 1:
            if not one_sample:
                return binarized_labels
        binarized_labels = np.round(binarized_labels).astype(int)
        if dealing_with_incosistency == 'sum':
            if one_sample:
                inverse_labels = np.sum(binarized_labels)
            else:
                inverse_labels = np.sum(binarized_labels, axis=1)
        elif dealing_with_incosistency == 'max':
            if one_sample:
                indices = np.where(binarized_labels == 1)[0]
                if indices.size > 0:
                    inverse_labels = indices.max() + 1
                else:
                    inverse_labels = 0
            else:
                result = np.where((binarized_labels == 1), np.arange(binarized_labels.shape[1]), -1) + 1
                inverse_labels = np.max(result, axis=1)
        else:
            raise ValueError("Invalid option for dealing_with_incosistency. Use 'sum' or 'max'.")
        return inverse_labels
    
    def torch_inverse_binarize_labels(self, binarized_labels, dealing_with_incosistency='sum', one_sample=False):
        if len(torch.squeeze(binarized_labels).shape) <= 1:
            if not one_sample:
                return binarized_labels
        binarized_labels = torch.round(binarized_labels)
        if dealing_with_incosistency == 'sum':
            if one_sample:
                inverse_labels = torch.sum(binarized_labels)
            else:
                inverse_labels = torch.sum(binarized_labels, dim=1)
        elif dealing_with_incosistency == 'max':
            # Add 1 to class indices, and mask zeros with -1
            if one_sample:
                indices = (binarized_labels == 1).nonzero(as_tuple=True)[0]
                if indices.numel() > 0:
                    inverse_labels = torch.tensor(indices.max().item() + 1)
                else:
                    inverse_labels = torch.tensor(0.0)
            else:
                idx = torch.arange(binarized_labels.shape[1], device=binarized_labels.device).unsqueeze(0) + 1
                masked = torch.where(binarized_labels == 1, idx, torch.full_like(idx, 0))
                inverse_labels = torch.max(masked, dim=1).values
                inverse_labels = inverse_labels.to(torch.float32)
        else:
            raise ValueError("Invalid option for dealing_with_incosistency. Use 'sum' or 'max'.")
        return inverse_labels