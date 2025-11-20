from sklearn.model_selection import StratifiedGroupKFold

def stratified_group_train_test_split(X, y, groups, test_size=0.2, random_state=None, stratify_data=None):
    n_splits = int(1 / test_size)
    kf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    if stratify_data is None:
        stratify_data = y
    for i, (train_index, test_index) in enumerate(kf.split(X, stratify_data, groups=groups)):
        break
    return X[train_index], X[test_index], y[train_index], y[test_index], groups[train_index], groups[test_index]