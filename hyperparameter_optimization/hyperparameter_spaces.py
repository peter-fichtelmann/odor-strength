import os
import pandas as pd

class HyperparameterSpaces:
    def hyperparameter_space_NativeEncoder(self, trial):
        return {}

    def hyperparameter_space_MorganFp(self, trial):
        hps = {
            'radius': trial.suggest_int('radius', 2, 4),
            'count': trial.suggest_categorical('count', [True, False]),
            'countSimulation': trial.suggest_categorical('countSimulation', [True, False]),
            'fpSize': trial.suggest_int('fpSize', 1024, 2048, step=512),
            'includeChirality': trial.suggest_categorical('includeChirality', [True, False]),
            'useBondTypes': trial.suggest_categorical('useBondTypes', [True, False]),
            'atomInvariantsGenerator': trial.suggest_categorical('atomInvariantsGenerator', [True, None]),
        }
        return hps
    
    def hyperparameter_space_RDKitFp(self, trial):
        hps = {
            'count': trial.suggest_categorical('count', [True, False]),
            'countSimulation': trial.suggest_categorical('countSimulation', [True, False]),
            'fpSize': trial.suggest_int('fpSize', 1024, 2048, step=512),
            'atomInvariantsGenerator': trial.suggest_categorical('atomInvariantsGenerator', [True, None]),
            'minPath': trial.suggest_int('minPath', 1, 3),
            'maxPath': trial.suggest_int('maxPath', 5, 9),
            'useHs': trial.suggest_categorical('useHs', [True, False]),
            'branchedPaths': trial.suggest_categorical('branchedPaths', [True, False]),
            'useBondOrder': trial.suggest_categorical('useBondOrder', [True, False]),
            'numBitsPerFeature': trial.suggest_int('numBitsPerFeature', 1, 3),
        }
        return hps
    
    def hyperparameter_space_TopologicalTorsionFp(self, trial):
        hps = {
            'count': trial.suggest_categorical('count', [True, False]),
            'countSimulation': trial.suggest_categorical('countSimulation', [True, False]),
            'fpSize': trial.suggest_int('fpSize', 1024, 2048, step=512),
            'includeChirality': trial.suggest_categorical('includeChirality', [True, False]),
            'torsionAtomCount': trial.suggest_int('torsionAtomCount', 2, 6),
        }
        return hps

    def hyperparameter_space_AtomPairFp(self, trial):
        hps = {
            'count': trial.suggest_categorical('count', [True, False]),
            'countSimulation': trial.suggest_categorical('countSimulation', [True, False]),
            'minDistance': trial.suggest_int('minDistance', 1, 3),
            'maxDistance': trial.suggest_int('maxDistance', 5, 9),
            'includeChirality': trial.suggest_categorical('includeChirality', [True, False]),
            # 'use2D': trial.suggest_categorical('use2D', [True, False]), # only if all molecules 3D compatible
            'fpSize': trial.suggest_int('fpSize', 1024, 2048, step=512),
            'atomInvariantsGenerator': trial.suggest_categorical('atomInvariantsGenerator', [True, None]),
        }
        return hps

    def hyperparameter_space_MACCSKeysFp(self, trial):
        return {}

    def hyperparameter_space_RDKitDescriptors(self, trial):
        return {}
    
    def hyperparameter_space_ChemBerta(self, trial):
        hps = {
            'target_layer': trial.suggest_int('target_layer', 0, 6),  # up to 6 target layers
            'pooling': trial.suggest_categorical('pooling', ['mean', 'cls']),
        }
        if not os.path.exists('chemberta_encodings/'):
            os.makedirs('chemberta_encodings/')
        hps['encoded_database_path'] = f'chemberta_encodings/chemberta_encoding_layer_{hps["target_layer"]}_{hps["pooling"]}.csv'
        return hps
    
    def hyperparameter_space_Average(self, trial):
        return {}
    
    def hyperparameter_space_binary_RuleOfThreePredictor(self, trial):
        return {}

    def hyperparameter_space_binary_LogisticRegressionPredictor(self, trial):
        hps = {
            'standardizer_name': trial.suggest_categorical('standardizer', ['standard', 'robust', 'minmax', 'yeo-johnson', None]),
            'penalty': trial.suggest_categorical('penalty', ['l2', 'l1', 'elasticnet', None]),
            'max_iter': 2000,
            'tol': trial.suggest_float('tol', 1e-5, 1e-3, log=True),
            'n_jobs': -1,
        }
        if hps['penalty']:
            hps['C'] = trial.suggest_float('C', 1e-2, 1e2, log=True)
        if hps['penalty'] == 'elasticnet':
            hps['solver'] = 'saga'
            hps['l1_ratio'] = trial.suggest_float('l1_ratio', 0.1, 0.9, step=0.1)
        elif hps['penalty'] == 'l1':
            hps['solver'] = trial.suggest_categorical('solver_l1', ['liblinear', 'saga'])
        elif hps['penalty'] is None:
            hps['solver'] = trial.suggest_categorical('solver_none', ['saga', 'newton-cg', 'lbfgs'])
        else:
            hps['solver'] = trial.suggest_categorical('solver_l2', ['liblinear', 'saga', 'newton-cg', 'lbfgs'])
        return hps

    def hyperparameter_space_LogisticRegressionPredictor(self, trial):
        hps = self.hyperparameter_space_binary_LogisticRegressionPredictor(trial)
        if hps['penalty'] == 'l1' and hps['solver'] == 'liblinear':
            hps['binarize_labels'] = True
        hps['binarize_labels'] = trial.suggest_categorical('binarize_labels', [True, False])
        if hps['binarize_labels']:
            hps['dealing_with_incosistency'] = trial.suggest_categorical('dealing_with_incosistency', ['sum', 'max'])
        return hps
    
    def hyperparameter_space_binary_GaussianProcessClassifierPredictor(self, trial):
        hps = {
            'standardizer_name': trial.suggest_categorical('standardizer', ['standard', 'robust', 'minmax', 'yeo-johnson', None]),
            'kernel': trial.suggest_categorical('kernel', ['RBF', 'Matern', 'RationalQuadratic', 'DotProduct']),
            'optimizer': trial.suggest_categorical('optimizer', ['fmin_l_bfgs_b', None]),
            'max_iter_predict': 100,
            'n_jobs': -1,
        }
        if hps['optimizer']:
            hps['n_restarts_optimizer'] = 3
        elif hps['kernel'] == 'RBF':
            hps['kernel_hyperparameters'] = {
                'length_scale': trial.suggest_float('length_scale', 1e-2, 1e2, log=True),
            }
        elif hps['kernel'] == 'Matern':
            hps['kernel_hyperparameters'] = {
                'length_scale': trial.suggest_float('length_scale', 1e-2, 1e2, log=True),
                'nu': trial.suggest_categorical('nu', [0.5, 1.5, 2.5]),
            }
        elif hps['kernel'] == 'RationalQuadratic':
            hps['kernel_hyperparameters'] = {
                'length_scale': trial.suggest_float('length_scale', 1e-2, 1e2, log=True),
                'alpha': trial.suggest_float('alpha', 1e-2, 1e2, log=True),
            }
        elif hps['kernel'] == 'DotProduct':
            hps['kernel_hyperparameters'] = {
                'sigma_0': trial.suggest_float('sigma_0', 1e-2, 1e2, log=True),
            }
        return hps
    
    def hyperparameter_space_GaussianProcessClassifierPredictor(self, trial):
        hps = self.hyperparameter_space_binary_GaussianProcessClassifierPredictor(trial)
        hps['binarize_labels'] = trial.suggest_categorical('binarize_labels', [True, False])
        if hps['binarize_labels']:
            hps['dealing_with_incosistency'] = trial.suggest_categorical('dealing_with_incosistency', ['sum', 'max'])
        return hps
    
    def hyperparameter_space_RandomForestRegressorPredictor(self, trial):
        hps = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 5, 20, step=5),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 4, step=1),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5, step=2),
            'min_weight_fraction_leaf': trial.suggest_float('min_weight_fraction_leaf', 0.0, 0.4, step=0.2),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 25, 100, step=25),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.1, step=0.05),
            'n_jobs': -1,
        }
        return hps
    
    def hyperparameter_space_binary_RandomForestClassifierPredictor(self, trial):
        hps = self.hyperparameter_space_RandomForestRegressorPredictor(trial)
        return hps

    def hyperparameter_space_RandomForestClassifierPredictor(self, trial):
        hps = self.hyperparameter_space_RandomForestRegressorPredictor(trial)
        hps['criterion'] = trial.suggest_categorical('criterion', ['gini', 'entropy', 'log_loss'])
        hps['binarize_labels'] = trial.suggest_categorical('binarize_labels', [True, False])
        if hps['binarize_labels']:
            hps['dealing_with_incosistency'] = trial.suggest_categorical('dealing_with_incosistency', ['sum', 'max'])
        return hps
    
    def hyperparameter_space_binary_RandomForestPredictor(self, trial):
        hps = self.hyperparameter_space_binary_RandomForestClassifierPredictor(trial)
        hps['objective'] = 'classification'
        return hps
    
    def hyperparameter_space_RandomForestPredictor(self, trial):
        objective = trial.suggest_categorical('objective', ['classification', 'regression'])
        if objective == 'classification':
            hps = self.hyperparameter_space_RandomForestClassifierPredictor(trial)
        else:
            hps = self.hyperparameter_space_RandomForestRegressorPredictor(trial)
        hps['objective'] = objective
        return hps

    def hyperparameter_space_binary_XGBoostPredictor(self, trial):
        hps = {
            'custom_metric': 'f1_score',
            # 'objective': trial.suggest_categorical('objective', ['weighted_mse', 'weighted_logloss', 'binary:logistic', 'multi:softmax']), # found weight parameter for loss function
            'objective': 'binary:logistic',
            # 'booster': trial.suggest_categorical('booster', ['gbtree', 'gblinear', 'dart']),
            'booster': trial.suggest_categorical('booster', ['gbtree', 'dart']),
            'num_boost_round': 250,
            'early_stopping_rounds': 50,
            'early_stopping_overwrite': 'num_boost_round',
            'eta': trial.suggest_float('eta', 5e-3, 5e-1, log=True),
            'lambda': trial.suggest_float('lambda', 1e-3, 10, log=True),
            'alpha': trial.suggest_float('alpha', 1e-3, 10, log=True),
            'verbose_eval': False,
        }
        # if hps['booster'] == 'gblinear':
            # hps['feature_selector'] = trial.suggest_categorical('feature_selector', ['cyclic', 'shuffle', 'random', 'thrifty'])
            # hps['top_k'] = trial.suggest_int('top_k', 0, 2, step=1)
        # if hps['booster'] in ['gbtree', 'dart']:
        hps['gamma'] = trial.suggest_float('gamma', 1e-3, 10, log=True)
        hps['max_depth'] = trial.suggest_int('max_depth', 4, 10, step=2)
        hps['min_child_weight'] = trial.suggest_float('min_child_weight', 1e-2, 1e2, log=True)
        hps['max_delta_step'] = trial.suggest_float('max_delta_step', 0, 2, step=1)
        hps['subsample'] = trial.suggest_float('subsample', 0.6, 1.0, step=0.2)
        # hps['sampling_method'] = trial.suggest_categorical('sampling_method', ['uniform', 'gradient_based']) # gradient based only GPU Hist
        hps['colsample_bytree'] = trial.suggest_float('colsample_bytree', 0.6, 1.0, step=0.2)
        hps['colsample_bylevel'] = trial.suggest_float('colsample_bylevel', 0.6, 1.0, step=0.2)
        hps['num_parallel_tree'] = trial.suggest_int('num_parallel_tree', 1, 3, step=1)
        if hps['booster'] == 'dart':
            hps['rate_drop'] = trial.suggest_float('rate_drop', 0.0, 0.4, step=0.2)
            hps['skip_drop'] = trial.suggest_float('skip_drop', 0.0, 0.4, step=0.2)
            hps['normalize_type'] = trial.suggest_categorical('normalize_type', ['tree', 'forest'])
            hps['sample_type'] = trial.suggest_categorical('sample_type', ['uniform', 'weighted'])
        return hps

    def hyperparameter_space_XGBoostPredictor(self, trial):
        hps = self.hyperparameter_space_binary_XGBoostPredictor(trial)
        hps['objective'] = trial.suggest_categorical('objective', ['binary:logistic', 'multi:softmax', 'reg:squarederror'])
        hps['custom_metric'] = 'mse_macro'
        if 'multi' in hps['objective']:
            hps['binarize_labels'] = False
            hps['num_class'] = 4
        elif hps['objective'] == 'weighted_logloss' or 'binary' in hps['objective']:
            hps['binarize_labels'] = True
        else:
            hps['binarize_labels'] = trial.suggest_categorical('binarize_labels', [True, False])
        if hps['binarize_labels']:
            hps['dealing_with_incosistency'] = trial.suggest_categorical('dealing_with_incosistency', ['sum', 'max'])
            if hps['booster'] == 'gbtree':
                hps['multi_strategy'] = trial.suggest_categorical('multi_strategy', ['one_output_per_tree', 'multi_output_tree'])
        return hps
    
    def hyperparameter_space_binary_MLP_Predictor(self, trial):
        hps = {
            'standardizer_name': trial.suggest_categorical('standardizer', ['standard', 'robust', 'minmax', 'yeo-johnson', None]),
            'n_layers': trial.suggest_int('n_layers', 2, 12, step=2),
            'dim': trial.suggest_int('mp_dim', 32, 512, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256]),
            'n_epochs': 250,
            'objective_name': 'binary_crossentropy',
            'learning_rate': trial.suggest_float('learning_rate', 5e-6, 5e-4, log=True),
            'optimizer_name': trial.suggest_categorical('optimizer_name', ['adam', 'adamw', 'sgd', 'rmsprop', 'adagrad']),
            'training_scheduler_name': trial.suggest_categorical('training_scheduler_name', [None, 'ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']),
            'warmup_epochs': trial.suggest_int('warmup_epochs', 0, 30, step=10),
            'activation': trial.suggest_categorical('activation', ['relu', 'leaky-relu', 'tanh', 'sigmoid', 'elu']),
            'dropout': trial.suggest_float('dropout', 0.0, 0.5, step=0.1),
            'early_stopping_params': {
                'early_stopping_rounds': 50,
                'delta': 0,
            },
            'early_stopping_overwrite': 'n_epochs',
            'weight_average': trial.suggest_categorical('weight_average', [False, 'swa', 'ema']),
            'metric_name': 'f1_score',
            'verbose_eval': False,
        }
        optimizer_params = {
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)
        }
        if hps['optimizer_name'] in ['adam', 'adamw']:
            optimizer_params['betas'] = (trial.suggest_float('beta1', 0.85, 0.95, step=0.05), trial.suggest_float('beta2', 0.99, 0.999, step=0.003))
        elif hps['optimizer_name'] in ['sgd', 'rmsprop']:
            optimizer_params['momentum'] = trial.suggest_float('momentum', 0, 0.5, step=0.1)
            if hps['optimizer_name'] == 'sgd' and optimizer_params['momentum'] > 0:
                optimizer_params['nesterov'] = trial.suggest_categorical('nesterov', [True, False])
                if not optimizer_params['nesterov']:
                    optimizer_params['dampening'] = trial.suggest_float('dampening', 0.0, 0.1, step=0.05)
            elif hps['optimizer_name'] == 'rmsprop':
                optimizer_params['alpha'] = trial.suggest_float('alpha', 0.9, 0.99, step=0.03)
        elif hps['optimizer_name'] == 'adagrad':
            optimizer_params['lr_decay'] = trial.suggest_float('lr_decay', 0.0, 0.1, step=0.05)
        hps['optimizer_params'] = optimizer_params
        training_scheduler_params = {}
        if hps['training_scheduler_name'] == 'ReduceLROnPlateau':
            training_scheduler_params['factor'] = trial.suggest_float('factor', 0.1, 0.5, step=0.1)
        elif hps['training_scheduler_name'] == 'CosineAnnealingWarmRestarts':
            training_scheduler_params['T_0'] = trial.suggest_int('T_0', 10, 40, step=10)
            training_scheduler_params['T_mult'] = trial.suggest_int('T_mult', 2, 4, step=1)
        return hps
    
    def hyperparameter_space_MLP_Predictor(self, trial):
        hps = self.hyperparameter_space_binary_MLP_Predictor(trial)
        hps['objective_name'] = trial.suggest_categorical('objective_name', ['binary_crossentropy', 'mse'])
        hps['binarize_labels'] = trial.suggest_categorical('binarize_labels', [True, False])
        if hps['binarize_labels']:
            hps['dealing_with_incosistency'] = trial.suggest_categorical('dealing_with_incosistency', ['sum', 'max'])
        hps['metric_name'] = 'mse_macro'
        return hps

    def hyperparameter_space_CoralPredictor(self, trial):
        hps = self.hyperparameter_space_MLP_Predictor(trial)
        hps['objective_name'] = 'coral'
        hps['binarize_labels'] = True
        return hps
    
    def hyperparameter_space_CornPredictor(self, trial):
        hps = self.hyperparameter_space_MLP_Predictor(trial)
        hps['objective_name'] = 'corn'
        hps['binarize_labels'] = True
        return hps

    def hyperparameters_without_mp_ChemPropMPNNPredictor(self, trial):
        hps = {
            'epochs': 250,
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128, 256]),
            'aggregation': trial.suggest_categorical('aggregation', ['norm', 'sum', 'mean', 'attentive']),
            'n_layers': trial.suggest_int('n_layers', 1, 12, step=1),
            'objective': 'binary_crossentropy',
            'predictor_dim': trial.suggest_int('prediction_dim', 32, 512, log=True),
            'predictor_dropout': trial.suggest_float('predictor_dropout', 0.0, 0.5, step=0.1),
            'predictor_activation': trial.suggest_categorical('predictor_activation', ['relu', 'leakyrelu', 'prelu', 'tanh', 'elu']),
            'warmup_epochs': trial.suggest_int('warmup_epochs', 0, 30, step=10),
            'initial_learning_rate': trial.suggest_float('initial_learning_rate', 5e-6, 1e-3, log=True),
            'max_learning_rate_factor': trial.suggest_float('max_learning_rate_factor', 1, 100, log=True),
            'final_learning_rate_factor': trial.suggest_float('final_learning_rate_factor', 1e-2, 10, log=True),
            'early_stopping_params': {
                'mode': 'min',
                'early_stopping_rounds': 50,
                'delta': 0,
            },
            'early_stopping_overwrite': 'epochs',
            'verbose_eval': False,
        }
        hps['max_learning_rate'] = hps['initial_learning_rate'] * hps.pop('max_learning_rate_factor')
        hps['final_learning_rate'] = hps['initial_learning_rate'] * hps.pop('final_learning_rate_factor')
        return hps


    def hyperparameter_space_binary_ChemPropMPNNPredictor(self, trial):
        hps = self.hyperparameters_without_mp_ChemPropMPNNPredictor(trial)
        hps_mp = {
            'message_passing': trial.suggest_categorical('message_passing', ['bond', 'atom']),
            'mp_dim': trial.suggest_int('mp_dim', 32, 512, log=True),
            'bias': trial.suggest_categorical('bias', [True, False]),
            'n_message_passing_iterations': trial.suggest_int('n_message_passing_iterations', 1, 8),
            'messages_undirected_edges': trial.suggest_categorical('messages_undirected_edges', [True, False]),
            'mp_dropout': trial.suggest_float('mp_dropout', 0.0, 0.5, step=0.1),
            'mp_activation': trial.suggest_categorical('mp_activation', ['relu', 'leakyrelu', 'prelu', 'tanh', 'elu']),
        }
        hps.update(hps_mp)
        return hps
    
    def hyperparameter_space_ChemPropMPNNPredictor(self, trial):
        hps = self.hyperparameter_space_binary_ChemPropMPNNPredictor(trial)
        hps['binarize_labels'] = trial.suggest_categorical('binarize_labels', [True, False])
        if hps['binarize_labels']:
            hps['dealing_with_incosistency'] = trial.suggest_categorical('dealing_with_incosistency', ['sum', 'max'])
        hps['objective'] = trial.suggest_categorical('objective', ['binary_crossentropy', 'mse'])
        return hps
    
    def hyperparameter_space_binary_ChemeleonPredictor(self, trial):
        hps = self.hyperparameters_without_mp_ChemPropMPNNPredictor(trial)
        hps['message_passing'] = 'chemeleon'
        return hps

    def hyperparameter_space_ChemeleonPredictor(self, trial):
        hps = self.hyperparameter_space_ChemPropMPNNPredictor(trial)
        hps['message_passing'] = 'chemeleon'
        hps['epochs'] = 100
        hps['early_stopping_params'] = {
                'mode': 'min',
                'early_stopping_rounds': 25,
                'delta': 0,
            }
        return hps