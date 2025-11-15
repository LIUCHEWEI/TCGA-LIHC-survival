# Required imports
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from lifelines.utils import concordance_index
from lifelines import KaplanMeierFitter
from scipy.stats import pearsonr
from scipy import stats
import os
import shap
import seaborn as sns
import warnings

# Configuration
DATA_FILE_PATH = '/content/drive/MyDrive/merge_data_set_with_TMB.csv'

# Set random seed for reproducibility
def set_seed(seed=42):
    """Set random seed for reproducibility"""
    np.random.seed(seed)

set_seed(42)

class CIndexCallback(xgb.callback.TrainingCallback):

    def __init__(self, dtrain, dval, y_time_train, y_event_train,
                 y_time_val=None, y_event_val=None, cindex_frequency=10, verbose=True):
        super().__init__()
        self.dtrain = dtrain
        self.dval = dval
        self.y_time_train = y_time_train
        self.y_event_train = y_event_train
        self.y_time_val = y_time_val
        self.y_event_val = y_event_val
        self.cindex_frequency = cindex_frequency
        self.verbose = verbose
        self.training_c_indices = []
        self.validation_c_indices = []
        self.training_losses = []
        self.validation_losses = []
        self.iteration_count = 0

    def after_iteration(self, model, epoch, evals_log):
        """Called after each iteration"""
        self.iteration_count = epoch

        # Calculate C-index at specified frequency
        if epoch % self.cindex_frequency == 0 or epoch == 0:
            # Training C-index
            train_pred = model.predict(self.dtrain)
            train_c_idx = concordance_index(self.y_time_train, -train_pred, self.y_event_train)
            self.training_c_indices.append(train_c_idx)

            # Validation C-index if available
            if self.dval is not None and self.y_time_val is not None:
                val_pred = model.predict(self.dval)
                val_c_idx = concordance_index(self.y_time_val, -val_pred, self.y_event_val)
                self.validation_c_indices.append(val_c_idx)

                if self.verbose and epoch % 20 == 0:
                    print(f"[{epoch}] train-c-index:{train_c_idx:.4f} | valid-c-index:{val_c_idx:.4f}")

        return False  # Continue training

    def after_training(self, model):
        """Called after training completes"""
        if self.verbose:
            print(f"Training completed after {self.iteration_count + 1} iterations")
            if self.training_c_indices:
                print(f"   Final train C-index: {self.training_c_indices[-1]:.4f}")
            if self.validation_c_indices:
                print(f"   Final valid C-index: {self.validation_c_indices[-1]:.4f}")
        return model

class CVFoldCallback(xgb.callback.TrainingCallback):
    """Enhanced callback for cross-validation fold tracking with validation curves"""

    def __init__(self, dtrain, dval, y_time_train, y_event_train,
                 y_time_val, y_event_val, cindex_frequency=10, verbose=False):
        super().__init__()
        self.dtrain = dtrain
        self.dval = dval
        self.y_time_train = y_time_train
        self.y_event_train = y_event_train
        self.y_time_val = y_time_val
        self.y_event_val = y_event_val
        self.cindex_frequency = cindex_frequency
        self.verbose = verbose
        self.training_c_indices = []
        self.validation_c_indices = []
        self.training_losses = []
        self.validation_losses = []
        self.iteration_count = 0

    def after_iteration(self, model, epoch, evals_log):
        """Called after each iteration"""
        self.iteration_count = epoch

        # Record training and validation loss if available in evals_log
        if 'train' in evals_log and 'cox-nloglik' in evals_log['train']:
            train_loss = evals_log['train']['cox-nloglik'][-1]
            self.training_losses.append(train_loss)

        if 'eval' in evals_log and 'cox-nloglik' in evals_log['eval']:
            val_loss = evals_log['eval']['cox-nloglik'][-1]
            self.validation_losses.append(val_loss)

        # Calculate C-index at specified frequency
        if epoch % self.cindex_frequency == 0 or epoch == 0:
            # Training C-index
            train_pred = model.predict(self.dtrain)
            train_c_idx = concordance_index(self.y_time_train, -train_pred, self.y_event_train)
            self.training_c_indices.append(train_c_idx)

            # Validation C-index
            val_pred = model.predict(self.dval)
            val_c_idx = concordance_index(self.y_time_val, -val_pred, self.y_event_val)
            self.validation_c_indices.append(val_c_idx)

            if self.verbose:
                print(f"[{epoch}] train-c-index:{train_c_idx:.4f} | val-c-index:{val_c_idx:.4f}")

        return False  # Continue training

    def after_training(self, model):
        """Called after training completes"""
        if self.verbose:
            print(f"Fold training completed after {self.iteration_count + 1} iterations")
            if self.training_c_indices:
                print(f"   Final train C-index: {self.training_c_indices[-1]:.4f}")
            if self.validation_c_indices:
                print(f"   Final validation C-index: {self.validation_c_indices[-1]:.4f}")
        return model

class SurvivalXGBoost:
    """XGBoost model for survival analysis using Cox objective"""

    def __init__(self, params=None, output_dir='outputs'):
        """
        Initialize XGBoost survival model

        Args:
            params: Dictionary of XGBoost parameters
            output_dir: Directory to save outputs
        """
        # Best Optimized parameters for LIHC dataset (Latest Optuna Results)
        self.default_params = {
          'objective': 'survival:cox',
          'eval_metric': 'cox-nloglik',
          'max_depth': 5,
          'min_child_weight': 10,
          'gamma': 0.9136715001046197,
          'learning_rate': 0.07189583747010814,
          'n_estimators': 336,
          'reg_alpha': 4.744619224985263,
          'reg_lambda': 13.803692806989263,
          'subsample': 0.9928682455283004,
          'colsample_bytree': 0.9723965058981796,
          'scale_pos_weight': 1.6999664722190158,
          'random_state': 42,
          'n_jobs': -1,
          'tree_method': 'hist',
          'predictor': 'cpu_predictor',
          'verbosity': 1
        }

        if params:
            self.default_params.update(params)

        # Store params for easy access (alias for default_params)
        self.params = self.default_params

        self.model = None
        self.scaler = MinMaxScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.feature_importance = None
        self.evals_result = {}
        self.training_c_indices = []
        self.validation_c_indices = []

        # Create output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def prepare_survival_data(self, y_time, y_event):
        """
        Prepare survival data for XGBoost Cox model

        Args:
            y_time: Survival time array
            y_event: Event indicator array (1=event, 0=censored)

        Returns:
            Formatted labels for XGBoost survival analysis
        """
        # XGBoost Cox expects negative values for censored observations
        # and positive values for events
        survival_labels = y_time.copy()
        survival_labels[y_event == 0] = -survival_labels[y_event == 0]

        return survival_labels

    def _train_xgb_model(self, X_train, y_train_survival, X_val=None, y_val_survival=None,
                        evals_result=None, callbacks=None, early_stopping_rounds=None, verbose_eval=None):
        """
        Core XGBoost training method - SHARED by both modes for consistency

        Args:
            X_train: Training features
            y_train_survival: Training survival labels
            X_val: Validation features (optional)
            y_val_survival: Validation survival labels (optional)
            evals_result: Dictionary to store evaluation results
            callbacks: List of callbacks
            early_stopping_rounds: Early stopping rounds
            verbose_eval: Verbose evaluation frequency

        Returns:
            Trained XGBoost model
        """
        # Create DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train_survival)
        evals = [(dtrain, 'train')]

        # Add validation set if provided
        if X_val is not None and y_val_survival is not None:
            dval = xgb.DMatrix(X_val, label=y_val_survival)
            evals.append((dval, 'eval'))

        # Initialize evals_result if not provided
        if evals_result is None:
            evals_result = {}

        # Train model using IDENTICAL parameters and logic
        model = xgb.train(
            params=self.params,
            dtrain=dtrain,
            num_boost_round=self.params.get('n_estimators', 336),
            evals=evals,
            evals_result=evals_result,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=verbose_eval,
            callbacks=callbacks or []
        )

        return model

    def _preprocess_data(self, df, time_col='OS_time', event_col='OS_event'):
        """
        Core data preprocessing - SHARED by both modes for consistency

        Args:
            df: Input DataFrame
            time_col: Column name for survival time
            event_col: Column name for event indicator

        Returns:
            Tuple: (X, y_time, y_event, feature_names) - preprocessed data
        """
        # Columns to exclude from features
        exclude_cols = [
            'OS_time', 'OS_event', 'DFS_time', 'DFS_event',
            'Overall Survival (Months)', 'Overall Survival Status',
            "Patient's Vital Status", 'Disease Free (Months)', 'Disease Free Status',
            'Has_Death_Year', 'Study ID', 'Patient ID', 'Sample ID',
            'Other Patient ID', 'Other Sample ID', 'Number of Samples Per Patient',
            'Sample Type', 'Project Identifier', 'Project Name', 'Project State',
            'American Joint Committee on Cancer Publication Version Type',
            'Oncotree Code', 'ICD-10 Classification', 'Cancer Type',
            'Cancer Type Detailed', 'Disease Type', 'Patient Primary Tumor Site',
            'Biopsy Site', 'dipLogR', 'Is FFPE',
        ]

        # Select feature columns
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        processed_df = df.copy()

        # Handle cancer staging with ordinal encoding
        staging_mappings = {
            'AJCC Pathologic T-Stage': {
                'TX': -1.0, 'T1': 1.0, 'T2': 2.0, 'T2a': 2.1, 'T2b': 2.2,
                'T3': 3.0, 'T3a': 3.1, 'T3b': 3.2, 'T4': 4.0, 'T4a': 4.1, 'T4b': 4.2
            },
            'AJCC Pathologic N-Stage': {
                'N0': 0.0, 'N1': 1.0, 'N1a': 1.1, 'N1b': 1.2, 'N2': 2.0, 'NX': -1.0
            },
            'AJCC Pathologic M-Stage': {
                'M0': 0.0, 'M1': 1.0, 'M1a': 1.1, 'M1b': 1.2, 'MX': -1.0
            },
            'AJCC Pathologic Stage': {
                'Stage I': 1.0, 'Stage IA': 1.1, 'Stage IB': 1.2,
                'Stage II': 2.0, 'Stage IIA': 2.1, 'Stage IIB': 2.2,
                'Stage III': 3.0, 'Stage IIIA': 3.1, 'Stage IIIB': 3.2, 'Stage IIIC': 3.3,
                'Stage IV': 4.0, 'Stage IVA': 4.1, 'Stage IVB': 4.2
            }
        }

        # Apply ordinal encoding to staging columns
        for col, mapping in staging_mappings.items():
            if col in processed_df.columns and col in feature_cols:
                processed_df[col] = processed_df[col].map(mapping)
                processed_df[col] = processed_df[col].fillna(0.0)

        # Handle other categorical variables
        for col in feature_cols.copy():
            if col in staging_mappings:
                continue

            if processed_df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(processed_df[col]):
                if processed_df[col].nunique() <= 1:
                    feature_cols.remove(col)
                    if col in processed_df.columns:
                        processed_df.drop(columns=[col], inplace=True)
                    continue

                # Label encoding for categorical variables
                le = LabelEncoder()
                try:
                    str_series = processed_df[col].astype(str)
                    str_series = str_series.fillna('Missing')
                    processed_df[col] = le.fit_transform(str_series)
                    self.label_encoders[col] = le
                except Exception as e:
                    if col in feature_cols:
                        feature_cols.remove(col)
                    if col in processed_df.columns:
                        processed_df.drop(columns=[col], inplace=True)

        # Handle missing values for numerical columns
        for col in feature_cols:
            if col in processed_df.columns and processed_df[col].isnull().sum() > 0:
                if pd.api.types.is_numeric_dtype(processed_df[col]):
                    fill_value = processed_df[col].median()
                    processed_df[col].fillna(fill_value, inplace=True)

        # Final feature selection
        final_feature_cols = [col for col in feature_cols if col in processed_df.columns]

        # Extract features and targets
        X = processed_df[final_feature_cols].values.astype(float)
        y_time = processed_df[time_col].values.astype(float)
        y_event = processed_df[event_col].values.astype(float)

        # Print summary (same for both modes)
        print(f"Dataset shape: {X.shape[0]} samples, {X.shape[1]} features")
        print(f"Event rate: {y_event.mean():.3f}")
        print(f"Median survival time: {np.median(y_time):.1f}")

        return X, y_time, y_event, final_feature_cols

    def prepare_data(self, df, time_col='OS_time', event_col='OS_event', test_size=0.15):
        """
        Prepare data for train/validation/test mode
        """
        # Use shared preprocessing
        X, y_time, y_event, feature_names = self._preprocess_data(df, time_col, event_col)
        self.feature_names = feature_names

        print("Mode: Train/Validation/Test split")

        # Split data
        X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test = train_test_split(
            X, y_time, y_event,
            test_size=test_size,
            random_state=42,
            stratify=y_event
        )

        # Normalize features (MinMaxScaler: 0-1 range)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Prepare survival labels for XGBoost
        y_train_survival = self.prepare_survival_data(y_time_train, y_event_train)
        y_test_survival = self.prepare_survival_data(y_time_test, y_event_test)

        return {
            'X_train': X_train_scaled,
            'X_test': X_test_scaled,
            'y_train_survival': y_train_survival,
            'y_test_survival': y_test_survival,
            'y_time_train': y_time_train,
            'y_time_test': y_time_test,
            'y_event_train': y_event_train,
            'y_event_test': y_event_test,
            'feature_names': feature_names
        }

    def prepare_data_for_cv(self, df, time_col='OS_time', event_col='OS_event'):
        """
        Prepare data for cross-validation mode (uses ALL data)
        """
        # Use IDENTICAL preprocessing as train/validation/test mode
        X, y_time, y_event, feature_names = self._preprocess_data(df, time_col, event_col)
        self.feature_names = feature_names

        print("Mode: 5-fold cross-validation (no train/test split)")

        # Normalize features using entire dataset (MinMaxScaler: 0-1 range)
        X_scaled = self.scaler.fit_transform(X)

        # Prepare survival labels for XGBoost
        y_survival = self.prepare_survival_data(y_time, y_event)

        # Create DataFrame with normalized features for analysis
        X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
        
        # Create pTMB boxplot if pTMB column exists
        self.create_ptmb_boxplot(X_scaled_df)
        
        # Save normalized dataset
        normalized_data_path = os.path.join(self.output_dir, 'normalized_dataset.csv')
        X_scaled_df.to_csv(normalized_data_path, index=False)
        print(f"Normalized dataset saved to: {normalized_data_path}")

        return {
            'X_all': X_scaled,
            'X_all_df': X_scaled_df,  # Add DataFrame version
            'y_all_survival': y_survival,
            'y_time_all': y_time,
            'y_event_all': y_event,
            'feature_names': feature_names
        }

    def train_with_validation(self, X_train, y_train_survival, y_time_train, y_event_train,
                            X_val=None, y_val_survival=None, y_time_val=None, y_event_val=None,
                            early_stopping_rounds=50, verbose=True, cindex_frequency=10):
        """
        Enhanced training with proper validation tracking
        """
        print("Starting enhanced training with validation tracking...")

        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train_survival)

        # Setup evaluation list
        evals = [(dtrain, 'train')]
        dval = None

        if X_val is not None and y_val_survival is not None:
            dval = xgb.DMatrix(X_val, label=y_val_survival)
            evals.append((dval, 'eval'))
            print(f"Validation set added: {X_val.shape[0]} samples")

        # Initialize evaluation results dictionary
        self.evals_result = {}

        # Create C-index callback
        cindex_callback = CIndexCallback(
            dtrain=dtrain,
            dval=dval,
            y_time_train=y_time_train,
            y_event_train=y_event_train,
            y_time_val=y_time_val,
            y_event_val=y_event_val,
            cindex_frequency=cindex_frequency,
            verbose=verbose
        )

        # Train model using unified training method
        print("Training XGBoost model...")
        self.model = self._train_xgb_model(
            X_train=X_train,
            y_train_survival=y_train_survival,
            X_val=X_val,
            y_val_survival=y_val_survival,
            evals_result=self.evals_result,
            callbacks=[cindex_callback],
            early_stopping_rounds=early_stopping_rounds if X_val is not None else None,
            verbose_eval=10 if verbose else None
        )

        # Store C-indices from callback
        self.training_c_indices = cindex_callback.training_c_indices
        self.validation_c_indices = cindex_callback.validation_c_indices

        # Verify data capture
        print("\nTraining completed. Data captured:")
        print(f"   - Training loss points: {len(self.evals_result.get('train', {}).get('cox-nloglik', []))}")
        print(f"   - Validation loss points: {len(self.evals_result.get('eval', {}).get('cox-nloglik', []))}")
        print(f"   - Training C-index points: {len(self.training_c_indices)}")
        print(f"   - Validation C-index points: {len(self.validation_c_indices)}")

        # Get feature importance
        self.feature_importance = self.model.get_score(importance_type='gain')

        return self.model

    def evaluate(self, X, y_time, y_event):
        """
        Evaluate model performance using C-index - SHARED by both modes for consistency
        """
        # Use identical prediction and evaluation logic
        risk_scores = self.predict(X)
        c_index = concordance_index(y_time, -risk_scores, y_event)
        return c_index

    def predict(self, X):
        """
        Predict risk scores - SHARED by both modes for consistency
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call training method first.")

        dtest = xgb.DMatrix(X)
        risk_scores = self.model.predict(dtest)
        return risk_scores

    def cross_validate(self, X, y_time, y_event, n_folds=5):
        """Perform cross-validation"""
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_scores = []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_event)):
            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_time_train_fold, y_time_val_fold = y_time[train_idx], y_time[val_idx]
            y_event_train_fold, y_event_val_fold = y_event[train_idx], y_event[val_idx]

            # Prepare survival labels
            y_train_survival = self.prepare_survival_data(y_time_train_fold, y_event_train_fold)
            y_val_survival = self.prepare_survival_data(y_time_val_fold, y_event_val_fold)

            # Train model for this fold using unified training method
            model = self._train_xgb_model(
                X_train=X_train_fold,
                y_train_survival=y_train_survival,
                X_val=None,  # No validation during CV fold training
                y_val_survival=None,
                evals_result=None,
                callbacks=None,
                early_stopping_rounds=None,
                verbose_eval=False
            )

            # Temporarily store the model for evaluation
            temp_model = self.model
            self.model = model

            # Evaluate on validation fold
            c_index = self.evaluate(X_val_fold, y_time_val_fold, y_event_val_fold)
            cv_scores.append(c_index)

            # Restore original model
            self.model = temp_model

            print(f"Fold {fold + 1}: C-index = {c_index:.4f}")

        return cv_scores

    def cross_validate_with_curves(self, X, y_time, y_event, n_folds=5):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        cv_scores = []
        fold_training_curves = []
        fold_event_stats = []

        print(f"Performing {n_folds}-fold cross-validation with training and validation curve tracking...")
        print(f"Total dataset: {len(X)} samples, {y_event.sum():.0f} events ({y_event.mean()*100:.1f}%)")
        print("=" * 80)

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_event)):
            print(f"\n--- Fold {fold + 1}/{n_folds} ---")

            X_train_fold, X_val_fold = X[train_idx], X[val_idx]
            y_time_train_fold, y_time_val_fold = y_time[train_idx], y_time[val_idx]
            y_event_train_fold, y_event_val_fold = y_event[train_idx], y_event[val_idx]

            # Calculate and display event statistics
            train_events = y_event_train_fold.sum()
            val_events = y_event_val_fold.sum()
            train_size = len(train_idx)
            val_size = len(val_idx)

            # Calculate available concordant pairs
            train_pairs = int(train_events * (train_size - train_events))
            val_pairs = int(val_events * (val_size - val_events))

            print(f"Training set: {train_size} samples, {train_events:.0f} events ({train_events/train_size*100:.1f}%)")
            print(f"Validation set: {val_size} samples, {val_events:.0f} events ({val_events/val_size*100:.1f}%)")
            print(f"Concordant pairs - Train: {train_pairs:,} | Val: {val_pairs:,} (ratio: {train_pairs/val_pairs:.1f}:1)")

            # Warning if validation events are too few
            if val_events < 30:
                print(f"WARNING: Only {val_events:.0f} events in validation set - C-index may be unstable!")

            # Store statistics data
            fold_event_stats.append({
                'fold': fold + 1,
                'train_size': train_size,
                'train_events': train_events,
                'train_event_rate': train_events/train_size,
                'val_size': val_size,
                'val_events': val_events,
                'val_event_rate': val_events/val_size,
                'train_pairs': train_pairs,
                'val_pairs': val_pairs
            })

            # Prepare survival labels
            y_train_survival = self.prepare_survival_data(y_time_train_fold, y_event_train_fold)
            y_val_survival = self.prepare_survival_data(y_time_val_fold, y_event_val_fold)

            # Create DMatrix
            dtrain = xgb.DMatrix(X_train_fold, label=y_train_survival)
            dval = xgb.DMatrix(X_val_fold, label=y_val_survival)

            # Create enhanced callback for this fold (with validation tracking)
            fold_callback = CVFoldCallback(
                dtrain=dtrain,
                dval=dval,
                y_time_train=y_time_train_fold,
                y_event_train=y_event_train_fold,
                y_time_val=y_time_val_fold,
                y_event_val=y_event_val_fold,
                cindex_frequency=10,
                verbose=False
            )

            # Initialize fold evals_result
            fold_evals_result = {}

            # Train model for this fold with validation tracking
            model = self._train_xgb_model(
                X_train=X_train_fold,
                y_train_survival=y_train_survival,
                X_val=X_val_fold,  # Include validation set
                y_val_survival=y_val_survival,
                evals_result=fold_evals_result,
                callbacks=[fold_callback],
                early_stopping_rounds=None,  # No early stopping in CV
                verbose_eval=False
            )

            # Final validation C-index
            val_pred = model.predict(dval)
            final_val_c_index = concordance_index(y_time_val_fold, -val_pred, y_event_val_fold)
            cv_scores.append(final_val_c_index)

            # Store fold training curves (with event stats)
            fold_training_curves.append({
                'fold': fold + 1,
                'training_losses': fold_callback.training_losses,
                'validation_losses': fold_callback.validation_losses,
                'training_c_indices': fold_callback.training_c_indices,
                'validation_c_indices': fold_callback.validation_c_indices,
                'final_validation_c_index': final_val_c_index,
                'final_training_c_index': fold_callback.training_c_indices[-1] if fold_callback.training_c_indices else None,
                'val_events': val_events,
                'val_size': val_size,
                'val_pairs': val_pairs
            })

            print(f"Fold {fold + 1}: Final training C-index = {fold_callback.training_c_indices[-1] if fold_callback.training_c_indices else 'N/A':.4f}")
            print(f"Fold {fold + 1}: Final validation C-index = {final_val_c_index:.4f}")

        # Display event statistics summary
        print("\n" + "=" * 80)
        print("Event Distribution Summary Across Folds")
        print("=" * 80)
        print("Fold | Val Size | Val Events | Event Rate | Concordant Pairs | Final C-idx")
        print("-" * 75)
        for i, stats in enumerate(fold_event_stats):
            print(f"{stats['fold']:4d} | {stats['val_size']:8d} | {stats['val_events']:10.0f} | "
                  f"{stats['val_event_rate']*100:9.1f}% | {stats['val_pairs']:16,d} | {cv_scores[i]:.4f}")

        # Calculate correlations
        val_events_list = [stats['val_events'] for stats in fold_event_stats]
        val_pairs_list = [stats['val_pairs'] for stats in fold_event_stats]

        if len(set(val_events_list)) > 1:  # Check if there is variation
            corr_events, p_events = pearsonr(val_events_list, cv_scores)
            corr_pairs, p_pairs = pearsonr(val_pairs_list, cv_scores)
            print(f"\nCorrelation Analysis:")
            print(f"  Val Events vs C-index: r={corr_events:.3f}, p={p_events:.3f}")
            print(f"  Concordant Pairs vs C-index: r={corr_pairs:.3f}, p={p_pairs:.3f}")

            if abs(corr_events) > 0.7 or abs(corr_pairs) > 0.7:
                print("  WARNING: Strong correlation detected - consider using 3-fold CV for more stable results!")

        return {
            'cv_scores': cv_scores,
            'training_curves': fold_training_curves,
            'event_stats': fold_event_stats
        }

    def train_final_model_cv(self, X_all, y_all_survival):
        """Train final model on all data for feature importance analysis"""
        print("Training final model on entire dataset...")

        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_all, label=y_all_survival)

        # Train model using unified training method
        self.model = self._train_xgb_model(
            X_train=X_all,
            y_train_survival=y_all_survival,
            X_val=None,
            y_val_survival=None,
            evals_result=None,
            callbacks=None,
            early_stopping_rounds=None,
            verbose_eval=10
        )

        # Get feature importance
        self.feature_importance = self.model.get_score(importance_type='gain')

        print("Final model training completed!")
        return self.model

    def plot_training_curves(self, prefix='train_val_test'):
        """
        Plot training and validation curves (both loss and C-index) as separate PDFs
        """
        if not hasattr(self, 'evals_result') or not self.evals_result:
            print("No evaluation results available.")
            return

        # Extract loss data
        train_loss = self.evals_result.get('train', {}).get('cox-nloglik', [])
        val_loss = self.evals_result.get('eval', {}).get('cox-nloglik', [])

        # Plot 1: Loss curves
        plt.figure(figsize=(10, 6))
        if train_loss:
            epochs_loss = range(1, len(train_loss) + 1)
            plt.plot(epochs_loss, train_loss, 'b-', label='Training Loss', linewidth=2)

            if val_loss:
                plt.plot(epochs_loss[:len(val_loss)], val_loss, 'r-', label='Validation Loss', linewidth=2)

                # Add text with final values
                plt.text(0.02, 0.98, f'Final Train Loss: {train_loss[-1]:.4f}',
                        transform=plt.gca().transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                plt.text(0.02, 0.88, f'Final Valid Loss: {val_loss[-1]:.4f}',
                        transform=plt.gca().transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

        plt.xlabel('Iteration (Boosting Round)', fontsize=12)
        plt.ylabel('Cox Negative Log-Likelihood', fontsize=12)
        plt.title('Training and Validation Loss Curves', fontsize=14)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        loss_path = os.path.join(self.output_dir, f'{prefix}_loss_curves.pdf')
        plt.savefig(loss_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Loss curves saved to: {loss_path}")

        # Plot 2: C-index curves
        plt.figure(figsize=(10, 6))
        if hasattr(self, 'training_c_indices') and self.training_c_indices:
            epochs_c = range(1, len(self.training_c_indices) + 1)
            plt.plot(epochs_c, self.training_c_indices, 'b-o',
                    label='Training C-index', linewidth=2, markersize=4)

            if hasattr(self, 'validation_c_indices') and self.validation_c_indices:
                plt.plot(epochs_c[:len(self.validation_c_indices)], self.validation_c_indices, 'r-o',
                        label='Validation C-index', linewidth=2, markersize=4)

                # Add text with final values
                plt.text(0.02, 0.98, f'Final Train C-idx: {self.training_c_indices[-1]:.4f}',
                        transform=plt.gca().transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
                plt.text(0.02, 0.88, f'Final Valid C-idx: {self.validation_c_indices[-1]:.4f}',
                        transform=plt.gca().transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

            plt.xlabel('C-Index Measurement Points', fontsize=12)
            plt.ylabel('C-Index', fontsize=12)
            plt.title('Training and Validation C-Index Curves', fontsize=14)
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.ylim([0.5, 1.0])

        plt.tight_layout()
        cindex_path = os.path.join(self.output_dir, f'{prefix}_cindex_curves.pdf')
        plt.savefig(cindex_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"C-index curves saved to: {cindex_path}")

        # Print summary
        print("\nTraining Summary:")
        print(f"   - Total iterations: {len(train_loss) if train_loss else 0}")
        if train_loss:
            print(f"   - Final training loss: {train_loss[-1]:.6f}")
        if val_loss:
            print(f"   - Final validation loss: {val_loss[-1]:.6f}")
        if hasattr(self, 'training_c_indices') and self.training_c_indices:
            print(f"   - Final training C-index: {self.training_c_indices[-1]:.4f}")
        if hasattr(self, 'validation_c_indices') and self.validation_c_indices:
            print(f"   - Final validation C-index: {self.validation_c_indices[-1]:.4f}")

    def plot_cv_training_curves(self, fold_training_curves, prefix='cv_mode'):
        """
        Plot cross-validation training curves as individual PDF files
        """
        if not fold_training_curves:
            print("No CV training curves available.")
            return

        n_folds = len(fold_training_curves)
        colors = plt.cm.tab10(np.linspace(0, 1, n_folds))

        # Plot 1: Training loss curves for all folds
        plt.figure(figsize=(12, 8))
        for i, fold_data in enumerate(fold_training_curves):
            fold_num = fold_data['fold']
            train_losses = fold_data['training_losses']
            if train_losses:
                epochs = range(1, len(train_losses) + 1)
                plt.plot(epochs, train_losses, color=colors[i],
                        label=f'Fold {fold_num}', alpha=0.7, linewidth=1.5)

        plt.xlabel('Iteration (Boosting Round)')
        plt.ylabel('Cox Negative Log-Likelihood')
        plt.title('Training Loss Curves - All Folds')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        train_loss_path = os.path.join(self.output_dir, f'{prefix}_training_loss_curves.pdf')
        plt.savefig(train_loss_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training loss curves saved to: {train_loss_path}")

        # Plot 2: Validation loss curves for all folds
        plt.figure(figsize=(12, 8))
        for i, fold_data in enumerate(fold_training_curves):
            fold_num = fold_data['fold']
            val_losses = fold_data['validation_losses']
            if val_losses:
                epochs = range(1, len(val_losses) + 1)
                plt.plot(epochs, val_losses, color=colors[i],
                        label=f'Fold {fold_num}', alpha=0.7, linewidth=1.5)

        plt.xlabel('Iteration (Boosting Round)')
        plt.ylabel('Cox Negative Log-Likelihood')
        plt.title('Validation Loss Curves - All Folds')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        val_loss_path = os.path.join(self.output_dir, f'{prefix}_validation_loss_curves.pdf')
        plt.savefig(val_loss_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Validation loss curves saved to: {val_loss_path}")

        # Plot 3: Training C-index curves for all folds
        plt.figure(figsize=(12, 8))
        for i, fold_data in enumerate(fold_training_curves):
            fold_num = fold_data['fold']
            train_c_indices = fold_data['training_c_indices']
            if train_c_indices:
                epochs_c = range(1, len(train_c_indices) + 1)
                plt.plot(epochs_c, train_c_indices, color=colors[i], marker='o',
                        label=f'Fold {fold_num}', alpha=0.7, linewidth=1.5, markersize=3)

        plt.xlabel('C-Index Measurement Points')
        plt.ylabel('Training C-Index')
        plt.title('Training C-Index Curves - All Folds')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([0.5, 1.0])
        plt.tight_layout()

        train_cindex_path = os.path.join(self.output_dir, f'{prefix}_training_cindex_curves.pdf')
        plt.savefig(train_cindex_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Training C-index curves saved to: {train_cindex_path}")

        # Plot 4: Validation C-index curves for all folds
        plt.figure(figsize=(12, 8))
        for i, fold_data in enumerate(fold_training_curves):
            fold_num = fold_data['fold']
            val_c_indices = fold_data['validation_c_indices']
            if val_c_indices:
                epochs_c = range(1, len(val_c_indices) + 1)
                plt.plot(epochs_c, val_c_indices, color=colors[i], marker='s',
                        label=f'Fold {fold_num}', alpha=0.7, linewidth=1.5, markersize=3)

        plt.xlabel('C-Index Measurement Points')
        plt.ylabel('Validation C-Index')
        plt.title('Validation C-Index Curves - All Folds')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([0.5, 1.0])
        plt.tight_layout()

        val_cindex_path = os.path.join(self.output_dir, f'{prefix}_validation_cindex_curves.pdf')
        plt.savefig(val_cindex_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Validation C-index curves saved to: {val_cindex_path}")

        # Plot 5: Final validation C-index for each fold
        plt.figure(figsize=(12, 8))
        fold_numbers = [fold_data['fold'] for fold_data in fold_training_curves]
        val_c_indices = [fold_data['final_validation_c_index'] for fold_data in fold_training_curves]
        train_c_indices_final = [fold_data['final_training_c_index'] for fold_data in fold_training_curves]

        x = np.arange(len(fold_numbers))
        width = 0.35

        bars1 = plt.bar(x - width/2, train_c_indices_final, width,
                       label='Final Training C-index', color='lightblue', alpha=0.7)
        bars2 = plt.bar(x + width/2, val_c_indices, width,
                       label='Final Validation C-index', color='lightcoral', alpha=0.7)

        # Add value labels on bars
        for bar, val in zip(bars1, train_c_indices_final):
            if val is not None:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                        f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        for bar, val in zip(bars2, val_c_indices):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9)

        plt.xlabel('Fold Number')
        plt.ylabel('C-Index')
        plt.title('Final Training vs Validation C-Index by Fold')
        plt.xticks(x, fold_numbers)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        final_comparison_path = os.path.join(self.output_dir, f'{prefix}_final_comparison_by_fold.pdf')
        plt.savefig(final_comparison_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Final comparison by fold saved to: {final_comparison_path}")

        # Plot 6: Summary statistics
        plt.figure(figsize=(10, 8))
        mean_val_c = np.mean(val_c_indices)
        std_val_c = np.std(val_c_indices)
        mean_train_c = np.mean([x for x in train_c_indices_final if x is not None])
        std_train_c = np.std([x for x in train_c_indices_final if x is not None])

        # Box plot
        bp1 = plt.boxplot([train_c_indices_final], positions=[1], patch_artist=True,
                         boxprops=dict(facecolor='lightblue', alpha=0.7),
                         widths=0.6)
        bp2 = plt.boxplot([val_c_indices], positions=[2], patch_artist=True,
                         boxprops=dict(facecolor='lightcoral', alpha=0.7),
                         widths=0.6)

        # Add statistics text
        stats_text = f'Training:\nMean: {mean_train_c:.4f}\nStd: {std_train_c:.4f}\n\nValidation:\nMean: {mean_val_c:.4f}\nStd: {std_val_c:.4f}'
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        plt.ylabel('C-Index')
        plt.title('CV Performance Summary')
        plt.grid(True, alpha=0.3)
        plt.xticks([1, 2], ['Training', 'Validation'])
        plt.tight_layout()

        summary_path = os.path.join(self.output_dir, f'{prefix}_cv_summary.pdf')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"CV performance summary saved to: {summary_path}")

        # Print comprehensive summary
        print("\nEnhanced CV Training Summary:")
        print(f"   - Number of folds: {n_folds}")
        print(f"   - Mean training C-index: {mean_train_c:.4f} ± {std_train_c:.4f}")
        print(f"   - Mean validation C-index: {mean_val_c:.4f} ± {std_val_c:.4f}")
        print(f"   - Best validation fold: Fold {fold_numbers[np.argmax(val_c_indices)]} (C-index: {max(val_c_indices):.4f})")
        print(f"   - Worst validation fold: Fold {fold_numbers[np.argmin(val_c_indices)]} (C-index: {min(val_c_indices):.4f})")

        # Print detailed fold-by-fold results
        print("\nDetailed Fold Results:")
        print("Fold | Final Train C-idx | Final Valid C-idx")
        print("-" * 45)
        for fold_data in fold_training_curves:
            fold_num = fold_data['fold']
            train_c = fold_data['final_training_c_index']
            val_c = fold_data['final_validation_c_index']
            train_str = f"{train_c:.4f}" if train_c is not None else "N/A"
            print(f"{fold_num:4d} | {train_str:17s} | {val_c:.4f}")

    def calculate_delong_ci(self, y_time, y_event, risk_scores, alpha=0.05):
        """
        Calculate 95% confidence interval for C-index using DeLong method

        Args:
            y_time: Survival times
            y_event: Event indicators
            risk_scores: Predicted risk scores
            alpha: Significance level (0.05 for 95% CI)

        Returns:
            dict: C-index, lower CI, upper CI, standard error
        """
        try:
            n = len(y_time)
            c_index = concordance_index(y_time, -risk_scores, y_event)

            # Calculate concordant and discordant pairs
            concordant_pairs = 0
            discordant_pairs = 0
            tied_pairs = 0
            total_pairs = 0

            for i in range(n):
                for j in range(i+1, n):
                    # Only consider pairs where at least one has an event
                    if y_event[i] == 0 and y_event[j] == 0:
                        continue

                    total_pairs += 1

                    # Determine which observation should have higher risk
                    if y_time[i] < y_time[j]:
                        # i had event first, should have higher risk score
                        if risk_scores[i] > risk_scores[j]:
                            concordant_pairs += 1
                        elif risk_scores[i] < risk_scores[j]:
                            discordant_pairs += 1
                        else:
                            tied_pairs += 1
                    elif y_time[i] > y_time[j]:
                        # j had event first, should have higher risk score
                        if risk_scores[j] > risk_scores[i]:
                            concordant_pairs += 1
                        elif risk_scores[j] < risk_scores[i]:
                            discordant_pairs += 1
                        else:
                            tied_pairs += 1
                    else:
                        # Same time, tied
                        tied_pairs += 1

            # Calculate variance using DeLong method approximation
            if total_pairs > 0:
                # Simplified variance calculation
                se = np.sqrt((c_index * (1 - c_index)) / total_pairs)
            else:
                se = 0

            # Calculate 95% confidence interval
            z_score = stats.norm.ppf(1 - alpha/2)
            ci_lower = max(0, c_index - z_score * se)
            ci_upper = min(1, c_index + z_score * se)

            return {
                'c_index': c_index,
                'se': se,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'total_pairs': total_pairs
            }

        except Exception as e:
            print(f"Error calculating DeLong CI: {e}")
            return {
                'c_index': concordance_index(y_time, -risk_scores, y_event),
                'se': None,
                'ci_lower': None,
                'ci_upper': None,
                'total_pairs': None
            }

    def bootstrap_c_index_ci(self, X, y_time, y_event, n_bootstrap=1000, alpha=0.05):
        """
        Calculate 95% confidence interval for C-index using bootstrap method

        Args:
            X: Feature matrix
            y_time: Survival times
            y_event: Event indicators
            n_bootstrap: Number of bootstrap samples
            alpha: Significance level

        Returns:
            dict: C-index statistics with confidence interval
        """
        if self.model is None:
            raise ValueError("Model not trained yet.")

        n_samples = len(X)
        bootstrap_c_indices = []

        print(f"Performing {n_bootstrap} bootstrap samples for CI calculation...")

        for i in range(n_bootstrap):
            # Bootstrap sampling with replacement
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_time_boot = y_time[indices]
            y_event_boot = y_event[indices]

            # Predict on bootstrap sample
            risk_scores_boot = self.predict(X_boot)

            # Calculate C-index for bootstrap sample
            try:
                c_index_boot = concordance_index(y_time_boot, -risk_scores_boot, y_event_boot)
                bootstrap_c_indices.append(c_index_boot)
            except:
                continue

        bootstrap_c_indices = np.array(bootstrap_c_indices)

        # Calculate statistics
        mean_c_index = np.mean(bootstrap_c_indices)
        std_c_index = np.std(bootstrap_c_indices)

        # Calculate percentile-based confidence interval
        ci_lower = np.percentile(bootstrap_c_indices, 100 * alpha/2)
        ci_upper = np.percentile(bootstrap_c_indices, 100 * (1 - alpha/2))

        return {
            'c_index': mean_c_index,
            'std': std_c_index,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'bootstrap_samples': len(bootstrap_c_indices)
        }

    def calculate_cv_conservative_ci(self, cv_scores, alpha=0.05):
        """
        Calculate conservative 95% CI based on CV results using t-distribution
        Note: This is a conservative approximation since CV folds are not independent

        Args:
            cv_scores: List of C-index scores from cross-validation
            alpha: Significance level

        Returns:
            dict: C-index statistics with confidence interval
        """
        cv_scores = np.array(cv_scores)
        n = len(cv_scores)
        mean_c = np.mean(cv_scores)
        std_c = np.std(cv_scores, ddof=1)  # Sample standard deviation
        se_c = std_c / np.sqrt(n)

        # Use t-distribution for small sample size
        df = n - 1
        t_value = stats.t.ppf(1 - alpha/2, df)

        ci_lower = mean_c - t_value * se_c
        ci_upper = mean_c + t_value * se_c

        return {
            'c_index': mean_c,
            'std': std_c,
            'se': se_c,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'df': df,
            't_value': t_value,
            'note': 'Conservative estimate - CV folds are not independent'
        }

    def plot_cv_summary_bars_with_ci(self, cv_scores, X_test=None, y_time_test=None, y_event_test=None, prefix='cv_summary_bars'):
        """
        Create a cross-validation summary plot with confidence intervals

        Args:
            cv_scores: List of CV C-index scores
            X_test: Test features for bootstrap CI (optional)
            y_time_test: Test survival times for bootstrap CI (optional)
            y_event_test: Test event indicators for bootstrap CI (optional)
        """
        if not cv_scores:
            print("No cross-validation scores available.")
            return

        # Calculate basic CV statistics
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)

        # Calculate conservative CI based on CV scores
        cv_ci_results = self.calculate_cv_conservative_ci(cv_scores)

        # Try to calculate more accurate CI if test data is provided
        bootstrap_ci_results = None
        delong_ci_results = None

        if X_test is not None and y_time_test is not None and y_event_test is not None and self.model is not None:
            print("Calculating more accurate confidence intervals...")

            # Bootstrap CI
            try:
                bootstrap_ci_results = self.bootstrap_c_index_ci(X_test, y_time_test, y_event_test, n_bootstrap=1000)
                print(f"Bootstrap CI: [{bootstrap_ci_results['ci_lower']:.4f}, {bootstrap_ci_results['ci_upper']:.4f}]")
            except Exception as e:
                print(f"Bootstrap CI calculation failed: {e}")

            # DeLong CI
            try:
                risk_scores = self.predict(X_test)
                delong_ci_results = self.calculate_delong_ci(y_time_test, y_event_test, risk_scores)
                if delong_ci_results['ci_lower'] is not None:
                    print(f"DeLong CI: [{delong_ci_results['ci_lower']:.4f}, {delong_ci_results['ci_upper']:.4f}]")
            except Exception as e:
                print(f"DeLong CI calculation failed: {e}")

        # Create figure
        plt.figure(figsize=(14, 10))

        # Create bars with color gradient from light blue to darker blue
        n_folds = len(cv_scores)
        fold_numbers = range(1, n_folds + 1)
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, n_folds))
        bars = plt.bar(fold_numbers, cv_scores, color=colors,
                      alpha=0.8, edgecolor='navy', linewidth=1.5)

        # Add value labels on top of bars
        for i, (bar, score) in enumerate(zip(bars, cv_scores)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{score:.4f}', ha='center', va='bottom',
                    fontsize=12, fontweight='bold')

        # Add mean line (red dashed)
        plt.axhline(y=cv_mean, color='red', linestyle='--', linewidth=2,
                   label=f'CV Mean: {cv_mean:.4f}')

        # Add CV-based confidence interval shading (±1 SD)
        plt.axhspan(cv_ci_results['ci_lower'], cv_ci_results['ci_upper'],
                   alpha=0.15, color='red',
                   label=f'CV 95% CI: [{cv_ci_results["ci_lower"]:.4f}, {cv_ci_results["ci_upper"]:.4f}]')

        # Add bootstrap CI if available
        if bootstrap_ci_results:
            plt.axhspan(bootstrap_ci_results['ci_lower'], bootstrap_ci_results['ci_upper'],
                       alpha=0.15, color='green',
                       label=f'Bootstrap 95% CI: [{bootstrap_ci_results["ci_lower"]:.4f}, {bootstrap_ci_results["ci_upper"]:.4f}]')

        # Add DeLong CI if available
        if delong_ci_results and delong_ci_results['ci_lower'] is not None:
            plt.axhspan(delong_ci_results['ci_lower'], delong_ci_results['ci_upper'],
                       alpha=0.15, color='orange',
                       label=f'DeLong 95% CI: [{delong_ci_results["ci_lower"]:.4f}, {delong_ci_results["ci_upper"]:.4f}]')

        # Customize plot
        plt.xlabel('Fold', fontsize=14)
        plt.ylabel('Final C-index', fontsize=14)
        plt.title('Cross-Validation Results Summary with 95% Confidence Intervals', fontsize=16, fontweight='bold')
        plt.xticks(fold_numbers, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, alpha=0.3)

        # Set y-axis limits with some padding
        all_values = cv_scores.copy()
        if bootstrap_ci_results:
            all_values.extend([bootstrap_ci_results['ci_lower'], bootstrap_ci_results['ci_upper']])
        if delong_ci_results and delong_ci_results['ci_lower'] is not None:
            all_values.extend([delong_ci_results['ci_lower'], delong_ci_results['ci_upper']])
        all_values.extend([cv_ci_results['ci_lower'], cv_ci_results['ci_upper']])

        y_min = max(0, min(all_values) - 0.02)
        y_max = min(1.0, max(all_values) + 0.02)
        plt.ylim([y_min, y_max])

        # Add legend
        plt.legend(loc='upper left', fontsize=10, framealpha=0.9)

        # Create comprehensive statistics text
        stats_lines = [
            f'Cross-Validation Results:',
            f'Mean C-index: {cv_mean:.4f} ± {cv_std:.4f}',
            f'Range: [{min(cv_scores):.4f}, {max(cv_scores):.4f}]',
            f'Best Fold: {np.argmax(cv_scores)+1} ({max(cv_scores):.4f})',
            f'Worst Fold: {np.argmin(cv_scores)+1} ({min(cv_scores):.4f})',
            f'',
            f'95% Confidence Intervals:',
            f'CV-based (conservative): [{cv_ci_results["ci_lower"]:.4f}, {cv_ci_results["ci_upper"]:.4f}]'
        ]

        if bootstrap_ci_results:
            stats_lines.append(f'Bootstrap: [{bootstrap_ci_results["ci_lower"]:.4f}, {bootstrap_ci_results["ci_upper"]:.4f}]')

        if delong_ci_results and delong_ci_results['ci_lower'] is not None:
            stats_lines.append(f'DeLong: [{delong_ci_results["ci_lower"]:.4f}, {delong_ci_results["ci_upper"]:.4f}]')

        stats_text = '\n'.join(stats_lines)

        plt.text(0.98, 0.02, stats_text, transform=plt.gca().transAxes,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.9),
                fontsize=9)

        plt.tight_layout()

        # Save as PDF
        summary_path = os.path.join(self.output_dir, f'{prefix}_with_ci.pdf')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
    def create_cv_results_table_image(self, cv_scores, fold_training_curves=None, prefix='cv_table_image'):
        """
        Create a publication-ready table image showing CV results

        Args:
            cv_scores: List of validation C-index scores from CV
            fold_training_curves: Training curves data with training C-indices
            prefix: File prefix for saved table image
        """

        # Calculate statistics
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores, ddof=1)
        cv_ci_results = self.calculate_cv_conservative_ci(cv_scores)

        # Extract training C-indices if available
        training_c_indices = []
        if fold_training_curves:
            for fold_data in fold_training_curves:
                train_c = fold_data.get('final_training_c_index')
                training_c_indices.append(train_c if train_c is not None else np.nan)

        # Calculate training statistics if available
        train_mean = np.nan
        train_std = np.nan
        train_ci_lower = np.nan
        train_ci_upper = np.nan

        if training_c_indices and not all(np.isnan(training_c_indices)):
            train_mean = np.nanmean(training_c_indices)
            train_std = np.nanstd(training_c_indices, ddof=1)
            train_stats = self.calculate_cv_conservative_ci(training_c_indices)
            train_ci_lower = train_stats['ci_lower']
            train_ci_upper = train_stats['ci_upper']

        # Create figure and table
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')

        # Prepare table data
        n_folds = len(cv_scores)
        table_data = []

        # Header row
        headers = ['Fold', 'Training C-index', 'Validation C-index']

        # Individual fold rows
        for i in range(n_folds):
            fold_num = str(i + 1)

            if i < len(training_c_indices) and not np.isnan(training_c_indices[i]):
                train_val = f'{training_c_indices[i]:.4f}'
            else:
                train_val = 'N/A'

            val_val = f'{cv_scores[i]:.4f}'

            table_data.append([fold_num, train_val, val_val])

        # Mean row - match the exact format from user's image
        if not np.isnan(train_mean):
            train_mean_str = f'{train_mean:.4f} ± {train_std:.4f}'
        else:
            train_mean_str = ''  # Leave empty like in the image

        # Use exact format from user's image: 0.6492 ± 0.0541
        val_mean_str = f'{cv_mean:.4f} ± {cv_std:.4f}'
        table_data.append(['Mean', train_mean_str, val_mean_str])

        # 95% CI row - leave empty to be filled manually like in the image
        table_data.append(['95% CI', '', ''])  # Empty cells for manual filling

        # 95% CI row - create filled and blank versions
        if not np.isnan(train_ci_lower):
            train_ci_str = f'[{train_ci_lower:.4f}, {train_ci_upper:.4f}]'
        else:
            train_ci_str = ''

        val_ci_str = f'[{cv_ci_results["ci_lower"]:.4f}, {cv_ci_results["ci_upper"]:.4f}]'

        # First create version with filled CI values
        table_data_filled = table_data.copy()
        table_data_filled.append(['95% CI', train_ci_str, val_ci_str])

        # Create table with filled CI
        self._create_table_plot(table_data_filled, headers, f'{prefix}_filled')

        # Then create version with blank CI for manual filling
        table_data_blank = table_data.copy()
        table_data_blank.append(['95% CI', '', ''])

        # Create table with blank CI
        self._create_table_plot(table_data_blank, headers, f'{prefix}_blank')

        # Print the calculated values for reference
        print(f"\nCalculated 95% Confidence Intervals:")
        print(f"Training C-index 95% CI: {train_ci_str if train_ci_str else 'N/A'}")
        print(f"Validation C-index 95% CI: {val_ci_str}")

        return {
            'validation_mean': cv_mean,
            'validation_std': cv_std,
            'validation_ci': [cv_ci_results['ci_lower'], cv_ci_results['ci_upper']],
            'training_mean': train_mean if not np.isnan(train_mean) else None,
            'training_std': train_std if not np.isnan(train_std) else None,
            'training_ci': [train_ci_lower, train_ci_upper] if not np.isnan(train_ci_lower) else None
        }

    def _create_table_plot(self, table_data, headers, prefix):
        """Helper function to create table plot"""
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')

        # Create table
        table = ax.table(cellText=table_data,
                        colLabels=headers,
                        cellLoc='center',
                        loc='center',
                        cellColours=None)

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)

        n_folds = len(table_data) - 2  # Subtract Mean and 95% CI rows

        # Style header row
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
            table[(0, i)].set_height(0.15)
            table[(0, i)].set_edgecolor('black')
            table[(0, i)].set_linewidth(1.5)

        # Style data rows (individual folds)
        for i in range(1, n_folds + 1):
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#f0f0f0' if i % 2 == 0 else 'white')
                table[(i, j)].set_height(0.12)
                table[(i, j)].set_edgecolor('black')
                table[(i, j)].set_linewidth(1)

        # Style summary rows (Mean and 95% CI)
        for i in range(n_folds + 1, n_folds + 3):
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#e3f2fd')
                table[(i, j)].set_text_props(weight='bold')
                table[(i, j)].set_height(0.15)
                table[(i, j)].set_edgecolor('black')
                table[(i, j)].set_linewidth(1.5)

        # Set title
        plt.title('Cross-Validation Results Summary', fontsize=16, fontweight='bold', pad=20)

        # Save as PDF
        table_path = os.path.join(self.output_dir, f'{prefix}.pdf')
        plt.savefig(table_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Table image saved to: {table_path}")

    def _create_simple_validation_table(self, cv_scores, cv_ci_results, prefix='validation_table'):
        """Create a simplified table with only validation results for main paper"""

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis('tight')
        ax.axis('off')

        # Calculate stats
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores, ddof=1)

        # Prepare simplified table data
        headers = ['Method', 'C-index (Mean ± SD)', '95% CI', 'Folds']
        table_data = [
            ['XGBoost-Cox',
             f'{cv_mean:.3f} ± {cv_std:.3f}',
             f'[{cv_ci_results["ci_lower"]:.3f}, {cv_ci_results["ci_upper"]:.3f}]',
             str(len(cv_scores))]
        ]

        # Create table
        table = ax.table(cellText=table_data,
                        colLabels=headers,
                        cellLoc='center',
                        loc='center')

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2)

        # Style header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#2196F3')
            table[(0, i)].set_text_props(weight='bold', color='white')
            table[(0, i)].set_height(0.2)

        # Style data row
        for j in range(len(headers)):
            table[(1, j)].set_facecolor('#f8f9fa')
            table[(1, j)].set_height(0.15)
            table[(1, j)].set_edgecolor('black')
            table[(1, j)].set_linewidth(1)

        # Header borders
        for i in range(len(headers)):
            table[(0, i)].set_edgecolor('black')
            table[(0, i)].set_linewidth(1)

        plt.title('Cross-Validation Performance Summary', fontsize=14, fontweight='bold', pad=20)

        # Save as PDF
        simple_table_path = os.path.join(self.output_dir, f'{prefix}.pdf')
        plt.savefig(simple_table_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Simple validation table saved to: {simple_table_path}")

    def create_detailed_comparison_table_image(self, cv_scores, fold_training_curves, prefix='detailed_comparison_table'):
        """
        Create detailed training vs validation comparison table image with 95% CI

        Args:
            cv_scores: List of validation C-index scores
            fold_training_curves: Training curves data with training C-indices
            prefix: File prefix for saved table
        """
        if not fold_training_curves:
            print("No training curve data available for detailed comparison.")
            return

        # Calculate statistics
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores, ddof=1)

        # Extract training C-indices
        training_c_indices = []
        for fold_data in fold_training_curves:
            train_c = fold_data.get('final_training_c_index')
            training_c_indices.append(train_c if train_c is not None else np.nan)

        # Calculate differences
        differences = []
        for train_c, val_c in zip(training_c_indices, cv_scores):
            if not np.isnan(train_c):
                differences.append(train_c - val_c)
            else:
                differences.append(np.nan)

        # Calculate statistics
        train_mean = np.nanmean(training_c_indices)
        train_std = np.nanstd(training_c_indices, ddof=1)
        diff_mean = np.nanmean(differences)
        diff_std = np.nanstd(differences, ddof=1)

        # Calculate 95% CI
        train_ci = self.calculate_cv_conservative_ci(training_c_indices) if not all(np.isnan(training_c_indices)) else None
        val_ci = self.calculate_cv_conservative_ci(cv_scores)
        diff_ci = self.calculate_cv_conservative_ci(differences) if not all(np.isnan(differences)) else None

        # Create figure
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.axis('tight')
        ax.axis('off')

        # Prepare table data
        headers = ['Fold', 'Training C-index', 'Validation C-index', 'Difference (Train-Val)']
        table_data = []

        # Individual fold rows
        for i, (train_c, val_c, diff) in enumerate(zip(training_c_indices, cv_scores, differences)):
            fold_num = str(i + 1)
            train_str = f'{train_c:.3f}' if not np.isnan(train_c) else 'N/A'
            val_str = f'{val_c:.3f}'
            diff_str = f'{diff:+.3f}' if not np.isnan(diff) else 'N/A'
            table_data.append([fold_num, train_str, val_str, diff_str])

        # Mean ± SD row
        train_mean_str = f'{train_mean:.3f} ± {train_std:.3f}' if not np.isnan(train_mean) else 'N/A'
        val_mean_str = f'{cv_mean:.3f} ± {cv_std:.3f}'
        diff_mean_str = f'{diff_mean:+.3f} ± {diff_std:.3f}' if not np.isnan(diff_mean) else 'N/A'
        table_data.append(['Mean ± SD', train_mean_str, val_mean_str, diff_mean_str])

        # 95% CI row
        train_ci_str = f'[{train_ci["ci_lower"]:.3f}, {train_ci["ci_upper"]:.3f}]' if train_ci else 'N/A'
        val_ci_str = f'[{val_ci["ci_lower"]:.3f}, {val_ci["ci_upper"]:.3f}]'
        diff_ci_str = f'[{diff_ci["ci_lower"]:.3f}, {diff_ci["ci_upper"]:.3f}]' if diff_ci else 'N/A'
        table_data.append(['95% CI', train_ci_str, val_ci_str, diff_ci_str])

        # Create table
        table = ax.table(cellText=table_data,
                        colLabels=headers,
                        cellLoc='center',
                        loc='center')

        # Style the table
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)

        n_folds = len(cv_scores)

        # Style header row
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#2196F3')
            table[(0, i)].set_text_props(weight='bold', color='white')
            table[(0, i)].set_height(0.15)
            table[(0, i)].set_edgecolor('black')
            table[(0, i)].set_linewidth(1.5)

        # Style individual fold rows
        for i in range(1, n_folds + 1):
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')
                table[(i, j)].set_height(0.12)
                table[(i, j)].set_edgecolor('black')
                table[(i, j)].set_linewidth(1)

        # Style summary rows (Mean and 95% CI)
        for i in range(n_folds + 1, n_folds + 3):
            for j in range(len(headers)):
                table[(i, j)].set_facecolor('#fff3e0')
                table[(i, j)].set_text_props(weight='bold')
                table[(i, j)].set_height(0.15)
                table[(i, j)].set_edgecolor('black')
                table[(i, j)].set_linewidth(1.5)

        # Set title
        plt.title('Training vs Validation Performance by Fold (with 95% CI)', fontsize=14, fontweight='bold', pad=20)

        # Save as PDF
        detailed_table_path = os.path.join(self.output_dir, f'{prefix}.pdf')
        plt.savefig(detailed_table_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"Detailed comparison table image saved to: {detailed_table_path}")

    def generate_cv_results_tables(self, cv_scores, fold_training_curves=None, prefix='cv_results_table'):
        """
        Generate publication-ready tables for cross-validation results

        Args:
            cv_scores: List of validation C-index scores from CV
            fold_training_curves: Training curves data with training C-indices
            prefix: File prefix for saved tables
        """

        # Calculate basic statistics
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores, ddof=1)
        cv_ci_results = self.calculate_cv_conservative_ci(cv_scores)

        # Table 1: Main Results Table (Validation Performance Only)
        print("\n" + "=" * 60)
        print("Table 1: Cross-Validation Performance (Main Results)")
        print("=" * 60)

        main_table = pd.DataFrame({
            'Method': ['XGBoost-Cox'],
            'C-index (Mean ± SD)': [f'{cv_mean:.3f} ± {cv_std:.3f}'],
            '95% CI': [f'[{cv_ci_results["ci_lower"]:.3f}, {cv_ci_results["ci_upper"]:.3f}]'],
            'Folds': [len(cv_scores)],
            'Range': [f'[{min(cv_scores):.3f}, {max(cv_scores):.3f}]']
        })

        print(main_table.to_string(index=False))

        # Save main table to CSV
        main_table_path = os.path.join(self.output_dir, f'{prefix}_main_results.csv')
        main_table.to_csv(main_table_path, index=False)
        print(f"Main results table saved to: {main_table_path}")

        # Table 2: Detailed Training vs Validation Comparison (if training data available)
        if fold_training_curves:
            print("\n" + "=" * 80)
            print("Table 2: Training vs Validation Performance by Fold (Detailed Analysis)")
            print("=" * 80)

            # Extract training C-indices
            training_c_indices = []
            for fold_data in fold_training_curves:
                train_c = fold_data.get('final_training_c_index')
                training_c_indices.append(train_c if train_c is not None else np.nan)

            # Calculate differences
            differences = []
            for i, (train_c, val_c) in enumerate(zip(training_c_indices, cv_scores)):
                if not np.isnan(train_c):
                    diff = train_c - val_c
                    differences.append(diff)
                else:
                    differences.append(np.nan)

            # Create detailed comparison table
            detailed_data = {
                'Fold': list(range(1, len(cv_scores) + 1)),
                'Training C-index': [f'{c:.3f}' if not np.isnan(c) else 'N/A' for c in training_c_indices],
                'Validation C-index': [f'{c:.3f}' for c in cv_scores],
                'Difference (Train-Val)': [f'{d:+.3f}' if not np.isnan(d) else 'N/A' for d in differences]
            }

            detailed_table = pd.DataFrame(detailed_data)

            # Add summary row with 95% CI
            train_mean = np.nanmean(training_c_indices)
            train_std = np.nanstd(training_c_indices, ddof=1)
            diff_mean = np.nanmean(differences)
            diff_std = np.nanstd(differences, ddof=1)

            # Calculate 95% CI for training and validation
            train_ci = self.calculate_cv_conservative_ci(training_c_indices) if not all(np.isnan(training_c_indices)) else None
            val_ci = self.calculate_cv_conservative_ci(cv_scores)
            diff_ci = self.calculate_cv_conservative_ci(differences) if not all(np.isnan(differences)) else None

            summary_row = {
                'Fold': 'Mean ± SD',
                'Training C-index': f'{train_mean:.3f} ± {train_std:.3f}',
                'Validation C-index': f'{cv_mean:.3f} ± {cv_std:.3f}',
                'Difference (Train-Val)': f'{diff_mean:+.3f} ± {diff_std:.3f}'
            }

            # Add 95% CI row
            ci_row = {
                'Fold': '95% CI',
                'Training C-index': f'[{train_ci["ci_lower"]:.3f}, {train_ci["ci_upper"]:.3f}]' if train_ci else 'N/A',
                'Validation C-index': f'[{val_ci["ci_lower"]:.3f}, {val_ci["ci_upper"]:.3f}]',
                'Difference (Train-Val)': f'[{diff_ci["ci_lower"]:.3f}, {diff_ci["ci_upper"]:.3f}]' if diff_ci else 'N/A'
            }

            # Add both summary rows using pd.concat
            summary_df = pd.DataFrame([summary_row, ci_row])
            detailed_table = pd.concat([detailed_table, summary_df], ignore_index=True)

            print(detailed_table.to_string(index=False))

            # Print enhanced table with CI information
            print(f"\nEnhanced Table 2: Training vs Validation with 95% Confidence Intervals")
            print("=" * 90)
            print("Fold | Training C-index | Validation C-index | Difference (Train-Val)")
            print("-" * 90)
            for i, row in detailed_table.iterrows():
                if row['Fold'] not in ['Mean ± SD', '95% CI']:
                    print(f"{str(row['Fold']):4s} | {row['Training C-index']:16s} | {row['Validation C-index']:18s} | {row['Difference (Train-Val)']:19s}")
            print("-" * 90)
            print(f"Mean | {train_mean:.3f} ± {train_std:.3f}     | {cv_mean:.3f} ± {cv_std:.3f}       | {diff_mean:+.3f} ± {diff_std:.3f}")
            if train_ci:
                print(f"95CI | [{train_ci['ci_lower']:.3f}, {train_ci['ci_upper']:.3f}]    | [{val_ci['ci_lower']:.3f}, {val_ci['ci_upper']:.3f}]      | [{diff_ci['ci_lower']:+.3f}, {diff_ci['ci_upper']:+.3f}]" if diff_ci else f"95CI | [{train_ci['ci_lower']:.3f}, {train_ci['ci_upper']:.3f}]    | [{val_ci['ci_lower']:.3f}, {val_ci['ci_upper']:.3f}]      | N/A")
            else:
                print(f"95CI | N/A              | [{val_ci['ci_lower']:.3f}, {val_ci['ci_upper']:.3f}]      | N/A")

            # Save detailed table to CSV
            detailed_table_path = os.path.join(self.output_dir, f'{prefix}_detailed_comparison.csv')
            detailed_table.to_csv(detailed_table_path, index=False)
            print(f"Detailed comparison table saved to: {detailed_table_path}")

        # Table 3: Individual Fold Results (Research Detail)
        print("\n" + "=" * 60)
        print("Table 3: Individual Fold Results (Research Detail)")
        print("=" * 60)

        fold_results = pd.DataFrame({
            'Fold': list(range(1, len(cv_scores) + 1)),
            'Validation C-index': cv_scores,
            'Deviation from Mean': [score - cv_mean for score in cv_scores],
            'Rank': [i+1 for i in np.argsort(-np.array(cv_scores))]  # Best to worst
        })

        print(fold_results.to_string(index=False, float_format='%.4f'))

        # Save individual fold results
        fold_results_path = os.path.join(self.output_dir, f'{prefix}_individual_folds.csv')
        fold_results.to_csv(fold_results_path, index=False)
        print(f"Individual fold results saved to: {fold_results_path}")

        return {
            'main_table': main_table,
            'detailed_table': detailed_table if fold_training_curves else None,
            'fold_results': fold_results,
            'statistics': {
                'cv_mean': cv_mean,
                'cv_std': cv_std,
                'cv_ci_lower': cv_ci_results['ci_lower'],
                'cv_ci_upper': cv_ci_results['ci_upper'],
                'training_mean': train_mean if fold_training_curves else None,
                'overfitting_gap': diff_mean if fold_training_curves else None
            }
        }

    def plot_cv_summary_bars(self, cv_scores, prefix='cv_summary_bars'):
        """
        Create a cross-validation summary plot with bars for each fold
        """
        if not cv_scores:
            print("No cross-validation scores available.")
            return

        # Calculate statistics
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)

        # Create figure
        plt.figure(figsize=(12, 8))

        # Create bars with gradient effect
        n_folds = len(cv_scores)
        fold_numbers = range(1, n_folds + 1)

        # Create bars with color gradient from light blue to darker blue
        colors = plt.cm.Blues(np.linspace(0.4, 0.8, n_folds))
        bars = plt.bar(fold_numbers, cv_scores, color=colors,
                      alpha=0.8, edgecolor='navy', linewidth=1.5)

        # Add value labels on top of bars
        for i, (bar, score) in enumerate(zip(bars, cv_scores)):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                    f'{score:.4f}', ha='center', va='bottom',
                    fontsize=12, fontweight='bold')

        # Add mean line (red dashed)
        plt.axhline(y=cv_mean, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {cv_mean:.4f}')

        # Add confidence interval shading (±1 SD)
        plt.axhspan(cv_mean - cv_std, cv_mean + cv_std,
                   alpha=0.2, color='red',
                   label=f'±1 SD: {cv_std:.4f}')

        # Customize plot
        plt.xlabel('Fold', fontsize=14)
        plt.ylabel('Final C-index', fontsize=14)
        plt.title('Cross-Validation Results Summary', fontsize=16, fontweight='bold')
        plt.xticks(fold_numbers, fontsize=12)
        plt.yticks(fontsize=12)
        plt.grid(True, alpha=0.3)

        # Set y-axis limits with some padding
        y_min = min(cv_scores) - 0.05
        y_max = max(cv_scores) + 0.05
        plt.ylim([max(0, y_min), min(1.0, y_max)])

        # Add legend in lower left
        plt.legend(loc='lower left', fontsize=11, framealpha=0.9)

        # Add statistics text box
        stats_text = (f'Results Summary:\n'
                     f'Mean C-index: {cv_mean:.4f}\n'
                     f'Std Deviation: {cv_std:.4f}\n'
                     f'Range: [{min(cv_scores):.4f}, {max(cv_scores):.4f}]\n'
                     f'Best Fold: {np.argmax(cv_scores)+1} ({max(cv_scores):.4f})\n'
                     f'Worst Fold: {np.argmin(cv_scores)+1} ({min(cv_scores):.4f})')

        plt.text(0.98, 0.02, stats_text, transform=plt.gca().transAxes,
                verticalalignment='bottom', horizontalalignment='right',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.9),
                fontsize=10)

        plt.tight_layout()

        # Save as PDF
        summary_path = os.path.join(self.output_dir, f'{prefix}.pdf')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"CV summary plot saved to: {summary_path}")

        # Print detailed results
        print(f"\nCross-Validation Summary:")
        print(f"   Mean C-index: {cv_mean:.4f} ± {cv_std:.4f}")
        print(f"   Individual fold results:")
        for i, score in enumerate(cv_scores):
            print(f"      Fold {i+1}: {score:.4f}")

    def plot_feature_importance(self, top_n=20, prefix='feature_importance'):
        """Plot feature importance with real feature names and save as PDF"""
        if self.feature_importance is None:
            print("No feature importance available. Train model first.")
            return

        if self.feature_names is None:
            print("No feature names available.")
            return

        # Convert XGBoost feature indices to real feature names
        importance_data = []
        for feature_idx, importance in self.feature_importance.items():
            if feature_idx.startswith('f'):
                try:
                    idx = int(feature_idx[1:])
                    if idx < len(self.feature_names):
                        real_name = self.feature_names[idx]
                        importance_data.append((real_name, importance))
                    else:
                        importance_data.append((feature_idx, importance))
                except ValueError:
                    importance_data.append((feature_idx, importance))
            else:
                importance_data.append((feature_idx, importance))

        # Convert to DataFrame and ensure we get exactly top_n features
        importance_df = pd.DataFrame(
            importance_data,
            columns=['feature', 'importance']
        ).sort_values('importance', ascending=False).head(top_n)

        # Debug: Print actual number of features available
        print(f"Total features with importance: {len(importance_data)}")
        print(f"Requested top_n: {top_n}")
        print(f"Features to display: {len(importance_df)}")

        # Adjust figure size based on number of features to display
        fig_height = max(8, len(importance_df) * 0.4)
        plt.figure(figsize=(14, fig_height))

        # Create horizontal bar plot
        bars = plt.barh(range(len(importance_df)), importance_df['importance'])

        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, importance_df['importance'])):
            width = bar.get_width()
            plt.text(width + width*0.01, bar.get_y() + bar.get_height()/2,
                    f'{importance:.3f}', ha='left', va='center', fontsize=10)

        # Set y-tick labels to feature names
        plt.yticks(range(len(importance_df)), importance_df['feature'])

        plt.xlabel('Feature Importance (Gain)', fontsize=12)
        plt.title(f'Top {len(importance_df)} Important Features - XGBoost Survival Model', fontsize=14)
        plt.gca().invert_yaxis()  # Highest importance at top

        # Adjust layout to accommodate feature names
        plt.tight_layout()
        plt.subplots_adjust(left=0.35)  # Increase left margin for longer feature names

        importance_path = os.path.join(self.output_dir, f'{prefix}.pdf')
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature importance plot saved to: {importance_path}")

        print(f"\nTop {len(importance_df)} Feature Importance:")
        print("-" * 60)
        for idx, row in importance_df.iterrows():
            print(f"{row['feature']:<45} {row['importance']:.4f}")

        return importance_df

    def create_ptmb_boxplot(self, X_scaled_df):
        """
        Create boxplot for pTMB column if it exists in the normalized dataset
        
        Args:
            X_scaled_df: DataFrame with normalized features
        """
        # Check if pTMB column exists (case insensitive search)
        ptmb_columns = [col for col in X_scaled_df.columns if 'ptmb' in col.lower()]
        
        if not ptmb_columns:
            print("Warning: No pTMB column found in dataset. Skipping pTMB boxplot.")
            return
        
        # Use the first pTMB column found
        ptmb_col = ptmb_columns[0]
        ptmb_values = X_scaled_df[ptmb_col]
        
        print(f"Creating pTMB boxplot for column: {ptmb_col}")
        print(f"pTMB statistics (normalized):")
        print(f"  - Min: {ptmb_values.min():.4f}")
        print(f"  - Max: {ptmb_values.max():.4f}")
        print(f"  - Mean: {ptmb_values.mean():.4f}")
        print(f"  - Median: {ptmb_values.median():.4f}")
        print(f"  - Std: {ptmb_values.std():.4f}")
        
        # Create boxplot
        plt.figure(figsize=(8, 6))
        
        # Create boxplot with custom styling
        box_plot = plt.boxplot(ptmb_values, patch_artist=True, 
                              boxprops=dict(facecolor='lightblue', alpha=0.7),
                              medianprops=dict(color='red', linewidth=2),
                              whiskerprops=dict(color='black', linewidth=1.5),
                              capprops=dict(color='black', linewidth=1.5),
                              flierprops=dict(marker='o', markerfacecolor='red', 
                                            markersize=5, alpha=0.5))
        
        # Add statistics text
        stats_text = (f'Statistics (Normalized):\n'
                     f'Min: {ptmb_values.min():.4f}\n'
                     f'Q1: {ptmb_values.quantile(0.25):.4f}\n'
                     f'Median: {ptmb_values.median():.4f}\n'
                     f'Q3: {ptmb_values.quantile(0.75):.4f}\n'
                     f'Max: {ptmb_values.max():.4f}\n'
                     f'Mean: {ptmb_values.mean():.4f}\n'
                     f'Std: {ptmb_values.std():.4f}\n'
                     f'N: {len(ptmb_values)}')
        
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', 
                facecolor='lightgray', alpha=0.9), fontsize=10)
        
        # Customize plot
        plt.ylabel('Normalized pTMB Values', fontsize=12)
        plt.title(f'Distribution of {ptmb_col} (MinMax Normalized)', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Remove x-axis ticks since we only have one boxplot
        plt.xticks([1], [ptmb_col])
        
        plt.tight_layout()
        
        # Save boxplot
        boxplot_path = os.path.join(self.output_dir, 'pTMB_distribution_boxplot.pdf')
        plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"pTMB boxplot saved to: {boxplot_path}")
        
        # Also create histogram for additional visualization
        plt.figure(figsize=(10, 6))
        
        # Create histogram
        n_bins = min(50, max(10, len(ptmb_values) // 10))  # Adaptive number of bins
        plt.hist(ptmb_values, bins=n_bins, alpha=0.7, color='lightblue', 
                edgecolor='black', density=True)
        
        # Add statistics lines
        plt.axvline(ptmb_values.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {ptmb_values.mean():.4f}')
        plt.axvline(ptmb_values.median(), color='orange', linestyle='--', 
                   linewidth=2, label=f'Median: {ptmb_values.median():.4f}')
        
        plt.xlabel('Normalized pTMB Values', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title(f'Histogram of {ptmb_col} (MinMax Normalized)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save histogram
        histogram_path = os.path.join(self.output_dir, 'pTMB_distribution_histogram.pdf')
        plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"pTMB histogram saved to: {histogram_path}")

    def plot_survival_predictions(self, X_test, y_time_test, y_event_test, prefix='survival_predictions'):
        """Plot survival predictions as individual PDF files"""
        risk_scores = self.predict(X_test)

        # Plot 1: Risk score distribution by event status
        plt.figure(figsize=(10, 6))
        event_scores = risk_scores[y_event_test == 1]
        no_event_scores = risk_scores[y_event_test == 0]

        plt.hist(event_scores, bins=20, alpha=0.6, label=f'Event (n={len(event_scores)})',
                color='red', density=True)
        plt.hist(no_event_scores, bins=20, alpha=0.6, label=f'No Event (n={len(no_event_scores)})',
                color='blue', density=True)
        plt.xlabel('Risk Score')
        plt.ylabel('Density')
        plt.title('Risk Score Distribution by Event Status')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        dist_path = os.path.join(self.output_dir, f'{prefix}_risk_distribution.pdf')
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Risk score distribution saved to: {dist_path}")

        # Plot 2: Risk score vs survival time
        plt.figure(figsize=(10, 6))
        scatter_colors = ['red' if e == 1 else 'blue' for e in y_event_test]
        plt.scatter(y_time_test, risk_scores, c=scatter_colors, alpha=0.6,
                   edgecolors='black', linewidth=0.5)
        plt.xlabel('Survival Time')
        plt.ylabel('Risk Score')
        plt.title('Risk Score vs Survival Time')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        scatter_path = os.path.join(self.output_dir, f'{prefix}_risk_vs_time.pdf')
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Risk vs survival time plot saved to: {scatter_path}")

        # Plot 3: Kaplan-Meier curves for risk groups
        plt.figure(figsize=(10, 6))
        median_risk = np.median(risk_scores)
        high_risk = risk_scores >= median_risk
        low_risk = risk_scores < median_risk

        kmf = KaplanMeierFitter()

        # High risk group
        kmf.fit(y_time_test[high_risk], y_event_test[high_risk], label='High Risk')
        kmf.plot_survival_function(color='red')

        # Low risk group
        kmf.fit(y_time_test[low_risk], y_event_test[low_risk], label='Low Risk')
        kmf.plot_survival_function(color='blue')

        plt.xlabel('Time')
        plt.ylabel('Survival Probability')
        plt.title('Kaplan-Meier Curves by Risk Group')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        km_path = os.path.join(self.output_dir, f'{prefix}_kaplan_meier.pdf')
        plt.savefig(km_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Kaplan-Meier curves saved to: {km_path}")

    def create_combined_pdf_report(self, mode='cv_mode'):
        """Create a combined PDF report with all plots"""
        if mode == 'train_val_test':
            pdf_files = [
                f'{mode}_loss_curves.pdf',
                f'{mode}_cindex_curves.pdf',
                f'feature_importance_{mode}.pdf',
                f'survival_predictions_{mode}_risk_distribution.pdf',
                f'survival_predictions_{mode}_risk_vs_time.pdf',
                f'survival_predictions_{mode}_kaplan_meier.pdf'
            ]
        else:  # cv_mode
            pdf_files = [
                f'{mode}_training_loss_curves.pdf',
                f'{mode}_validation_loss_curves.pdf',
                f'{mode}_training_cindex_curves.pdf',
                f'{mode}_validation_cindex_curves.pdf',
                f'{mode}_final_comparison_by_fold.pdf',
                f'{mode}_cv_summary.pdf',
                f'cv_summary_bars.pdf',
                f'cv_summary_bars_with_ci.pdf',
                f'cv_table_image_filled.pdf',  # Table with CI values filled
                f'cv_table_image_blank.pdf',   # Table with blank CI for manual filling
                f'cv_table_image_validation_only.pdf',  # Simple validation-only table
                f'detailed_comparison_table.pdf',  # New detailed table with CI
                f'feature_importance_{mode}.pdf'
            ]

        # Try to combine PDFs if pypdf is available
        try:
            from pypdf import PdfWriter, PdfReader

            combined_path = os.path.join(self.output_dir, f'{mode}_combined_report.pdf')
            writer = PdfWriter()

            for pdf_file in pdf_files:
                pdf_path = os.path.join(self.output_dir, pdf_file)
                if os.path.exists(pdf_path):
                    with open(pdf_path, 'rb') as f:
                        reader = PdfReader(f)
                        for page in reader.pages:
                            writer.add_page(page)

            with open(combined_path, 'wb') as f:
                writer.write(f)

            print(f"Combined PDF report created: {combined_path}")

        except ImportError:
            print("pypdf not available. Individual PDF files created separately.")
            print("To create combined PDF, install pypdf: pip install pypdf")
        except Exception as e:
            print(f"Error creating combined PDF: {e}")
            print("Individual PDF files are still available.")

    def perform_shap_analysis_best_fold(self, X_all, y_time_all, y_event_all, cv_results, n_folds=5, prefix='shap_analysis'):
        """
        Perform SHAP analysis using the best performing fold and analyze only its validation set

        Args:
            X_all: Feature matrix (all data)
            y_time_all: Survival times (all data)
            y_event_all: Event indicators (all data)
            cv_results: Cross-validation results to determine best fold
            n_folds: Number of folds for cross-validation
            prefix: Prefix for output files

        Returns:
            Dictionary containing SHAP analysis results
        """
        print("\n" + "=" * 80)
        print("SHAP Analysis using Best Performing Fold (Validation Set Only)")
        print("=" * 80)

        # Find the best performing fold
        best_fold_idx = int(np.argmax(cv_results['cv_scores']))
        best_c_index = cv_results['cv_scores'][best_fold_idx]
        
        print(f"Best performing fold: Fold {best_fold_idx + 1} (C-index: {best_c_index:.4f})")
        print(f"Using ONLY the validation set from this fold for SHAP analysis...")

        # Recreate the same cross-validation splits to get the best fold's data
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        event_stratify = y_event_all  # Stratify by event occurrence
        
        fold_splits = list(skf.split(X_all, event_stratify))
        train_idx, val_idx = fold_splits[best_fold_idx]

        print(f"Best fold data split:")
        print(f"  - Training samples: {len(train_idx)}")
        print(f"  - Validation samples: {len(val_idx)} (SHAP analysis target)")

        # Split data for the best fold
        if hasattr(X_all, 'iloc'):  # pandas DataFrame
            X_train_best = X_all.iloc[train_idx]
            X_val_best = X_all.iloc[val_idx]
            y_time_train_best = y_time_all.iloc[train_idx]
            y_event_train_best = y_event_all.iloc[train_idx]
            y_time_val_best = y_time_all.iloc[val_idx]
            y_event_val_best = y_event_all.iloc[val_idx]
        else:  # numpy array
            X_train_best = X_all[train_idx]
            X_val_best = X_all[val_idx]
            y_time_train_best = y_time_all[train_idx] if hasattr(y_time_all, '__getitem__') else y_time_all.iloc[train_idx]
            y_event_train_best = y_event_all[train_idx] if hasattr(y_event_all, '__getitem__') else y_event_all.iloc[train_idx]
            y_time_val_best = y_time_all[val_idx] if hasattr(y_time_all, '__getitem__') else y_time_all.iloc[val_idx]
            y_event_val_best = y_event_all[val_idx] if hasattr(y_event_all, '__getitem__') else y_event_all.iloc[val_idx]

        # Calculate validation set event statistics
        val_events = y_event_val_best.sum() if hasattr(y_event_val_best, 'sum') else np.sum(y_event_val_best)
        val_event_rate = val_events / len(val_idx)
        print(f"  - Validation events: {val_events:.0f} ({val_event_rate*100:.1f}%)")

        # Prepare survival labels for training
        if hasattr(y_time_train_best, 'values'):  # pandas Series
            time_values = y_time_train_best.values
            event_values = y_event_train_best.values
        else:  # numpy array
            time_values = y_time_train_best
            event_values = y_event_train_best

        y_train_survival_best = self.prepare_survival_data(time_values, event_values)

        # Train model for the best fold
        print(f"\nTraining XGBoost model for best fold (Fold {best_fold_idx + 1})...")
        dtrain_best = xgb.DMatrix(X_train_best, label=y_train_survival_best)
        
        model_best = xgb.train(
            self.params,
            dtrain_best,
            num_boost_round=self.params.get('n_estimators', 300),
            verbose_eval=False
        )

        # Suppress SHAP warnings
        warnings.filterwarnings('ignore', category=UserWarning)

        # Calculate SHAP values for validation set ONLY
        print(f"\nComputing SHAP values for validation set of best fold...")
        try:
            # Create SHAP explainer using ALL training data as background for more accurate baseline
            print(f"Using all {len(X_train_best)} training samples as SHAP background (may take longer but more accurate)")
            background_data = X_train_best

            explainer = shap.TreeExplainer(model_best, background_data)
            shap_values = explainer.shap_values(X_val_best)

            # Convert validation data to DataFrame if it's numpy array
            if not hasattr(X_val_best, 'columns'):  # numpy array
                val_data_df = pd.DataFrame(X_val_best, columns=self.feature_names)
            else:  # already DataFrame
                val_data_df = X_val_best.copy()

            print(f"   SHAP analysis completed successfully!")
            print(f"   Analyzed {len(X_val_best)} validation samples from best fold")

        except Exception as e:
            print(f"   Error: SHAP analysis failed for best fold: {e}")
            return None

        # Calculate feature importance
        feature_importance = np.abs(shap_values).mean(axis=0)

        # Get feature names appropriately
        if hasattr(X_all, 'columns'):  # pandas DataFrame
            feature_names = X_all.columns.tolist()
        else:  # numpy array, use stored feature names
            feature_names = self.feature_names

        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP_Importance': feature_importance
        }).sort_values('SHAP_Importance', ascending=False)

        # Save feature importance results
        importance_path = os.path.join(self.output_dir, f'{prefix}_feature_importance_best_fold.csv')
        importance_df.to_csv(importance_path, index=False)
        print(f"SHAP feature importance saved to: {importance_path}")

        # Create visualizations
        print("\nCreating SHAP visualizations...")
        self._create_shap_visualizations(
            shap_values,
            val_data_df,
            importance_df,
            prefix=f"{prefix}_best_fold"
        )

        # Prepare results dictionary
        shap_results = {
            'shap_values': shap_values,
            'validation_data': val_data_df,
            'validation_indices': val_idx,
            'feature_importance': importance_df,
            'best_fold_model': model_best,
            'best_fold_idx': best_fold_idx + 1,
            'best_fold_c_index': best_c_index,
            'n_validation_samples': len(X_val_best),
            'validation_events': val_events,
            'validation_event_rate': val_event_rate
        }

        print(f"\nSHAP analysis completed successfully!")
        print(f"   Best fold: {best_fold_idx + 1} (C-index: {best_c_index:.4f})")
        print(f"   Validation samples analyzed: {len(X_val_best)}")
        print(f"   Validation events: {val_events:.0f} ({val_event_rate*100:.1f}%)")
        print(f"   Top 5 most important features:")
        for i, row in importance_df.head().iterrows():
            print(f"      {i+1}. {row['Feature']}: {row['SHAP_Importance']:.4f}")

        return shap_results

    def _create_shap_visualizations(self, shap_values, test_data, importance_df, prefix='shap_analysis'):
        """
        Create comprehensive SHAP visualizations

        Args:
            shap_values: Combined SHAP values from all folds
            test_data: Combined test data from all folds
            importance_df: Feature importance DataFrame
            prefix: Prefix for output files
        """

        # Set style for better plots
        plt.style.use('default')

        # Get used features info for filtering
        used_features_info = self.get_used_features_info()

        # Filter SHAP values and data to only used features
        if used_features_info and len(used_features_info['indices']) > 0:
            used_indices = used_features_info['indices']
            shap_values_filtered = shap_values[:, used_indices]
            test_data_filtered = test_data.iloc[:, used_indices] if hasattr(test_data, 'iloc') else test_data[:, used_indices]
            used_feature_names = used_features_info['names']

            # Create DataFrame with used feature names for SHAP plots
            if hasattr(test_data, 'iloc'):  # pandas DataFrame
                test_data_filtered = pd.DataFrame(test_data_filtered, columns=used_feature_names)
            else:
                test_data_filtered = test_data_filtered  # numpy array, feature names handled separately

            print(f"   Using {len(used_indices)} features actually used by the model")
        else:
            # Fallback to all features
            shap_values_filtered = shap_values
            test_data_filtered = test_data
            print("   Using all features (used features info not available)")

        # 1. SHAP Summary Plot (Beeswarm) - Used features only
        print("   Creating SHAP summary plot...")
        plt.figure(figsize=(10, 6))
        if used_features_info:
            shap.summary_plot(shap_values_filtered, test_data_filtered,
                            feature_names=used_feature_names, show=False,
                            max_display=len(used_feature_names))
        else:
            shap.summary_plot(shap_values_filtered, test_data_filtered, show=False, max_display=20)
        plt.tight_layout()
        summary_path = os.path.join(self.output_dir, f'{prefix}_summary_plot.pdf')
        plt.savefig(summary_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      Summary plot saved to: {summary_path}")

        # 2. SHAP Bar Plot (Mean absolute values) - Used features only
        print("   Creating SHAP bar plot...")
        plt.figure(figsize=(8, 6))
        if used_features_info:
            shap.summary_plot(shap_values_filtered, test_data_filtered,
                            feature_names=used_feature_names, plot_type="bar",
                            show=False, max_display=len(used_feature_names))
        else:
            shap.summary_plot(shap_values_filtered, test_data_filtered, plot_type="bar", show=False, max_display=20)
        plt.tight_layout()
        bar_path = os.path.join(self.output_dir, f'{prefix}_bar_plot.pdf')
        plt.savefig(bar_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      Bar plot saved to: {bar_path}")

        # 3. Simple Feature Importance (No Error Bars) - Used features only
        print("   Creating simplified feature importance plot...")

        # Filter importance_df to only used features
        if used_features_info:
            used_features_df = importance_df[importance_df['Feature'].isin(used_feature_names)].copy()
        else:
            used_features_df = importance_df.head(20)  # Top 20 features if no used features info

        plt.figure(figsize=(8, max(4, len(used_features_df) * 0.3)))

        y_pos = np.arange(len(used_features_df))
        
        # Handle different column names for SHAP importance
        importance_col = 'SHAP_Importance_Mean' if 'SHAP_Importance_Mean' in used_features_df.columns else 'SHAP_Importance'
        plt.barh(y_pos, used_features_df[importance_col],
                alpha=0.8, color='steelblue')

        plt.yticks(y_pos, used_features_df['Feature'])
        plt.xlabel('Mean |SHAP value|')
        plt.gca().invert_yaxis()
        plt.tight_layout()

        importance_plot_path = os.path.join(self.output_dir, f'{prefix}_importance_with_error_bars.pdf')
        plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      Simplified importance plot saved to: {importance_plot_path}")

        # 4. Top Features Waterfall Plot (for first sample)
        if len(test_data) > 0:
            print("   Creating SHAP waterfall plot for sample case...")
            plt.figure(figsize=(10, 8))
            # Use the first sample as an example
            sample_idx = 0
            shap.waterfall_plot(
                shap.Explanation(
                    values=shap_values[sample_idx],
                    base_values=shap_values.mean(axis=0).mean(),  # Average base value
                    data=test_data.iloc[sample_idx].values,
                    feature_names=test_data.columns.tolist()
                ),
                show=False,
                max_display=15
            )
            plt.title(f'SHAP Waterfall Plot (Sample Patient)', fontsize=14, fontweight='bold')
            plt.tight_layout()

            waterfall_path = os.path.join(self.output_dir, f'{prefix}_waterfall_example.pdf')
            plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"      Waterfall plot saved to: {waterfall_path}")

        # 5. SHAP Dependence Plots for used features (simplified)
        if used_features_info and len(used_features_info['names']) > 0:
            print("   Creating SHAP dependence plots for used features...")

            # Get top used features (up to 6)
            used_features_df_sorted = used_features_df.head(6)
            top_used_features = used_features_df_sorted['Feature'].tolist()
            n_features = len(top_used_features)

            if n_features > 0:
                # Dynamic subplot layout based on number of features
                if n_features <= 2:
                    fig, axes = plt.subplots(1, n_features, figsize=(6*n_features, 5))
                elif n_features <= 4:
                    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                else:
                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

                if n_features == 1:
                    axes = [axes]
                else:
                    axes = axes.ravel()

                for i, feature in enumerate(top_used_features):
                    try:
                        plt.sca(axes[i])
                        # Use filtered data for dependence plots
                        feature_idx = used_feature_names.index(feature)
                        shap.dependence_plot(
                            feature_idx, shap_values_filtered, test_data_filtered,
                            feature_names=used_feature_names,
                            show=False, ax=axes[i]
                        )
                        axes[i].set_title(f'{feature}', fontsize=10)
                    except Exception as e:
                        print(f"      Warning: Could not create dependence plot for {feature}: {e}")
                        axes[i].text(0.5, 0.5, f'Error\n{feature}',
                                   ha='center', va='center', transform=axes[i].transAxes)

                # Hide unused subplots
                for i in range(n_features, len(axes)):
                    axes[i].set_visible(False)

                plt.tight_layout()
                dependence_path = os.path.join(self.output_dir, f'{prefix}_dependence_plots.pdf')
                plt.savefig(dependence_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"      Dependence plots saved to: {dependence_path}")
            else:
                print("      No used features available for dependence plots")
        else:
            print("      Skipping dependence plots (no used features info available)")

        print("   All SHAP visualizations completed successfully!")

    def get_used_features_info(self):
        """Get information about features actually used by the model"""
        if self.model is None or self.feature_importance is None:
            return None

        # Get indices of used features
        used_feature_indices = []
        used_feature_names = []

        for feature_idx in self.feature_importance.keys():
            if feature_idx.startswith('f'):
                try:
                    idx = int(feature_idx[1:])
                    if idx < len(self.feature_names):
                        used_feature_indices.append(idx)
                        used_feature_names.append(self.feature_names[idx])
                except ValueError:
                    pass

        # Sort by index to maintain order
        sorted_pairs = sorted(zip(used_feature_indices, used_feature_names))
        used_feature_indices, used_feature_names = zip(*sorted_pairs) if sorted_pairs else ([], [])

        return {
            'indices': list(used_feature_indices),
            'names': list(used_feature_names),
            'count': len(used_feature_indices)
        }

    def setup_shap_explainer(self, background_data, max_background_samples=None):
        """
        Setup SHAP explainer for model interpretability

        Args:
            background_data: Training data to use as background for SHAP
            max_background_samples: Maximum number of background samples to use (None = use all samples)
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Train model first before setting up SHAP explainer.")

        print("Setting up SHAP explainer...")

        # Get used features info
        used_features_info = self.get_used_features_info()
        if used_features_info:
            print(f"Model uses {used_features_info['count']} out of {len(self.feature_names)} total features")
            print("Used features:", ", ".join(used_features_info['names'][:5]) +
                  ("..." if used_features_info['count'] > 5 else ""))

        # Handle different data types
        if max_background_samples is None:
            # Use all background data
            print(f"Using all {len(background_data)} samples as SHAP background")
            self.background_data = background_data
        else:
            # Limit background samples
            if hasattr(background_data, 'sample'):  # pandas DataFrame
                if len(background_data) > max_background_samples:
                    self.background_data = background_data.sample(n=max_background_samples, random_state=42)
                    print(f"Sampled {max_background_samples} from {len(background_data)} background samples")
                else:
                    self.background_data = background_data
                    print(f"Using all {len(background_data)} background samples")
            else:  # numpy array
                if len(background_data) > max_background_samples:
                    indices = np.random.choice(len(background_data), max_background_samples, replace=False)
                    self.background_data = background_data[indices]
                    print(f"Sampled {max_background_samples} from {len(background_data)} background samples")
                else:
                    self.background_data = background_data
                    print(f"Using all {len(background_data)} background samples")

        # Create SHAP explainer
        # For XGBoost models, we use TreeExplainer
        self.shap_explainer = shap.TreeExplainer(self.model)

        print(f"SHAP explainer setup completed with {len(self.background_data)} background samples")

    def calculate_shap_values(self, X_data):
        """
        Calculate SHAP values for given data

        Args:
            X_data: Data to calculate SHAP values for

        Returns:
            SHAP values array
        """
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not setup. Call setup_shap_explainer() first.")

        # Calculate SHAP values
        shap_values = self.shap_explainer.shap_values(X_data)
        return shap_values

    def plot_shap_waterfall(self, X_sample, sample_idx=0, max_display=15,
                           save_path='shap_waterfall.png', used_features_only=True):
        """
        Plot SHAP waterfall plot for a single sample

        Args:
            X_sample: Single sample data (1D array or 2D array with one row)
            sample_idx: Index of the sample if X_sample contains multiple samples
            max_display: Maximum number of features to display
            save_path: Path to save the plot
            used_features_only: If True, only show features actually used by the model
        """
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not setup. Call setup_shap_explainer() first.")

        # Handle different data types and ensure 2D
        if hasattr(X_sample, 'values'):  # pandas DataFrame
            X_sample_array = X_sample.values
        else:  # numpy array
            X_sample_array = X_sample

        if X_sample_array.ndim == 1:
            X_sample_array = X_sample_array.reshape(1, -1)

        # Calculate SHAP values for the sample
        shap_values = self.calculate_shap_values(X_sample_array)

        # Get the sample for plotting
        if shap_values.ndim > 1:
            sample_shap_values = shap_values[sample_idx]
            sample_data = X_sample_array[sample_idx]
        else:
            sample_shap_values = shap_values
            sample_data = X_sample_array[0]

        # Get used features info
        used_features_info = self.get_used_features_info()

        if used_features_only and used_features_info:
            # Filter to only used features
            used_indices = used_features_info['indices']
            used_names = used_features_info['names']

            # Extract values for used features only
            sample_shap_values_filtered = sample_shap_values[used_indices]
            sample_data_filtered = sample_data[used_indices]
            feature_names_filtered = used_names

            print(f"Showing waterfall plot for {len(used_indices)} features actually used by the model")

            # Adjust max_display to not exceed number of used features
            max_display = min(max_display, len(used_indices))

        else:
            # Use all features
            sample_shap_values_filtered = sample_shap_values
            sample_data_filtered = sample_data
            if self.feature_names is not None:
                feature_names_filtered = self.feature_names
            else:
                feature_names_filtered = [f'Feature_{i}' for i in range(len(sample_data))]

        # Create SHAP Explanation object with proper base_values handling
        try:
            # First try to get base_values from the explainer
            if hasattr(self.shap_explainer, 'expected_value'):
                base_value = self.shap_explainer.expected_value
            else:
                # Fallback: calculate expected value from background data
                if hasattr(self, 'background_data') and self.background_data is not None:
                    background_predictions = self.predict(self.background_data)
                    base_value = np.mean(background_predictions)
                else:
                    base_value = 0.0

            # Ensure base_value is a scalar
            if isinstance(base_value, (list, np.ndarray)):
                base_value = float(base_value[0]) if len(base_value) > 0 else 0.0
            elif base_value is None:
                base_value = 0.0
            else:
                base_value = float(base_value)

            explanation = shap.Explanation(
                values=sample_shap_values_filtered,
                base_values=base_value,
                data=sample_data_filtered,
                feature_names=feature_names_filtered
            )

        except Exception as e:
            print(f"Warning: Could not create proper SHAP explanation with base_values: {e}")
            print("Falling back to basic explanation...")

            # Fallback: create explanation without base_values
            explanation = shap.Explanation(
                values=sample_shap_values_filtered,
                data=sample_data_filtered,
                feature_names=feature_names_filtered
            )
            # Manually set base_values to 0
            explanation.base_values = 0.0

        # Create waterfall plot with error handling
        plt.figure(figsize=(10, max(8, len(sample_shap_values_filtered) * 0.3)))

        try:
            shap.waterfall_plot(explanation, max_display=max_display, show=False)

            title_suffix = " (Used Features Only)" if used_features_only and used_features_info else ""
            plt.title(f'SHAP Waterfall Plot - Sample {sample_idx}{title_suffix}', fontsize=14, pad=20)
            plt.tight_layout()

            # Ensure output directory exists
            waterfall_path = os.path.join(self.output_dir, save_path)
            plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"SHAP waterfall plot saved to: {waterfall_path}")

        except Exception as e:
            print(f"Error creating waterfall plot: {e}")
            print("Creating alternative bar plot instead...")

            # Alternative: Create a simple bar plot of SHAP values
            plt.close()
            plt.figure(figsize=(12, max(6, len(feature_names_filtered) * 0.4)))

            # Sort by absolute SHAP value
            sorted_indices = np.argsort(np.abs(sample_shap_values_filtered))[::-1]
            sorted_values = sample_shap_values_filtered[sorted_indices]
            sorted_names = [feature_names_filtered[i] for i in sorted_indices]

            # Take top max_display features
            display_count = min(max_display, len(sorted_values))
            plot_values = sorted_values[:display_count]
            plot_names = sorted_names[:display_count]

            # Create horizontal bar plot
            colors = ['red' if v < 0 else 'blue' for v in plot_values]
            y_pos = np.arange(len(plot_names))

            plt.barh(y_pos, plot_values, color=colors, alpha=0.7)
            plt.yticks(y_pos, plot_names)
            plt.xlabel('SHAP Value (Impact on Prediction)')

            title_suffix = " (Used Features Only)" if used_features_only and used_features_info else ""
            plt.title(f'SHAP Feature Impact - Sample {sample_idx}{title_suffix}', fontsize=14)
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            plt.gca().invert_yaxis()

            # Add value labels
            for i, v in enumerate(plot_values):
                plt.text(v + (0.01 * max(abs(plot_values)) if v >= 0 else -0.01 * max(abs(plot_values))),
                        i, f'{v:.3f}', va='center', fontsize=9)

            plt.tight_layout()

            waterfall_path = os.path.join(self.output_dir, save_path)
            plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Alternative SHAP plot saved to: {waterfall_path}")

        # Print feature contributions
        print(f"\nFeature contributions for Sample {sample_idx}:")
        print("-" * 60)

        # Sort features by absolute SHAP value
        feature_importance = list(zip(feature_names_filtered, sample_shap_values_filtered, sample_data_filtered))
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)

        display_count = min(max_display, len(feature_importance))
        for i, (fname, shap_val, fval) in enumerate(feature_importance[:display_count]):
            direction = "↑" if shap_val > 0 else "↓"
            print(f"{i+1:2d}. {fname:<30} {direction} SHAP: {shap_val:8.4f} | Value: {fval:8.4f}")

        if used_features_only and used_features_info:
            print(f"\nNote: Showing {len(feature_names_filtered)} features actually used by the model")

    def plot_shap_directional_importance(self, X_data, save_path='shap_directional_importance.png',
                                       used_features_only=True):
        """
        Plot SHAP directional feature importance showing positive/negative impact

        Args:
            X_data: Data to calculate SHAP values for
            save_path: Path to save the plot
            used_features_only: If True, only show features actually used by the model
        """
        if self.shap_explainer is None:
            raise ValueError("SHAP explainer not setup. Call setup_shap_explainer() first.")

        print("Calculating SHAP values for directional importance...")
        shap_values = self.calculate_shap_values(X_data)

        # Get used features info
        used_features_info = self.get_used_features_info()

        if used_features_only and used_features_info:
            # Filter to only used features
            used_indices = used_features_info['indices']
            used_names = used_features_info['names']

            # Extract SHAP values for used features only
            shap_values_filtered = shap_values[:, used_indices]
            feature_names_filtered = used_names

            print(f"Calculating directional SHAP importance for {len(used_indices)} features actually used by the model")

        else:
            # Use all features
            shap_values_filtered = shap_values
            if self.feature_names is not None:
                feature_names_filtered = self.feature_names
            else:
                feature_names_filtered = [f'Feature_{i}' for i in range(shap_values.shape[1])]

        # Calculate mean SHAP values (NOT absolute - this shows direction)
        mean_shap = np.mean(shap_values_filtered, axis=0)

        # Sort features by mean SHAP value (most positive to most negative)
        feature_importance = list(zip(feature_names_filtered, mean_shap))
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        # For used features only, show all of them; otherwise limit to top 20
        max_features = len(feature_importance) if used_features_only and used_features_info else 20
        top_features = feature_importance[:max_features]

        # Create the plot
        fig_height = max(8, len(top_features) * 0.4)
        plt.figure(figsize=(12, fig_height))
        features, importances = zip(*top_features)

        # Color bars based on positive/negative impact
        colors = ['darkred' if imp < 0 else 'darkblue' for imp in importances]

        y_pos = np.arange(len(features))
        bars = plt.barh(y_pos, importances, color=colors, alpha=0.7)
        plt.yticks(y_pos, features)
        plt.xlabel('Mean SHAP Value (Directional Impact)', fontsize=12)

        title_suffix = " (Used Features Only)" if used_features_only and used_features_info else ""
        plt.title(f'SHAP Directional Feature Importance{title_suffix}', fontsize=14)
        plt.gca().invert_yaxis()

        # Add a vertical line at x=0 to separate positive/negative impacts
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)

        # Add value labels on bars
        for i, (bar, importance) in enumerate(zip(bars, importances)):
            width = bar.get_width()
            # Position text based on bar direction
            if width >= 0:
                plt.text(width + abs(width)*0.02, bar.get_y() + bar.get_height()/2,
                        f'{importance:.4f}', ha='left', va='center', fontsize=9)
            else:
                plt.text(width - abs(width)*0.02, bar.get_y() + bar.get_height()/2,
                        f'{importance:.4f}', ha='right', va='center', fontsize=9)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='darkblue', alpha=0.7, label='Increases Risk (Positive Impact)'),
            Patch(facecolor='darkred', alpha=0.7, label='Decreases Risk (Negative Impact)')
        ]
        plt.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()

        directional_path = os.path.join(self.output_dir, save_path)
        plt.savefig(directional_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"SHAP directional importance plot saved to: {directional_path}")

        # Print directional importance
        print(f"\nDirectional SHAP Feature Importance (Mean Impact):")
        print("-" * 70)
        print("Positive values = Increase Risk | Negative values = Decrease Risk")
        print("-" * 70)

        for i, (fname, importance) in enumerate(top_features):
            direction = "↑ Risk" if importance > 0 else "↓ Risk"
            impact_type = "Increases" if importance > 0 else "Decreases"
            print(f"{i+1:2d}. {fname:<35} {direction:7s} | {importance:8.6f} ({impact_type} risk)")

        # Summary statistics
        positive_features = [imp for _, imp in top_features if imp > 0]
        negative_features = [imp for _, imp in top_features if imp < 0]

        print(f"\nSummary:")
        print(f"  - Features that increase risk: {len(positive_features)}")
        print(f"  - Features that decrease risk: {len(negative_features)}")
        if positive_features:
            print(f"  - Strongest risk increaser: {max(positive_features):.6f}")
        if negative_features:
            print(f"  - Strongest risk decreaser: {min(negative_features):.6f}")

        return feature_importance


def main(mode='train_val_test'):
    """
    Main function to run XGBoost survival analysis

    Args:
        mode: 'train_val_test' or 'cv_only'
            - 'train_val_test': Traditional train/validation/test split
            - 'cv_only': 5-fold cross-validation using entire dataset
    """

    print("=" * 80)
    if mode == 'train_val_test':
        print("XGBoost Survival Analysis - Train/Validation/Test Mode")
        print("Data split: 72.25% train / 12.75% validation / 15% test")
    else:
        print("XGBoost Survival Analysis - Cross-Validation Mode")
        print("5-Fold CV using 100% of data (Enhanced with validation curves)")
    print("=" * 80)

    # Load data
    file_path = DATA_FILE_PATH

    try:
        df = pd.read_csv(file_path)
        print(f"Data loaded successfully: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None, None
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

    # Initialize XGBoost survival model
    xgb_model = SurvivalXGBoost()

    if mode == 'train_val_test':
        return run_train_val_test_mode(xgb_model, df)
    else:
        return run_cv_mode(xgb_model, df)


def run_train_val_test_mode(xgb_model, df):
    """
    Run traditional train/validation/test mode with event statistics
    """

    # Prepare data
    print("\nPreparing data for train/validation/test...")
    data_dict = xgb_model.prepare_data(df)

    # Split validation set from training set
    X_train = data_dict['X_train']
    y_train_survival = data_dict['y_train_survival']
    y_time_train = data_dict['y_time_train']
    y_event_train = data_dict['y_event_train']

    # Create validation split (15% of training data)
    val_size = int(0.15 * len(X_train))
    X_train_sub = X_train[:-val_size]
    X_val = X_train[-val_size:]
    y_train_survival_sub = y_train_survival[:-val_size]
    y_val_survival = y_train_survival[-val_size:]
    y_time_train_sub = y_time_train[:-val_size]
    y_time_val = y_time_train[-val_size:]
    y_event_train_sub = y_event_train[:-val_size]
    y_event_val = y_event_train[-val_size:]

    # Calculate event statistics
    train_events = y_event_train_sub.sum()
    val_events = y_event_val.sum()
    test_events = data_dict['y_event_test'].sum()

    train_pairs = int(train_events * (len(X_train_sub) - train_events))
    val_pairs = int(val_events * (len(X_val) - val_events))
    test_pairs = int(test_events * (len(data_dict['X_test']) - test_events))

    print(f"\nData splits with event statistics:")
    print(f"   - Training: {len(X_train_sub)} samples, {train_events:.0f} events ({train_events/len(X_train_sub)*100:.1f}%)")
    print(f"              Concordant pairs: {train_pairs:,}")
    print(f"   - Validation: {len(X_val)} samples, {val_events:.0f} events ({val_events/len(X_val)*100:.1f}%)")
    print(f"                Concordant pairs: {val_pairs:,}")
    print(f"   - Test: {len(data_dict['X_test'])} samples, {test_events:.0f} events ({test_events/len(data_dict['X_test'])*100:.1f}%)")
    print(f"          Concordant pairs: {test_pairs:,}")

    # Warning checks
    if val_events < 30:
        print(f"\nWARNING: Validation set has only {val_events:.0f} events!")
        print("   Consider using 3-fold CV or adjusting the split ratio for more stable results.")

    if val_pairs < train_pairs * 0.1:
        print(f"\nWARNING: Validation concordant pairs ({val_pairs:,}) are less than 10% of training pairs!")
        print("   This may lead to unstable C-index estimates.")

    # Train model with enhanced validation tracking
    print("\nTraining XGBoost survival model with validation tracking...")
    print("-" * 40)

    xgb_model.train_with_validation(
        X_train_sub, y_train_survival_sub, y_time_train_sub, y_event_train_sub,
        X_val, y_val_survival, y_time_val, y_event_val,
        early_stopping_rounds=50,
        verbose=True,
        cindex_frequency=10
    )

    # Evaluate on all sets
    print("\n" + "=" * 40)
    print("Model Evaluation")
    print("=" * 40)

    # Training performance
    train_c_index = xgb_model.evaluate(X_train_sub, y_time_train_sub, y_event_train_sub)
    print(f"Training C-index: {train_c_index:.4f}")

    # Validation performance
    val_c_index = xgb_model.evaluate(X_val, y_time_val, y_event_val)
    print(f"Validation C-index: {val_c_index:.4f}")

    # Test performance with confidence intervals
    test_c_index = xgb_model.evaluate(
        data_dict['X_test'],
        data_dict['y_time_test'],
        data_dict['y_event_test']
    )
    print(f"Test C-index: {test_c_index:.4f}")

    # Calculate confidence intervals for test performance
    print("\nCalculating 95% confidence intervals for test performance...")
    try:
        # DeLong CI
        risk_scores_test = xgb_model.predict(data_dict['X_test'])
        delong_ci = xgb_model.calculate_delong_ci(
            data_dict['y_time_test'],
            data_dict['y_event_test'],
            risk_scores_test
        )
        if delong_ci['ci_lower'] is not None:
            print(f"Test C-index with DeLong 95% CI: {delong_ci['c_index']:.4f} [{delong_ci['ci_lower']:.4f}, {delong_ci['ci_upper']:.4f}]")

        # Bootstrap CI
        bootstrap_ci = xgb_model.bootstrap_c_index_ci(
            data_dict['X_test'],
            data_dict['y_time_test'],
            data_dict['y_event_test'],
            n_bootstrap=1000
        )
        print(f"Test C-index with Bootstrap 95% CI: {bootstrap_ci['c_index']:.4f} [{bootstrap_ci['ci_lower']:.4f}, {bootstrap_ci['ci_upper']:.4f}]")

    except Exception as e:
        print(f"Error calculating confidence intervals: {e}")

    # Plot training curves as separate PDFs
    print("\n" + "=" * 40)
    print("Plotting Training Curves (Individual PDFs)")
    print("=" * 40)
    xgb_model.plot_training_curves(prefix='train_val_test')

    # Plot feature importance as PDF
    print("\n" + "=" * 40)
    print("Feature Importance Analysis (PDF)")
    print("=" * 40)
    xgb_model.plot_feature_importance(top_n=20, prefix='feature_importance_train_val_test')

    # Plot survival predictions as separate PDFs
    print("\nGenerating survival prediction plots (Individual PDFs)...")
    xgb_model.plot_survival_predictions(
        data_dict['X_test'],
        data_dict['y_time_test'],
        data_dict['y_event_test'],
        prefix='survival_predictions_train_val_test'
    )

    # Create combined PDF report
    print("\nCreating combined PDF report...")
    xgb_model.create_combined_pdf_report(mode='train_val_test')

    # Print summary
    print("\n" + "=" * 80)
    print("Summary - Train/Validation/Test Mode")
    print("=" * 80)
    print(f"Model: XGBoost with Cox Proportional Hazards")
    print(f"Features: {len(data_dict['feature_names'])}")
    print(f"Training samples: {len(X_train_sub)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(data_dict['X_test'])}")
    print(f"Final Test C-index: {test_c_index:.4f}")
    print(f"Output directory: {xgb_model.output_dir}")
    print("=" * 80)

    return xgb_model, data_dict


def run_cv_mode(xgb_model, df):
    """
    Run enhanced 5-fold cross-validation mode with validation curves
    """

    # Prepare data for CV (no train/test split)
    print("\nPreparing data for enhanced 5-fold cross-validation...")
    data_dict = xgb_model.prepare_data_for_cv(df)

    print(f"\nData info:")
    print(f"   - Total samples: {len(data_dict['X_all'])} (100% used for CV)")
    print(f"   - Features: {len(data_dict['feature_names'])}")

    # Perform enhanced cross-validation with both training and validation curves
    print("\nRunning enhanced 5-fold cross-validation with complete curve tracking...")
    print("-" * 60)

    cv_results = xgb_model.cross_validate_with_curves(
        data_dict['X_all'],
        data_dict['y_time_all'],
        data_dict['y_event_all'],
        n_folds=5
    )

    # Extract results
    cv_scores = cv_results['cv_scores']
    fold_training_curves = cv_results['training_curves']

    # Results
    print("\n" + "=" * 40)
    print("Enhanced Cross-Validation Results")
    print("=" * 40)

    for i, score in enumerate(cv_scores):
        print(f"Fold {i+1}: C-index = {score:.4f}")

    cv_mean = np.mean(cv_scores)
    cv_std = np.std(cv_scores)
    print(f"\nFINAL CV RESULTS:")
    print(f"   Mean C-index: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"   Range: [{min(cv_scores):.4f}, {max(cv_scores):.4f}]")

    # Generate publication-ready tables (CSV format)
    print("\n" + "=" * 40)
    print("Generating Publication-Ready Tables (CSV)")
    print("=" * 40)
    table_results = xgb_model.generate_cv_results_tables(cv_scores, fold_training_curves, prefix='cv_results_table')

    # Generate table image (PDF format)
    print("\n" + "=" * 40)
    print("Creating Table Images (PDF)")
    print("=" * 40)
    table_image_results = xgb_model.create_cv_results_table_image(cv_scores, fold_training_curves, prefix='cv_table_image')

    # Create detailed comparison table image
    print("\nCreating Detailed Comparison Table Image...")
    xgb_model.create_detailed_comparison_table_image(cv_scores, fold_training_curves, prefix='detailed_comparison_table')

    # Plot enhanced CV training curves as individual PDFs
    print("\n" + "=" * 40)
    print("Plotting Enhanced Cross-Validation Training Curves (Individual PDFs)")
    print("=" * 40)
    xgb_model.plot_cv_training_curves(fold_training_curves, prefix='cv_mode')
    xgb_model.plot_cv_training_curves(fold_training_curves, prefix='cv_mode')

    # Plot CV summary bars with confidence intervals
    print("\n" + "=" * 40)
    print("Creating Cross-Validation Summary Plot with 95% Confidence Intervals")
    print("=" * 40)
    xgb_model.plot_cv_summary_bars_with_ci(cv_scores, prefix='cv_summary_bars_with_ci')

    # Also create the original summary plot for comparison
    xgb_model.plot_cv_summary_bars(cv_scores, prefix='cv_summary_bars')

    # Train final model on all data for feature importance
    print("\nTraining final model on all data for feature importance analysis...")
    xgb_model.train_final_model_cv(data_dict['X_all'], data_dict['y_all_survival'])

    # Plot feature importance as PDF
    print("\n" + "=" * 40)
    print("Feature Importance Analysis (PDF)")
    print("=" * 40)
    xgb_model.plot_feature_importance(top_n=20, prefix='feature_importance_cv_mode')

    # Enhanced SHAP Analysis with advanced visualizations
    print("\n" + "=" * 80)
    print("Enhanced SHAP Analysis with 5-Fold Cross-Validation")
    print("=" * 80)

    # Setup SHAP explainer using the final trained model
    print("Setting up SHAP explainer with trained model...")

    # Convert data to appropriate format for SHAP explainer
    if hasattr(data_dict['X_all'], 'values'):
        X_background = data_dict['X_all'].values
    else:
        X_background = data_dict['X_all']

    # Setup SHAP explainer with ALL training data as background for more accurate baseline
    xgb_model.setup_shap_explainer(X_background, max_background_samples=None)

    # Perform SHAP analysis using best fold validation set only
    shap_results = xgb_model.perform_shap_analysis_best_fold(
        data_dict['X_all'],
        data_dict['y_time_all'],
        data_dict['y_event_all'],
        cv_results,
        n_folds=5,
        prefix='shap_cv_analysis'
    )

    if shap_results:
        print("\n" + "=" * 60)
        print("Additional SHAP Visualizations (Used Features Only)")
        print("=" * 60)

        # Use a subset of the validation data for additional analysis
        combined_test_data = shap_results['validation_data']
        analysis_sample_size = min(100, len(combined_test_data))

        if hasattr(combined_test_data, 'sample'):
            X_analysis = combined_test_data.sample(n=analysis_sample_size, random_state=42)
        else:
            indices = np.random.choice(len(combined_test_data), analysis_sample_size, replace=False)
            X_analysis = combined_test_data.iloc[indices] if hasattr(combined_test_data, 'iloc') else combined_test_data[indices]

        # Convert to numpy array if needed
        if hasattr(X_analysis, 'values'):
            X_analysis_array = X_analysis.values
        else:
            X_analysis_array = X_analysis

        # SHAP directional importance (showing positive/negative impact)
        print("\nGenerating SHAP directional importance (showing risk increase/decrease)...")
        xgb_model.plot_shap_directional_importance(
            X_analysis_array,
            save_path='shap_directional_importance_used_features_cv.pdf',
            used_features_only=True
        )

        # Generate waterfall plots for interesting samples
        print("\nGenerating SHAP waterfall plots for sample cases...")

        # Get predictions for analysis samples to find interesting cases
        if hasattr(xgb_model, 'predict'):
            test_predictions = xgb_model.predict(X_analysis_array)
            high_risk_idx = np.argmax(test_predictions)  # Highest risk sample
            low_risk_idx = np.argmin(test_predictions)   # Lowest risk sample
        else:
            # Fallback: use first and middle samples
            high_risk_idx = 0
            low_risk_idx = analysis_sample_size // 2

        # Plot waterfall for high-risk sample (only used features)
        print(f"\nAnalyzing high-risk sample (index {high_risk_idx}) - used features only...")
        xgb_model.plot_shap_waterfall(
            X_analysis_array[high_risk_idx:high_risk_idx+1],
            sample_idx=0,
            save_path='shap_waterfall_high_risk_used_features_cv.pdf',
            used_features_only=True
        )

        # Plot waterfall for low-risk sample (only used features)
        print(f"\nAnalyzing low-risk sample (index {low_risk_idx}) - used features only...")
        xgb_model.plot_shap_waterfall(
            X_analysis_array[low_risk_idx:low_risk_idx+1],
            sample_idx=0,
            save_path='shap_waterfall_low_risk_used_features_cv.pdf',
            used_features_only=True
        )
    else:
        print("Warning: SHAP analysis failed, skipping additional visualizations.")

    # Create combined PDF report
    print("\nCreating combined PDF report...")
    xgb_model.create_combined_pdf_report(mode='cv_mode')

    # Print summary
    print("\n" + "=" * 80)
    print("Summary - Enhanced Cross-Validation Mode with Advanced SHAP Analysis")
    print("=" * 80)
    print(f"Model: XGBoost with Cox Proportional Hazards")
    print(f"Features: {len(data_dict['feature_names'])}")

    # Show used features info if available
    used_features_info = xgb_model.get_used_features_info()
    if used_features_info:
        print(f"Features used by model: {used_features_info['count']}/{len(data_dict['feature_names'])} ({used_features_info['count']/len(data_dict['feature_names'])*100:.1f}%)")

    print(f"Total samples: {len(data_dict['X_all'])}")
    print(f"CV Method: 5-fold stratified with complete curve tracking")
    print(f"Final CV C-index: {cv_mean:.4f} ± {cv_std:.4f}")
    print(f"Enhancement: Both training and validation curves tracked")

    if shap_results:
        # Handle different SHAP result formats (old vs new method)
        if 'n_successful_folds' in shap_results:
            print(f"SHAP Analysis: Completed across {shap_results['n_successful_folds']} folds")
        else:
            print(f"SHAP Analysis: Completed using best fold (Fold {shap_results['best_fold_idx']})")
        print(f"Top feature: {shap_results['feature_importance'].iloc[0]['Feature']}")
        print(f"SHAP Visualizations: Summary plots, directional importance, waterfall plots")
        if used_features_info:
            print(f"SHAP Focus: Used features only ({used_features_info['count']} features)")

    print(f"Output directory: {xgb_model.output_dir}")
    print("=" * 80)

    return xgb_model, data_dict, cv_results, shap_results


if __name__ == "__main__":
    import sys

    # Choose mode
    print("Enhanced XGBoost Survival Analysis - Mode Selection")
    print("=" * 50)
    print("Available modes:")
    print("1. 'train_val_test' - Traditional train/validation/test split")
    print("2. 'cv_only' - Enhanced 5-fold cross-validation with validation curves")
    print("=" * 50)

    # Automatically run cross-validation mode (you can change this to 'train_val_test' if preferred)
    selected_mode = 'cv_only'  # Change to 'train_val_test' if you want the other mode

    print(f"\nRunning mode: {selected_mode}")

    # Execute the selected mode
    try:
        if selected_mode == 'train_val_test':
            model, data_dict = main(mode='train_val_test')
        else:
            model, data_dict, cv_results, shap_results = main(mode='cv_only')

        print("\nProgram completed successfully!")
        print(f"All individual PDF files saved in: {model.output_dir}")

    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
