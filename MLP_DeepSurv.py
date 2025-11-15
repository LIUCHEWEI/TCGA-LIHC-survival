try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
except AttributeError as e:
    print(f"PyTorch import error: {e}")
    print("Please restart your kernel and try again.")
    raise

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
from scipy import stats
import seaborn as sns
import os

# SHAP imports
try:
    import shap
    SHAP_AVAILABLE = True
    print("SHAP library loaded successfully")
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP library not available. Install with: pip install shap")

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class SurvivalDataset(Dataset):
    def __init__(self, features, times, events):
        self.features = torch.FloatTensor(features)
        self.times = torch.FloatTensor(times)
        self.events = torch.FloatTensor(events)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'time': self.times[idx],
            'event': self.events[idx]
        }

class ProgressiveMLPExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=32, dropout_rate=0.5):
        super(ProgressiveMLPExtractor, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        self.output_dim = hidden_dim
        self._init_weights()

        print(f"MLP Extractor: {input_dim} → {hidden_dim} (parameters: {input_dim * hidden_dim + hidden_dim})")

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.layers(x)

class ProgressiveDeepSurv(nn.Module):
    """Modified DeepSurv predictor: 32 → 8 → 1"""
    def __init__(self, input_dim, dropout_rates=[0.5, 0.3]):
        super(ProgressiveDeepSurv, self).__init__()

        self.layers = nn.Sequential(
            # First layer: 32 → 8
            nn.Linear(input_dim, 8),
            nn.BatchNorm1d(8),
            nn.GELU(),
            nn.Dropout(dropout_rates[0]),

            # Output layer: 8 → 1
            nn.Dropout(dropout_rates[1]),
            nn.Linear(8, 1)
        )

        self._init_weights()

        # Update parameter calculation
        params = (input_dim * 8 + 8) + (8 * 1 + 1)
        print(f"DeepSurv Predictor: {input_dim} → 8 → 1 (parameters: {params})")

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        return self.layers(x)

class ProgressiveMLPDeepSurv(nn.Module):
    """Improved MLP model for survival analysis: 78 → 32 → 8 → 1"""
    def __init__(self, input_dim, hidden_dim=32, dropout_rate=0.5):
        super(ProgressiveMLPDeepSurv, self).__init__()
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        
        self.survival_model = nn.Sequential(
            nn.Linear(hidden_dim, 8),
            nn.BatchNorm1d(8),
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.8),
            nn.Linear(8, 1)
        )
        
        self._init_weights()
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total parameters: {total_params}")
        
    def _init_weights(self):
        """Improved weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward propagation method"""
        features = self.feature_extractor(x)
        risk_scores = self.survival_model(features)
        return risk_scores, features

    def predict_risk_scores(self, x):
        """Prediction method for SHAP compatibility"""
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            
            # Handle device mismatch
            if hasattr(self, 'device_'):
                x = x.to(self.device_)
            else:
                # Try to infer device from model parameters
                try:
                    device = next(self.parameters()).device
                    x = x.to(device)
                    self.device_ = device
                except:
                    pass  # Stay on CPU
            
            # Handle batch dimension
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            risk_scores, _ = self(x)
            return risk_scores.cpu().numpy()

def stable_cox_loss(risk_scores, times, events):
    """Stable Cox loss function"""
    current_device = risk_scores.device if hasattr(risk_scores, 'device') and risk_scores.nelement() > 0 else \
                     (times.device if hasattr(times, 'device') and times.nelement() > 0 else \
                     (events.device if hasattr(events, 'device') and events.nelement() > 0 else 'cpu'))

    if risk_scores.nelement() == 0:
        return torch.tensor(0.0, device=current_device, requires_grad=True)

    if events.sum() == 0:
        return risk_scores.sum() * torch.tensor(0.0, device=risk_scores.device)

    risk_scores = risk_scores.squeeze()
    if risk_scores.dim() == 0:
        risk_scores = risk_scores.unsqueeze(0)

    sorted_indices = torch.argsort(times, descending=True)
    risk_scores_s = risk_scores[sorted_indices]
    times_s = times[sorted_indices]
    events_s = events[sorted_indices]

    risk_scores_c = risk_scores_s - risk_scores_s.mean()
    event_idx_s = torch.where(events_s == 1)[0]

    if len(event_idx_s) == 0:
        return risk_scores.sum() * torch.tensor(0.0, device=risk_scores.device)

    log_likelihood = torch.tensor(0.0, device=risk_scores.device, dtype=risk_scores_c.dtype)

    for i in event_idx_s:
        current_risk_score = risk_scores_c[i]
        at_risk_pool_indices = torch.where(times_s >= times_s[i])[0]
        at_risk_hazard_scores = risk_scores_c[at_risk_pool_indices]

        if at_risk_hazard_scores.nelement() == 0:
            continue

        max_hazard_score = torch.max(at_risk_hazard_scores)
        log_sum_exp_val = max_hazard_score + torch.log(torch.sum(torch.exp(at_risk_hazard_scores - max_hazard_score)) + 1e-8)
        log_likelihood = log_likelihood + (current_risk_score - log_sum_exp_val)

    num_actual_events = torch.tensor(len(event_idx_s), dtype=risk_scores.dtype, device=risk_scores.device)
    if num_actual_events == 0:
        return risk_scores.sum() * torch.tensor(0.0, device=risk_scores.device)

    return -log_likelihood / num_actual_events

class OptimizedTrainer:
    """Optimized trainer for survival model"""
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.train_losses = []
        self.val_losses = []
        self.train_c_indices = []
        self.val_c_indices = []

    def compute_c_index(self, risk_scores, times, events):
        """Calculate C-index"""
        try:
            if len(risk_scores) == 0 or events.sum() == 0:
                return 0.5

            risk_scores_np = risk_scores.detach().cpu().numpy().flatten()
            times_np = times.detach().cpu().numpy().flatten()
            events_np = events.detach().cpu().numpy().flatten()

            if np.any(np.isnan(risk_scores_np)) or np.any(np.isinf(risk_scores_np)) or \
               np.any(np.isnan(times_np)) or np.any(np.isinf(times_np)) or \
               np.any(np.isnan(events_np)) or np.any(np.isinf(events_np)):
                return 0.5

            c_index = concordance_index(times_np, -risk_scores_np, events_np)
            return c_index if not np.isnan(c_index) else 0.5
        except Exception as e:
            return 0.5

    def train_epoch(self, train_loader, optimizer):
        self.model.train()
        total_loss = 0
        all_risk_scores, all_times, all_events = [], [], []

        for batch_idx, batch in enumerate(train_loader):
            features = batch['features'].to(self.device)
            times = batch['time'].to(self.device)
            events = batch['event'].to(self.device)

            if features.nelement() == 0:
                continue

            optimizer.zero_grad()
            risk_scores, _ = self.model(features)

            if risk_scores.nelement() == 0:
                continue

            cox_loss = stable_cox_loss(risk_scores, times, events)
            total_loss_batch = cox_loss

            if torch.isfinite(total_loss_batch) and total_loss_batch.requires_grad:
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += total_loss_batch.item()
                all_risk_scores.append(risk_scores.detach())
                all_times.append(times.detach())
                all_events.append(events.detach())

        if all_risk_scores:
            all_risk_scores_cat = torch.cat(all_risk_scores)
            all_times_cat = torch.cat(all_times)
            all_events_cat = torch.cat(all_events)
            train_c_index = self.compute_c_index(all_risk_scores_cat, all_times_cat, all_events_cat)
        else:
            train_c_index = 0.5

        return total_loss / max(len(train_loader), 1), train_c_index

    def validate_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0
        all_risk_scores, all_times, all_events = [], [], []

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                features = batch['features'].to(self.device)
                times = batch['time'].to(self.device)
                events = batch['event'].to(self.device)

                if features.nelement() == 0:
                    continue

                risk_scores, _ = self.model(features)

                if risk_scores.nelement() == 0:
                    continue

                cox_loss = stable_cox_loss(risk_scores, times, events)

                if torch.isfinite(cox_loss):
                    total_loss += cox_loss.item()

                all_risk_scores.append(risk_scores)
                all_times.append(times)
                all_events.append(events)

        if all_risk_scores:
            all_risk_scores_cat = torch.cat(all_risk_scores)
            all_times_cat = torch.cat(all_times)
            all_events_cat = torch.cat(all_events)
            val_c_index = self.compute_c_index(all_risk_scores_cat, all_times_cat, all_events_cat)
        else:
            val_c_index = 0.5

        return total_loss / max(len(val_loader), 1), val_c_index

    def train(self, train_loader, val_loader, epochs=200, lr=0.01,
              patience=20, model_save_path='best_progressive_model.pth', verbose=True):
        """Improved training process with EMA smoothing"""
        
        # EMA smoothing setup
        ema_alpha = 0.2
        best_val_metric = -float('inf')
        best_val_epoch = 0
        running_val_metric = None
        patience_counter = 0
        best_train_c_index = 0

        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=5e-3,  # Increased L2 regularization
            eps=1e-8,
            betas=(0.9, 0.999)
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,  # More aggressive LR reduction
            patience=patience//3,
            min_lr=lr * 0.001,
            threshold=0.001  # More sensitive threshold
        )

        if verbose:
            print(f"\nStarting progressive architecture training")
            print(f"Parameters: Epochs={epochs}, LR={lr}, Patience={patience}")
            print("=" * 80)

        for epoch in range(epochs):
            train_loss, train_c_index = self.train_epoch(train_loader, optimizer)
            val_loss, val_c_index = self.validate_epoch(val_loader)

            scheduler.step(val_c_index)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_c_indices.append(train_c_index)
            self.val_c_indices.append(val_c_index)

            if train_c_index > best_train_c_index:
                best_train_c_index = train_c_index

            # Update EMA of validation metric
            if running_val_metric is None:
                running_val_metric = val_c_index
            else:
                running_val_metric = (1 - ema_alpha) * running_val_metric + ema_alpha * val_c_index

            # Model saving logic based on smoothed metric
            if running_val_metric > best_val_metric:
                best_val_metric = running_val_metric
                best_val_epoch = epoch
                torch.save(self.model.state_dict(), model_save_path)
                patience_counter = 0
                if verbose:
                    print(f"New best! Epoch {epoch+1:3d} - Train C: {train_c_index:.4f} | Val C: {val_c_index:.4f} (EMA: {running_val_metric:.4f}) | Loss: {train_loss:.4f}")
            else:
                patience_counter += 1
                if verbose and epoch % 5 == 0:
                    print(f"   Epoch {epoch+1:3d} - Train C: {train_c_index:.4f} | Val C: {val_c_index:.4f} (EMA: {running_val_metric:.4f}) | Patience: {patience_counter}/{patience}")

            # Early stopping check with additional condition
            if patience_counter >= patience or \
               (epoch - best_val_epoch > patience//2 and running_val_metric < best_val_metric * 0.95):
                if verbose:
                    print(f"\nEarly stopping at epoch {epoch+1}")
                break

        if verbose:
            print("=" * 80)
            print(f"Progressive architecture training completed!")
            print(f"   Best training C-index: {best_train_c_index:.4f}")
            print(f"   Best validation C-index: {best_val_metric:.4f}")
            print(f"   Total training epochs: {len(self.train_losses)}")

        if os.path.exists(model_save_path):
            self.model.load_state_dict(torch.load(model_save_path))

        return best_val_metric

class SHAPAnalyzer:
    """SHAP Analysis class for PyTorch survival models (feature-name safe)"""

    def __init__(self, output_dir='outputs2'):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.feature_names = None
        self.background_data = None
        self.shap_explainer = None

    def setup_shap_explainer(self, model, background_data, feature_names=None, max_background_samples=100):
        """
        Setup SHAP explainer with column names preserved.
        - Keep background_data as a DataFrame with columns = feature_names
        - Use shap.maskers.Independent to bind names to the masker
        """
        if not SHAP_AVAILABLE:
            print("Error: SHAP library not available. Install with: pip install shap")
            return False

        print("Setting up SHAP explainer for PyTorch model...")
        self.feature_names = list(feature_names) if feature_names is not None else None

        # Downsample background if needed
        if isinstance(background_data, pd.DataFrame):
            bg = background_data.copy()
            if self.feature_names is not None and list(bg.columns) != self.feature_names:
                bg.columns = self.feature_names
            if len(bg) > max_background_samples:
                bg = bg.sample(n=max_background_samples, random_state=42)
        else:
            arr = np.asarray(background_data)
            if len(arr) > max_background_samples:
                idx = np.random.choice(len(arr), max_background_samples, replace=False)
                arr = arr[idx]
            bg = pd.DataFrame(arr, columns=self.feature_names)

        self.background_data = bg

        # model wrapper
        def model_predict(x):
            # x: DataFrame or ndarray
            x_np = x.values if isinstance(x, pd.DataFrame) else np.asarray(x)
            return model.predict_risk_scores(x_np).flatten()

        # masker with names
        try:
            masker = shap.maskers.Independent(self.background_data)
        except Exception:
            # fallback: pass raw array (still works, but names depend on DF later)
            masker = self.background_data.values

        # 直接使用 KernelExplainer，它更穩定且不依賴 numba
        try:
            self.shap_explainer = shap.KernelExplainer(model_predict, self.background_data)
            print(f"SHAP KernelExplainer setup completed with {len(self.background_data)} background samples")
            return True
        except Exception as e:
            print(f"Failed to setup SHAP explainer: {e}")
            return False

    def calculate_shap_values(self, X_data, max_evals=100):
        """
        Calculate SHAP values; ensure inputs carry column names.
        Returns (Explanation, X_df_values)
        """
        if not hasattr(self, 'shap_explainer'):
            raise ValueError("SHAP explainer not setup. Call setup_shap_explainer() first.")

        # sample for efficiency
        if isinstance(X_data, pd.DataFrame):
            X = X_data.copy()
        else:
            X = pd.DataFrame(np.asarray(X_data), columns=self.feature_names)

        if len(X) > max_evals:
            X = X.sample(n=max_evals, random_state=42)

        # make sure columns are correct
        if self.feature_names is not None and list(X.columns) != self.feature_names:
            X.columns = self.feature_names

        print(f"Calculating SHAP values for {len(X)} samples...")

        try:
            # KernelExplainer 返回的是 numpy array
            shap_values = self.shap_explainer.shap_values(X)
            # 創建一個類似 Explanation 的對象
            class ShapExplanation:
                def __init__(self, values, feature_names):
                    self.values = values
                    self.feature_names = feature_names
                    self.base_values = None
                    self.data = X.values
            
            explanation = ShapExplanation(shap_values, list(X.columns))
            return explanation, X.values
        except Exception as e:
            print(f"Error calculating SHAP values: {e}")
            return None, None

    def perform_shap_analysis_cv(self, X_all, y_time_all, y_event_all, all_trainers,
                                 cv_results, feature_names, n_folds=5, prefix='shap_analysis'):
        """
        Use best CV fold model for SHAP; X_all can be ndarray or DataFrame.
        """
        if not SHAP_AVAILABLE:
            print("Warning: SHAP library not available. Skipping SHAP analysis.")
            return None

        print("\n" + "=" * 80)
        print("SHAP Analysis with Cross-Validation Results")
        print("=" * 80)

        self.feature_names = list(feature_names)

        best_fold_idx = int(np.argmax(cv_results))
        best_trainer = all_trainers[best_fold_idx]
        if best_trainer is None:
            print("Error: No valid trainer found for SHAP analysis")
            return None

        print(f"Using model from best performing fold: Fold {best_fold_idx + 1} (C-index: {cv_results[best_fold_idx]:.4f})")

        # ---- Ensure DataFrame with names for background ----
        X_all_df = X_all.copy() if isinstance(X_all, pd.DataFrame) else pd.DataFrame(X_all, columns=self.feature_names)

        # Setup and compute
        if not self.setup_shap_explainer(best_trainer.model, X_all_df, feature_names=self.feature_names, max_background_samples=100):
            print("Failed to setup SHAP explainer")
            return None

        print("\nCalculating SHAP values...")
        shap_values, X_sample = self.calculate_shap_values(X_all_df, max_evals=100)
        if shap_values is None:
            print("Failed to calculate SHAP values")
            return None

        # importance
        shap_values_array = shap_values.values if hasattr(shap_values, 'values') else np.asarray(shap_values)
        importance_scores = np.abs(shap_values_array).mean(axis=0)

        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'SHAP_Importance': importance_scores
        }).sort_values('SHAP_Importance', ascending=False)

        out_csv = os.path.join(self.output_dir, f'{prefix}_feature_importance_cv.csv')
        importance_df.to_csv(out_csv, index=False)
        print(f"SHAP feature importance saved to: {out_csv}")

        # plots
        print("\nCreating SHAP visualizations...")
        self._create_shap_visualizations(shap_values, X_sample, importance_df, prefix=prefix)

        return {
            'shap_values': shap_values,
            'sample_data': X_sample,
            'feature_importance': importance_df,
            'best_fold': best_fold_idx + 1,
            'n_samples_analyzed': shap_values_array.shape[0]
        }

    def _create_shap_visualizations(self, shap_values, test_data, importance_df, prefix='shap_analysis'):
        """Create plots with correct feature names."""
        print("   Creating SHAP visualizations...")

        # ensure feature names exist on Explanation
        try:
            if hasattr(shap_values, 'feature_names'):
                if not shap_values.feature_names:
                    shap_values.feature_names = self.feature_names
        except Exception:
            pass

        # 1) summary / beeswarm
        print("   Creating SHAP summary plot...")
        plt.figure(figsize=(10, 6))
        try:
            shap.summary_plot(shap_values.values, test_data, feature_names=self.feature_names, show=False, max_display=20)
            plt.tight_layout()
            path1 = os.path.join(self.output_dir, f'{prefix}_summary_plot.pdf')
            plt.savefig(path1, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"      Summary plot saved to: {path1}")
        except Exception as e:
            print(f"      Error creating summary plot: {e}")
            plt.close()

        # 2) bar
        print("   Creating SHAP bar plot...")
        plt.figure(figsize=(8, 6))
        try:
            shap.summary_plot(shap_values.values, test_data, feature_names=self.feature_names, plot_type="bar", show=False, max_display=20)
            plt.tight_layout()
            path2 = os.path.join(self.output_dir, f'{prefix}_bar_plot.pdf')
            plt.savefig(path2, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"      Bar plot saved to: {path2}")
        except Exception as e:
            print(f"      Error creating bar plot: {e}")
            plt.close()

        # 3) custom importance (top 15)
        print("   Creating feature importance plot...")
        plt.figure(figsize=(8, max(4, len(importance_df.head(15)) * 0.3)))
        top_features = importance_df.head(15)
        y_pos = np.arange(len(top_features))
        plt.barh(y_pos, top_features['SHAP_Importance'], alpha=0.8)
        plt.yticks(y_pos, top_features['Feature'])
        plt.xlabel('Mean |SHAP value|')
        plt.title('Top 15 Feature Importance (SHAP)', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        path3 = os.path.join(self.output_dir, f'{prefix}_importance_plot.pdf')
        plt.savefig(path3, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"      Feature importance plot saved to: {path3}")

        # 4) waterfall of one sample
        print("   Creating SHAP waterfall plot for sample case...")
        try:
            plt.figure(figsize=(10, 8))
            shap.plots._waterfall.waterfall_legacy(shap_values.base_values[0] if shap_values.base_values is not None else 0,
                                            shap_values.values[0],
                                            feature_names=self.feature_names,
                                            max_display=15,
                                            show=False)
            plt.title('SHAP Waterfall Plot (Sample Patient)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            path4 = os.path.join(self.output_dir, f'{prefix}_waterfall_example.pdf')
            plt.savefig(path4, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"      Waterfall plot saved to: {path4}")
        except Exception as e:
            print(f"      Error creating waterfall plot: {e}")
            plt.close()

        # 5) directional
        print("   Creating directional importance plot...")
        shap_values_array = shap_values.values if hasattr(shap_values, 'values') else np.asarray(shap_values)
        self.plot_shap_directional_importance(shap_values_array, test_data,
                                              save_path=f'{prefix}_directional_importance.pdf')
        print("   All SHAP visualizations completed!")

    def plot_shap_directional_importance(self, shap_values_array, test_data,
                                         save_path='shap_directional_importance.pdf'):
        """Same as before; unchanged except relies on self.feature_names."""
        plt.figure(figsize=(12, max(6, len(self.feature_names) * 0.3)))
        mean_shap = np.mean(shap_values_array, axis=0)
        feature_importance = list(zip(self.feature_names, mean_shap))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        top_features = feature_importance[:20]
        features, importances = zip(*top_features)
        colors = ['darkred' if imp < 0 else 'darkblue' for imp in importances]
        y_pos = np.arange(len(features))
        bars = plt.barh(y_pos, importances, color=colors, alpha=0.7)
        plt.yticks(y_pos, features)
        plt.xlabel('Mean SHAP Value (Directional Impact)', fontsize=12)
        plt.title('SHAP Directional Feature Importance', fontsize=14)
        plt.gca().invert_yaxis()
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
        for bar, importance in zip(bars, importances):
            w = bar.get_width()
            if w >= 0:
                plt.text(w + abs(w)*0.02, bar.get_y()+bar.get_height()/2, f'{importance:.4f}',
                         ha='left', va='center', fontsize=9)
            else:
                plt.text(w - abs(w)*0.02, bar.get_y()+bar.get_height()/2, f'{importance:.4f}',
                         ha='right', va='center', fontsize=9)
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
        print(f"      Directional importance plot saved to: {directional_path}")


def prepare_data_unified(df, time_col='OS_time', event_col='OS_event', test_size=0.15, for_cv=False):
    """Improved data preparation function with data balancing strategy"""
    from sklearn.utils import resample  # Add this import at the top of the file
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

    feature_cols = [col for col in df.columns if col not in exclude_cols]
    processed_df = df.copy()

    # Cancer staging ordinal encoding
    staging_mappings = {
        'AJCC Pathologic T-Stage': {
            'TX': -1.0, 'T1': 1.0, 'T2': 2.0, 'T2a': 2.1, 'T2b': 2.2,
            'T3': 3.0, 'T3a': 3.1, 'T3b': 3.2, 'T4': 4.0
        },
        'AJCC Pathologic N-Stage': {
            'N0': 0.0, 'N1': 1.0, 'NX': -1.0
        },
        'AJCC Pathologic M-Stage': {
            'M0': 0.0, 'M1': 1.0, 'MX': -1.0
        },
        'AJCC Pathologic Stage': {
            'Stage I': 1.0, 'Stage II': 2.0, 'Stage III': 3.0,
            'Stage IIIA': 3.1, 'Stage IIIB': 3.2, 'Stage IIIC': 3.3,
            'Stage IV': 4.0, 'Stage IVA': 4.1, 'Stage IVB': 4.2
        }
    }

    # Apply ordinal encoding to staging columns
    for col, mapping in staging_mappings.items():
        if col in processed_df.columns and col in feature_cols:
            processed_df[col] = processed_df[col].map(mapping)
            unmapped_mask = processed_df[col].isna() & processed_df[col].notna()
            if unmapped_mask.any():
                processed_df[col] = processed_df[col].fillna(0.0)

    # Handle other categorical variables (non-staging columns)
    label_encoders = {}
    for col in feature_cols.copy():
        if col in staging_mappings:
            continue

        if processed_df[col].dtype == 'object' or pd.api.types.is_categorical_dtype(processed_df[col]):
            if processed_df[col].nunique() <= 1:
                feature_cols.remove(col)
                processed_df.drop(columns=[col], inplace=True)
                continue
            try:
                str_series = processed_df[col].astype(str)
                if str_series.isnull().sum() > 0:
                    str_series.fillna('Missing', inplace=True)

                le = LabelEncoder()
                processed_df[col] = le.fit_transform(str_series)
                label_encoders[col] = le
            except Exception as e:
                if col in feature_cols:
                    feature_cols.remove(col)
                processed_df.drop(columns=[col], inplace=True)

    # Handle missing values
    for col in feature_cols:
        if processed_df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(processed_df[col]):
                fill_value = processed_df[col].median()
                processed_df[col].fillna(fill_value, inplace=True)

    final_feature_cols = [col for col in feature_cols if col in processed_df.columns]
    if not final_feature_cols:
        raise ValueError("No feature columns remaining after preprocessing.")

    X = processed_df[final_feature_cols].values.astype(float)
    y_time = processed_df[time_col].values.astype(float)
    y_event = processed_df[event_col].values.astype(float)

    print(f"Actual feature count: {X.shape[1]}")
    print(f"Sample count: {X.shape[0]}")

    # For cross-validation, return data without balancing
    if for_cv:
        print(f"Events: {y_event.sum():.0f} ({y_event.mean():.1%})")
        return {
            'X': X,
            'y_time': y_time,
            'y_event': y_event,
            'features': final_feature_cols,
            'label_encoders': label_encoders,
            'staging_mappings': staging_mappings
        }
    else:
        X_train, X_test, y_time_train, y_time_test, y_event_train, y_event_test = train_test_split(
            X, y_time, y_event,
            test_size=test_size,
            random_state=42,
            stratify=y_event
        )

        # 使用StandardScaler進行z-score標準化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return {
            'X_train': X_train_scaled, 'X_test': X_test_scaled,
            'y_time_train': y_time_train, 'y_time_test': y_time_test,
            'y_event_train': y_event_train, 'y_event_test': y_event_test,
            'features': final_feature_cols,
            'scaler': scaler,
            'label_encoders': label_encoders,
            'staging_mappings': staging_mappings
        }

def plot_fold_training_history(trainer, fold_num, save_dir='outputs2'):
    """Plot training history for a specific fold and save as separate PDFs"""
    if not trainer.train_losses or not trainer.val_losses or \
       not trainer.train_c_indices or not trainer.val_c_indices:
        return

    # Create plots directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    epochs = range(1, len(trainer.train_losses) + 1)
    span_size = min(15, len(trainer.train_losses) // 3)

    # Smooth curves for better visualization
    # 对training曲线使用更强的平滑
    train_span_size = min(25, len(trainer.train_losses) // 2)  # 更大的平滑窗口
    val_span_size = min(15, len(trainer.train_losses) // 3)    # 保持原有的平滑窗口
    
    # Training curves: 更强的平滑
    smooth_train_loss = pd.Series(trainer.train_losses).ewm(span=train_span_size, adjust=False).mean()
    smooth_train_c = pd.Series(trainer.train_c_indices).ewm(span=train_span_size, adjust=False).mean()
    
    # Validation curves: 较轻的平滑
    smooth_val_loss = pd.Series(trainer.val_losses).ewm(span=val_span_size).mean()
    smooth_val_c = pd.Series(trainer.val_c_indices).ewm(span=val_span_size).mean()

    # Plot 1: Loss curves
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    ax1.plot(epochs, smooth_train_loss, color='blue', linewidth=2, label='Train')
    ax1.plot(epochs, smooth_val_loss, color='red', linewidth=2, label='Validation')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'Training Loss Curves - Fold {fold_num}', fontsize=14)
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    # Set fixed y-axis range for loss
    ax1.set_ylim([3.0, 8.5])
    plt.tight_layout()
    
    # Save loss plot as PDF
    loss_pdf_path = os.path.join(save_dir, f'fold_{fold_num}_loss_curves.pdf')
    plt.savefig(loss_pdf_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig1)

    # Plot 2: C-index curves
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    ax2.plot(epochs, smooth_train_c, color='blue', linewidth=2, label='Train')
    ax2.plot(epochs, smooth_val_c, color='red', linewidth=2, label='Validation')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('C-index', fontsize=12)
    ax2.set_title(f'Training C-index Curves - Fold {fold_num}', fontsize=14)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Set fixed y-axis range for C-index
    ax2.set_ylim([0.35, 1.0])
    plt.tight_layout()
    
    # Save C-index plot as PDF
    cindex_pdf_path = os.path.join(save_dir, f'fold_{fold_num}_cindex_curves.pdf')
    plt.savefig(cindex_pdf_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)

    print(f"  - Saved: {loss_pdf_path}")
    print(f"  - Saved: {cindex_pdf_path}")

def plot_all_folds_summary(all_trainers, cv_results, save_dir='outputs2'):
    """Plot summary of all folds and save as separate PDFs"""
    # Create plots directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Plot 1: Cross-Validation Results Summary
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    folds = range(1, len(cv_results) + 1)
    bars = ax1.bar(folds, cv_results, color='skyblue', edgecolor='navy', linewidth=1.5)
    mean_c_index = np.mean(cv_results)
    std_c_index = np.std(cv_results)

    ax1.axhline(y=mean_c_index, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_c_index:.4f}')
    ax1.fill_between([0.5, len(cv_results) + 0.5], mean_c_index - std_c_index, mean_c_index + std_c_index,
                    alpha=0.2, color='red', label=f'±1 SD: {std_c_index:.4f}')

    for bar, value in zip(bars, cv_results):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.005, f'{value:.4f}', ha='center', va='bottom', fontsize=10)

    ax1.set_xlabel('Fold', fontsize=12)
    ax1.set_ylabel('Final C-index', fontsize=12)
    ax1.set_title('Cross-Validation Results Summary', fontsize=14)
    ax1.set_xticks(folds)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    cv_summary_path = os.path.join(save_dir, 'cross_validation_summary.pdf')
    plt.savefig(cv_summary_path, dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # Plot 2: Validation C-index Curves - All Folds
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_trainers)))
    for i, trainer in enumerate(all_trainers):
        if trainer and trainer.val_c_indices:
            epochs = range(1, len(trainer.val_c_indices) + 1)
            ax2.plot(epochs, trainer.val_c_indices, color=colors[i], alpha=0.7, label=f'Fold {i+1}')

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation C-index', fontsize=12)
    ax2.set_title('Validation C-index Curves - All Folds', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    # Set fixed y-axis range for C-index
    ax2.set_ylim([0.35, 1.0])
    plt.tight_layout()
    
    val_cindex_path = os.path.join(save_dir, 'validation_cindex_curves_all_folds.pdf')
    plt.savefig(val_cindex_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # Plot 3: Validation Loss Curves - All Folds
    fig3, ax3 = plt.subplots(1, 1, figsize=(12, 6))
    for i, trainer in enumerate(all_trainers):
        if trainer and trainer.val_losses:
            epochs = range(1, len(trainer.val_losses) + 1)
            ax3.plot(epochs, trainer.val_losses, color=colors[i], alpha=0.7, label=f'Fold {i+1}')

    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Validation Loss', fontsize=12)
    ax3.set_title('Validation Loss Curves - All Folds', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    # Set fixed y-axis range for loss
    ax3.set_ylim([3.0, 8.5])
    plt.tight_layout()
    
    val_loss_path = os.path.join(save_dir, 'validation_loss_curves_all_folds.pdf')
    plt.savefig(val_loss_path, dpi=300, bbox_inches='tight')
    plt.close(fig3)

    # Plot 4: Training vs Validation Performance
    fig4, ax4 = plt.subplots(1, 1, figsize=(10, 8))
    train_final = []
    val_final = []
    for trainer in all_trainers:
        if trainer and trainer.train_c_indices and trainer.val_c_indices:
            train_final.append(trainer.train_c_indices[-1])
            val_final.append(max(trainer.val_c_indices))

    if train_final and val_final:
        ax4.scatter(train_final, val_final, s=100, alpha=0.7, c=range(len(train_final)), cmap='tab10')
        for i, (train_c, val_c) in enumerate(zip(train_final, val_final)):
            ax4.annotate(f'Fold {i+1}', (train_c, val_c), xytext=(5, 5), textcoords='offset points', fontsize=9)

        # Add diagonal line
        min_val = min(min(train_final), min(val_final))
        max_val = max(max(train_final), max(val_final))
        ax4.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5, label='Perfect correlation')

        ax4.set_xlabel('Final Training C-index', fontsize=12)
        ax4.set_ylabel('Best Validation C-index', fontsize=12)
        ax4.set_title('Training vs Validation Performance', fontsize=14)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    plt.tight_layout()
    
    train_vs_val_path = os.path.join(save_dir, 'training_vs_validation_performance.pdf')
    plt.savefig(train_vs_val_path, dpi=300, bbox_inches='tight')
    plt.close(fig4)

    print(f"  - Saved: {cv_summary_path}")
    print(f"  - Saved: {val_cindex_path}")
    print(f"  - Saved: {val_loss_path}")
    print(f"  - Saved: {train_vs_val_path}")

def plot_training_curves_all_folds(all_trainers, save_dir='outputs2'):
    """Plot training loss and C-index curves for all folds"""
    # Create plots directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_trainers)))

    # Plot 1: Training Loss Curves - All Folds
    fig1, ax1 = plt.subplots(1, 1, figsize=(12, 6))
    for i, trainer in enumerate(all_trainers):
        if trainer and trainer.train_losses:
            epochs = range(1, len(trainer.train_losses) + 1)
            ax1.plot(epochs, trainer.train_losses, color=colors[i], alpha=0.7, label=f'Fold {i+1}')

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss Curves - All Folds', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Set fixed y-axis range for loss
    ax1.set_ylim([3.0, 8.5])
    plt.tight_layout()
    
    train_loss_path = os.path.join(save_dir, 'training_loss_curves_all_folds.pdf')
    plt.savefig(train_loss_path, dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # Plot 2: Training C-index Curves - All Folds
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 6))
    for i, trainer in enumerate(all_trainers):
        if trainer and trainer.train_c_indices:
            epochs = range(1, len(trainer.train_c_indices) + 1)
            ax2.plot(epochs, trainer.train_c_indices, color=colors[i], alpha=0.7, label=f'Fold {i+1}')

    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Training C-index', fontsize=12)
    ax2.set_title('Training C-index Curves - All Folds', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    # Set fixed y-axis range for C-index
    ax2.set_ylim([0.35, 1.0])
    plt.tight_layout()
    
    train_cindex_path = os.path.join(save_dir, 'training_cindex_curves_all_folds.pdf')
    plt.savefig(train_cindex_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)

    print(f"  - Saved: {train_loss_path}")
    print(f"  - Saved: {train_cindex_path}")

def improved_cross_validation_with_balanced_events(df, k_folds=5, time_col='OS_time', event_col='OS_event', test_size=0.15):
    """Cross-validation with detailed plotting for each fold and SHAP analysis (fixed for SHAP feature names)"""
    print("=== Starting 5-Fold Cross Validation with Detailed Plotting and SHAP Analysis ===")

    full_data = prepare_data_unified(df, time_col, event_col, test_size=test_size, for_cv=True)
    X_full, y_time_full, y_event_full = full_data['X'], full_data['y_time'], full_data['y_event']
    feature_names = full_data['features']  # 真正欄名

    # 使用StandardScaler進行z-score標準化全部資料
    scaler_cv = StandardScaler()
    X_full_scaled = scaler_cv.fit_transform(X_full)

    # 把 scaled 資料包成 DataFrame 並保留欄名（供 SHAP 使用）
    X_full_scaled_df = pd.DataFrame(X_full_scaled, columns=feature_names)

    print(f"Event distribution in full data:")
    print(f"  Total events: {y_event_full.sum():.0f} ({y_event_full.mean():.1%})")
    print(f"  Total censored: {(1-y_event_full).sum():.0f} ({(1-y_event_full).mean():.1%})")

    # 使用簡單的事件分層
    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_splits = list(skf.split(X_full_scaled, y_event_full))

    cv_results = []
    all_trainers = []
    input_dim_cv = X_full_scaled.shape[1]

    print(f"Data prepared: {X_full_scaled.shape[0]} samples, {input_dim_cv} features")
    print(f"Starting {k_folds}-fold cross validation...\n")

    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        print(f"{'='*60}")
        print(f"                    FOLD {fold+1}/{k_folds}")
        print(f"{'='*60}")

        # 這裡訓練仍用 numpy array 即可
        X_train_fold = X_full_scaled[train_idx]
        X_val_fold   = X_full_scaled[val_idx]
        y_time_train_fold, y_time_val_fold   = y_time_full[train_idx],  y_time_full[val_idx]
        y_event_train_fold, y_event_val_fold = y_event_full[train_idx], y_event_full[val_idx]

        print(f"Train samples: {len(X_train_fold)}, Validation samples: {len(X_val_fold)}")
        print(f"Train events: {y_event_train_fold.sum():.0f} ({y_event_train_fold.mean():.1%}), Validation events: {y_event_val_fold.sum():.0f} ({y_event_val_fold.mean():.1%})")

        val_event_rate = y_event_val_fold.mean()
        if val_event_rate < 0.15 or val_event_rate > 0.85:
            print(f"  警告: 驗證集事件率 ({val_event_rate:.1%}) 可能過於極端，C指數計算可能不可靠")
        if y_event_val_fold.sum() < 10:
            print(f"  警告: 驗證集事件數很少 ({y_event_val_fold.sum():.0f}) - C指數可能不可靠")

        # DataLoader
        train_dataset = SurvivalDataset(X_train_fold, y_time_train_fold, y_event_train_fold)
        val_dataset   = SurvivalDataset(X_val_fold,   y_time_val_fold,   y_event_val_fold)

        if len(train_dataset) == 0:
            print(f"跳過fold {fold+1}: 空的訓練資料集")
            all_trainers.append(None)
            continue

        batch_size  = len(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)

        # 建模 & 訓練
        model   = ProgressiveMLPDeepSurv(input_dim=input_dim_cv, hidden_dim=32, dropout_rate=0.5)
        trainer = OptimizedTrainer(model)
        model_save_path_fold = f'best_progressive_model_fold_{fold+1}.pth'

        best_c_index = trainer.train(
            train_loader, val_loader,
            epochs=200, lr=0.01, patience=35,
            model_save_path=model_save_path_fold, verbose=True
        )

        cv_results.append(best_c_index)
        all_trainers.append(trainer)

        print(f"\nFold {fold+1} 結果:")
        print(f"  最佳驗證 C-index: {best_c_index:.4f}")
        print(f"  訓練輪數總計: {len(trainer.train_losses)}")
        print(f"  最終訓練 C-index: {trainer.train_c_indices[-1]:.4f}")

        plot_fold_training_history(trainer, fold+1)

        if os.path.exists(model_save_path_fold):
            os.remove(model_save_path_fold)

        print(f"Fold {fold+1} 完成!\n")

    # 總結統計與圖
    mean_c_index = float(np.mean(cv_results)) if cv_results else 0.0
    std_c_index  = float(np.std(cv_results))  if cv_results else 0.0

    print(f"{'='*60}")
    print(f"              交叉驗證總結")
    print(f"{'='*60}")
    print(f"個別fold結果:")
    for i, r in enumerate(cv_results):
        print(f"  Fold {i+1}: {r:.4f}")
    print(f"\n整體統計:")
    print(f"  平均 C-index: {mean_c_index:.4f}")
    print(f"  標準差: {std_c_index:.4f}")
    print(f"  95% 信賴區間: [{mean_c_index - 1.96*std_c_index:.4f}, {mean_c_index + 1.96*std_c_index:.4f}]")

    print("\n" + "="*60)
    print("              生成總結圖表")
    print("="*60)
    plot_all_folds_summary(all_trainers, cv_results)
    plot_training_curves_all_folds(all_trainers)

    # ---------------- SHAP 分析（改用帶欄名的 DataFrame） ----------------
    if SHAP_AVAILABLE and any(trainer is not None for trainer in all_trainers):
        print("\n" + "="*80)
        print("              SHAP 分析開始")
        print("="*80)

        shap_analyzer = SHAPAnalyzer(output_dir='outputs2')

        # 使用全部數據進行SHAP分析
        shap_results = shap_analyzer.perform_shap_analysis_cv(
            X_full_scaled_df,     # 使用已經標準化的全部數據
            y_time_full,
            y_event_full,
            all_trainers,
            cv_results,
            feature_names,
            n_folds=k_folds,
            prefix='shap_pytorch_analysis'
        )

        print("="*80)
        print("              SHAP 分析完成")
        print("="*80)
    else:
        shap_results = None
        if not SHAP_AVAILABLE:
            print("\n警告: SHAP庫不可用，跳過SHAP分析")
        else:
            print("\n警告: 沒有有效的訓練器，跳過SHAP分析")

    return {
        'cv_results': cv_results,
        'mean_c_index': mean_c_index,
        'std_c_index': std_c_index,
        'all_trainers': all_trainers,
        'cv_data_size': len(X_full),
        'shap_results': shap_results,
        'feature_names': feature_names
    }

def main():
    """Main function focused on cross-validation only with SHAP analysis"""
    print("=== DeepSurv 5-Fold Cross Validation Analysis with SHAP ===")
    print("Architecture: 78 → 32 → 8 → 1")
    print("Focus: Detailed training curves for each fold + SHAP interpretability analysis\n")

    # Try to load data from possible paths
    possible_paths = [
        '/content/drive/MyDrive/lihc_mice_imputed_final_with_individual_primate_scores.csv',
        'lihc_mice_imputed_final_with_individual_primate_scores.csv',
        '/Users/liuzhewei/Downloads/lihc_mice_imputed_final_with_individual_primate_scores.csv'
    ]

    df = None
    for file_path in possible_paths:
        try:
            df = pd.read_csv(file_path)
            print(f"資料讀取成功: {df.shape} (路徑: {file_path})")
            break
        except FileNotFoundError:
            print(f"檔案 {file_path} 未找到。")
            continue
        except Exception as e:
            print(f"讀取 {file_path} 時發生錯誤: {e}")
            continue

    if df is None:
        print("錯誤: 找不到資料檔案。請確保資料檔案在正確位置。")
        print("預期檔案名稱: lihc_mice_imputed_final_with_individual_primate_scores.csv")
        return None

    # Run cross-validation with detailed plotting and SHAP analysis
    cv_analysis = improved_cross_validation_with_balanced_events(df, k_folds=5, test_size=0.15)

    print("\n" + "="*60)
    print("              分析完成")
    print("="*60)
    print("在 'outputs2/' 目錄中生成的PDF圖表:")
    print("  個別fold圖表:")
    print("    - fold_1_loss_curves.pdf 到 fold_5_loss_curves.pdf")
    print("    - fold_1_cindex_curves.pdf 到 fold_5_cindex_curves.pdf")
    print("  總結圖表:")
    print("    - cross_validation_summary.pdf")
    print("    - training_loss_curves_all_folds.pdf")
    print("    - training_cindex_curves_all_folds.pdf")
    print("    - validation_loss_curves_all_folds.pdf")
    print("    - validation_cindex_curves_all_folds.pdf")
    print("    - training_vs_validation_performance.pdf")
    
    if cv_analysis.get('shap_results'):
        print("  SHAP 分析圖表:")
        print("    - shap_pytorch_analysis_summary_plot.pdf")
        print("    - shap_pytorch_analysis_bar_plot.pdf")
        print("    - shap_pytorch_analysis_importance_plot.pdf")
        print("    - shap_pytorch_analysis_waterfall_example.pdf")
        print("    - shap_pytorch_analysis_directional_importance.pdf")
        print("    - shap_pytorch_analysis_feature_importance_cv.csv")
    
    print(f"\n最終交叉驗證效能: {cv_analysis['mean_c_index']:.4f} ± {cv_analysis['std_c_index']:.4f}")

    # Analyze CV performance issues
    analyze_cv_performance(cv_analysis)

    return cv_analysis

def analyze_cv_performance(cv_analysis):
    """Analyze cross-validation performance and identify potential issues"""
    print("\n" + "="*60)
    print("              CV 效能分析")
    print("="*60)

    cv_results = cv_analysis['cv_results']
    mean_c = cv_analysis['mean_c_index']
    std_c = cv_analysis['std_c_index']

    # Calculate coefficient of variation
    cv_coefficient = std_c / mean_c if mean_c > 0 else float('inf')

    print(f"效能統計:")
    print(f"  範圍: {min(cv_results):.4f} - {max(cv_results):.4f}")
    print(f"  變異係數: {cv_coefficient:.3f}")

    # Identify potential issues
    issues = []

    if std_c > 0.05:
        issues.append("fold間變異較大 (std > 0.05)")

    if cv_coefficient > 0.15:
        issues.append("變異係數較高 (> 15%)")

    if max(cv_results) - min(cv_results) > 0.15:
        issues.append("最佳和最差fold間差距較大 (> 0.15)")

    if any(c < 0.55 for c in cv_results):
        issues.append("某些fold效能較差 (C-index < 0.55)")

    if issues:
        print(f"\n⚠️  發現的潛在問題:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

        print(f"\n💡 可能原因與解決方案:")
        print(f"  • 事件分佈不平衡 → 嘗試分層抽樣")
        print(f"  • 驗證集過小 → 考慮不同的CV策略")
        print(f"  • 模型不穩定 → 調整學習率或架構")
        print(f"  • 資料異質性 → 檢查批次效應或離群值")

        if std_c > 0.08:
            print(f"  • 變異很大可能表示基本資料問題")
    else:
        print(f"\n✅ CV效能看起來穩定可靠")

    # Print SHAP analysis summary if available
    if cv_analysis.get('shap_results'):
        shap_results = cv_analysis['shap_results']
        print(f"\n📊 SHAP 分析總結:")
        print(f"  使用的模型: Fold {shap_results['best_fold']} (最佳效能)")
        print(f"  分析樣本數: {shap_results['n_samples_analyzed']}")
        print(f"  特徵重要性已儲存至CSV檔案")
        print(f"  已生成多種SHAP視覺化圖表")

if __name__ == "__main__":
    try:
        results = main()
        if results:
            print("交叉驗證與SHAP分析成功完成!")
        else:
            print("分析失敗 - 請檢查資料檔案路徑並重試。")
    except Exception as e:
        print(f"執行過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()