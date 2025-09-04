import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from typing import Tuple, Dict, List
import logging

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ImprovedBaseballSystem:
    """Improved Baseball Analytics System with Classification Approach"""
    
    def __init__(self, data_dir: str = "../data"):
        self.data_dir = data_dir
        self.scaler = None
        self.label_encoder = None
        self.model = None
        self.stance_clusters = None
        
        # Improved outcome mapping - more realistic scoring
        self.outcome_mapping = {
            'swinging_strike': 0,        # Pitcher wins
            'swinging_strike_blocked': 0,
            'called_strike': 0,
            'missed_bunt': 0,
            'pitchout': 1,               # Neutral
            'blocked_ball': 1,
            'foul': 1,                   # Neutral - extends AB
            'bunt_foul_tip': 1,
            'foul_bunt': 1,
            'foul_tip': 0,               # Strike
            'ball': 2,                   # Hitter wins
            'hit_by_pitch': 2,
            'hit_into_play': 2           # Opportunity for hitter
        }
        
    def load_and_preprocess_data(self) -> pd.DataFrame:
        """Enhanced data loading with better preprocessing"""
        logger.info("Loading and preprocessing data...")
        
        try:
            # Load datasets
            pitch_df = pd.read_csv(os.path.join(self.data_dir, "sample_statcast.csv"))
            stance_df = pd.read_csv(os.path.join(self.data_dir, "batting-stance.csv"))
            
            logger.info(f"Loaded {len(pitch_df)} pitch records and {len(stance_df)} stance records")
            
            # Clean and normalize names
            pitch_df['name'] = pitch_df['player_name'].apply(
                lambda x: ', '.join(x.strip().split()[::-1]) if isinstance(x, str) and x.strip() else x
            )
            
            # Enhanced date processing
            pitch_df['game_date'] = pd.to_datetime(pitch_df['game_date'], errors='coerce')
            pitch_df = pitch_df.dropna(subset=['game_date'])
            
            # Add temporal features
            pitch_df['year'] = pitch_df['game_date'].dt.year
            pitch_df['month'] = pitch_df['game_date'].dt.month
            pitch_df['day_of_year'] = pitch_df['game_date'].dt.dayofyear
            
            # Process stance dates
            stance_df['stance_date'] = pd.to_datetime(
                stance_df['year'].astype(str) + '-' + 
                stance_df['api_game_date_month_mm'].astype(str).str.zfill(2) + '-01',
                errors='coerce'
            )
            stance_df = stance_df.dropna(subset=['stance_date'])
            
            # Merge with temporal matching
            merged_df = self._improved_temporal_merge(pitch_df, stance_df)
            
            # Enhanced feature creation
            merged_df = self._create_enhanced_features(merged_df)
            
            return merged_df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _improved_temporal_merge(self, pitch_df: pd.DataFrame, stance_df: pd.DataFrame) -> pd.DataFrame:
        """Improved temporal matching with better performance"""
        logger.info("Performing improved temporal matching...")
        
        pitch_df_reset = pitch_df.reset_index().rename(columns={'index': 'pitch_index'})
        merged = pitch_df_reset.merge(stance_df, on='name', how='left', suffixes=('', '_stance'))
        
        # Only keep rows with stance data
        merged = merged.dropna(subset=['stance_date', 'game_date'])
        
        # More efficient date matching
        merged['date_diff'] = (merged['stance_date'] - merged['game_date']).abs().dt.days
        
        # Keep only closest match per pitch (within reasonable timeframe)
        merged = merged[merged['date_diff'] <= 90]  # Within 3 months
        idx = merged.groupby('pitch_index')['date_diff'].idxmin()
        closest_stance = merged.loc[idx].set_index('pitch_index')
        
        # Merge back with original pitch data
        stance_cols = [
            'avg_batter_y_position', 'avg_batter_x_position', 'avg_foot_sep',
            'avg_stance_angle', 'avg_intercept_y_vs_batter', 'avg_intercept_y_vs_plate',
            'bat_side', 'stance_date'
        ]
        
        result_df = pitch_df.reset_index(drop=True).join(closest_stance[stance_cols])
        
        return result_df
    
    def _create_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive enhanced features"""
        logger.info("Creating enhanced features...")
        
        # Basic pitch location features
        df['distance_from_center'] = np.sqrt(df['plate_x']**2 + (df['plate_z'] - 2.5)**2)
        df['in_strike_zone'] = ((df['plate_x'].abs() <= 0.83) & 
                               (df['plate_z'] >= 1.5) & (df['plate_z'] <= 3.5)).astype(int)
        
        # Edge of zone (borderline pitches)
        df['edge_of_zone'] = (((df['plate_x'].abs() > 0.7) & (df['plate_x'].abs() <= 1.0)) |
                             ((df['plate_z'] < 1.7) & (df['plate_z'] >= 1.3)) |
                             ((df['plate_z'] > 3.3) & (df['plate_z'] <= 3.7))).astype(int)
        
        # Movement features
        df['total_movement'] = np.sqrt(df['pfx_x']**2 + df['pfx_z']**2)
        df['horizontal_movement'] = df['pfx_x'].abs()
        df['vertical_movement'] = df['pfx_z']
        
        # Release point features
        df['effective_velocity'] = df['release_speed'] * (df['release_extension'] / 6.0)  # Normalized
        df['release_height_normalized'] = df['release_pos_z'] - 6.0  # Relative to mound
        
        # Count-based features (if available)
        if 'balls' in df.columns and 'strikes' in df.columns:
            df['count_pressure'] = df['strikes'] - df['balls']
            df['pitcher_ahead'] = (df['strikes'] > df['balls']).astype(int)
            df['hitter_ahead'] = (df['balls'] > df['strikes']).astype(int)
            df['two_strike_count'] = (df['strikes'] == 2).astype(int)
            df['three_ball_count'] = (df['balls'] == 3).astype(int)
            df['full_count'] = ((df['balls'] == 3) & (df['strikes'] == 2)).astype(int)
        else:
            # Create dummy count features
            df['count_pressure'] = 0
            df['pitcher_ahead'] = 0
            df['hitter_ahead'] = 0
            df['two_strike_count'] = 0
            df['three_ball_count'] = 0
            df['full_count'] = 0
        
        # Stance-derived features
        if 'avg_foot_sep' in df.columns and 'avg_stance_angle' in df.columns:
            df['stance_openness'] = df['avg_stance_angle'].abs()
            df['stance_width_ratio'] = df['avg_foot_sep'] / (df['avg_batter_y_position'] + 1e-6)
            df['stance_closed'] = (df['avg_stance_angle'] < -15).astype(int)
            df['stance_open'] = (df['avg_stance_angle'] > 15).astype(int)
            df['stance_wide'] = (df['avg_foot_sep'] > 35).astype(int)
            df['stance_narrow'] = (df['avg_foot_sep'] < 25).astype(int)
        
        # Pitch type groupings
        fastball_types = ['FF', 'FA', 'FT', 'FC', 'SI']
        breaking_types = ['SL', 'CU', 'KC', 'SV', 'ST']
        offspeed_types = ['CH', 'FS', 'FO', 'SC']
        
        df['is_fastball'] = df['pitch_type'].isin(fastball_types).astype(int)
        df['is_breaking'] = df['pitch_type'].isin(breaking_types).astype(int)
        df['is_offspeed'] = df['pitch_type'].isin(offspeed_types).astype(int)
        
        # Speed relative to pitch type average
        pitch_type_avg_speed = df.groupby('pitch_type')['release_speed'].transform('mean')
        df['speed_vs_pitch_type_avg'] = df['release_speed'] - pitch_type_avg_speed
        
        return df
    
    def advanced_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive imputation strategy"""
        logger.info("Performing advanced imputation...")
        
        # Stance features
        stance_features = [
            'avg_batter_y_position', 'avg_batter_x_position', 'avg_foot_sep',
            'avg_stance_angle', 'avg_intercept_y_vs_batter', 'avg_intercept_y_vs_plate'
        ]
        
        # Pitch features  
        pitch_features = [
            'release_speed', 'release_pos_x', 'release_pos_z',
            'plate_x', 'plate_z', 'pfx_x', 'pfx_z',
            'zone', 'release_spin_rate', 'release_extension'
        ]
        
        # Multi-level imputation for stance features
        for col in stance_features:
            if col in df.columns:
                # Player-specific median
                df[col] = df.groupby('name')[col].transform(
                    lambda x: x.fillna(x.median())
                )
                # Handedness-specific median
                df[col] = df.groupby('stand')[col].transform(
                    lambda x: x.fillna(x.median())
                )
                # Global median
                df[col] = df[col].fillna(df[col].median())
        
        # Multi-level imputation for pitch features
        for col in pitch_features:
            if col in df.columns:
                # Pitch-type specific median
                df[col] = df.groupby('pitch_type')[col].transform(
                    lambda x: x.fillna(x.median())
                )
                # Global median
                df[col] = df[col].fillna(df[col].median())
        
        # Categorical variables
        for col in ['bat_side', 'stand', 'pitch_type']:
            if col in df.columns:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val.iloc[0])
        
        return df
    
    def perform_stance_clustering(self, df: pd.DataFrame, n_clusters: int = 5) -> Dict:
        """Enhanced stance clustering"""
        logger.info(f"Performing stance clustering with {n_clusters} clusters...")
        
        # Encode categorical
        df['bat_side_enc'] = df['bat_side'].map({'R': 0, 'L': 1}).fillna(0.5)
        
        stance_features = [
            'avg_batter_y_position', 'avg_batter_x_position', 'avg_foot_sep',
            'avg_stance_angle', 'avg_intercept_y_vs_batter', 'avg_intercept_y_vs_plate',
            'bat_side_enc'
        ]
        
        # Clean data
        stance_data = df[stance_features].dropna()
        
        if len(stance_data) == 0:
            logger.warning("No valid stance data for clustering")
            return {}
        
        # Scale and cluster
        scaler = RobustScaler()
        stance_scaled = scaler.fit_transform(stance_data)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(stance_scaled)
        
        # Add to dataframe
        df['stance_cluster'] = np.nan
        df.loc[stance_data.index, 'stance_cluster'] = clusters
        
        # Calculate stats
        cluster_stats = {}
        for i in range(n_clusters):
            mask = df['stance_cluster'] == i
            if mask.sum() > 0:
                cluster_data = df[mask][stance_features]
                cluster_stats[i] = {
                    'size': mask.sum(),
                    'means': cluster_data.mean().to_dict(),
                    'stds': cluster_data.std().to_dict()
                }
        
        self.stance_clusters = {
            'model': kmeans,
            'scaler': scaler,
            'features': stance_features,
            'stats': cluster_stats
        }
        
        return cluster_stats

class ImprovedPitchClassifier(nn.Module):
    """Simplified classification model"""
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_classes: int = 3, dropout: float = 0.4):
        super().__init__()
        
        self.input_norm = nn.BatchNorm1d(input_size)
        
        
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_size // 4, num_classes)
        )
        
    def forward(self, x):
        if len(x.shape) == 3:  # If sequences, flatten
            batch_size, seq_len, features = x.shape
            x = x.reshape(batch_size, seq_len * features)
        
        x = self.input_norm(x)
        return self.layers(x)

def create_pitch_sequences(df: pd.DataFrame, feature_cols: List[str], 
                          label_col: str, seq_len: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """Create sequences from consecutive pitches to same batter"""
    logger.info(f"Creating sequences of length {seq_len}...")
    
    sequences = []
    labels = []
    
    # Group by game and batter
    grouped = df.groupby(['game_date', 'name'])
    
    for (date, player), group in grouped:
        if len(group) < seq_len:
            continue
            
        # Sort by some sequence indicator (inning, at_bat, etc.)
        if 'inning' in group.columns and 'at_bat_number' in group.columns:
            group = group.sort_values(['inning', 'at_bat_number'])
        else:
            group = group.sort_index()
        
        features = group[feature_cols].values
        targets = group[label_col].values
        
        # Create overlapping sequences
        for i in range(len(group) - seq_len + 1):
            seq_features = features[i:i+seq_len]
            seq_label = targets[i+seq_len-1]  # Predict outcome of last pitch in sequence
            
            # Check for missing values
            if not (np.isnan(seq_features).any() or np.isnan(seq_label)):
                sequences.append(seq_features)
                labels.append(seq_label)
    
    return np.array(sequences), np.array(labels)

def create_single_pitch_features(df: pd.DataFrame, feature_cols: List[str], 
                                label_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """Create single pitch features (no sequences)"""
    logger.info("Creating single pitch features...")
    
    # Clean data
    clean_df = df[feature_cols + [label_col]].dropna()
    
    X = clean_df[feature_cols].values
    y = clean_df[label_col].values
    
    return X, y

def train_improved_model(X_train: torch.Tensor, y_train: torch.Tensor,
                        X_val: torch.Tensor, y_val: torch.Tensor,
                        input_size: int, num_classes: int = 3,
                        epochs: int = 100, batch_size: int = 1024,
                        learning_rate: float = 0.001) -> Tuple[ImprovedPitchClassifier, List, List]:
    """Train the improved classification model"""
    
    # Handle class imbalance
    class_counts = Counter(y_train.numpy())
    total_samples = len(y_train)
    class_weights = {cls: total_samples / (len(class_counts) * count) 
                    for cls, count in class_counts.items()}
    
    weights = torch.tensor([class_weights[int(y.item())] for y in y_train], dtype=torch.float32)
    sampler = WeightedRandomSampler(weights, len(weights))
    
    # Model and training setup
    model = ImprovedPitchClassifier(input_size, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Data loaders
    train_dataset = TensorDataset(X_train, y_train.long())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    
    best_val_acc = 0
    patience_counter = 0
    patience = 15
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val.long()).item()
            val_predictions = torch.argmax(val_outputs, dim=1)
            val_accuracy = (val_predictions == y_val.long()).float().mean().item()
        
        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
        
        if epoch % 20 == 0:
            logger.info(f"Epoch {epoch}: Train Loss: {avg_train_loss:.4f}, "
                       f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses

def evaluate_classification_model(model: ImprovedPitchClassifier, 
                                 X_test: torch.Tensor, y_test: torch.Tensor) -> Dict:
    """Comprehensive evaluation of classification model"""
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        predictions = torch.argmax(outputs, dim=1).numpy()
        probabilities = torch.softmax(outputs, dim=1).numpy()
        actuals = y_test.numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(actuals, predictions)
    f1 = f1_score(actuals, predictions, average='weighted')
    
    # Class-wise metrics
    class_report = classification_report(actuals, predictions, 
                                       target_names=['Pitcher Win', 'Neutral', 'Hitter Win'],
                                       output_dict=True)
    
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Weighted F1: {f1:.4f}")
    logger.info("\nClassification Report:")
    print(classification_report(actuals, predictions, 
                              target_names=['Pitcher Win', 'Neutral', 'Hitter Win']))
    
    return {
        'accuracy': accuracy,
        'f1_weighted': f1,
        'classification_report': class_report,
        'predictions': predictions,
        'probabilities': probabilities,
        'actuals': actuals
    }

def analyze_stance_performance(df: pd.DataFrame) -> pd.DataFrame:
    """Enhanced stance performance analysis"""
    if 'stance_cluster' not in df.columns:
        logger.warning("No stance clusters found")
        return pd.DataFrame()
    
    analysis_results = []
    
    for cluster in sorted(df['stance_cluster'].dropna().unique()):
        cluster_data = df[df['stance_cluster'] == cluster]
        
        result = {
            'stance_cluster': int(cluster),
            'total_pitches': len(cluster_data),
            'pitcher_win_rate': (cluster_data['outcome_category'] == 0).mean(),
            'neutral_rate': (cluster_data['outcome_category'] == 1).mean(),
            'hitter_win_rate': (cluster_data['outcome_category'] == 2).mean(),
            'avg_foot_separation': cluster_data['avg_foot_sep'].mean(),
            'avg_stance_angle': cluster_data['avg_stance_angle'].mean(),
            'avg_batter_position': cluster_data['avg_batter_x_position'].mean()
        }
        
        # Best and worst pitch types
        pitch_performance = cluster_data.groupby('pitch_type')['outcome_category'].agg(['mean', 'count'])
        pitch_performance = pitch_performance[pitch_performance['count'] >= 50]  # Min sample size
        
        if not pitch_performance.empty:
            best_pitch = pitch_performance['mean'].idxmin()  # Lowest score = best for pitcher
            worst_pitch = pitch_performance['mean'].idxmax()  # Highest score = best for hitter
            
            result['best_pitch_vs_stance'] = best_pitch
            result['worst_pitch_vs_stance'] = worst_pitch
            result[f'best_pitch_success_rate'] = 1 - pitch_performance.loc[best_pitch, 'mean']  # Invert for pitcher success
            result[f'worst_pitch_success_rate'] = 1 - pitch_performance.loc[worst_pitch, 'mean']
        
        analysis_results.append(result)
    
    return pd.DataFrame(analysis_results)

def main():
    """Main execution with improved pipeline"""
    logger.info("Starting improved baseball analytics system...")
    
    try:
        # Initialize system
        system = ImprovedBaseballSystem()
        
        # Load and preprocess data
        df = system.load_and_preprocess_data()
        df = system.advanced_imputation(df)
        
        # Map outcomes to categories
        df['outcome_category'] = df['description'].map(system.outcome_mapping)
        df = df.dropna(subset=['outcome_category'])
        
        logger.info(f"Outcome distribution: {Counter(df['outcome_category'])}")
        
        # Perform stance clustering
        cluster_stats = system.perform_stance_clustering(df)
        
        # Define features for model
        feature_cols = [
            # Basic pitch features
            'release_speed', 'release_pos_x', 'release_pos_z',
            'plate_x', 'plate_z', 'pfx_x', 'pfx_z',
            'zone', 'release_spin_rate', 'release_extension',
            
            # Enhanced features
            'distance_from_center', 'in_strike_zone', 'edge_of_zone',
            'total_movement', 'horizontal_movement', 'vertical_movement',
            'effective_velocity', 'release_height_normalized',
            
            # Count features
            'count_pressure', 'pitcher_ahead', 'hitter_ahead',
            'two_strike_count', 'three_ball_count', 'full_count',
            
            # Stance features
            'avg_batter_y_position', 'avg_batter_x_position',
            'avg_foot_sep', 'avg_stance_angle',
            'avg_intercept_y_vs_batter', 'avg_intercept_y_vs_plate',
            'stance_openness', 'stance_width_ratio',
            'stance_closed', 'stance_open', 'stance_wide', 'stance_narrow',
            
            # Pitch type features
            'is_fastball', 'is_breaking', 'is_offspeed',
            'speed_vs_pitch_type_avg'
        ]
        
        # Filter available features
        available_features = [col for col in feature_cols if col in df.columns]
        logger.info(f"Using {len(available_features)} features")
        
        # Create features (single pitch, not sequences)
        X, y = create_single_pitch_features(df, available_features, 'outcome_category')
        logger.info(f"Created {X.shape[0]} samples with {X.shape[1]} features")
        
        # Stratified split to maintain class balance
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(sss.split(X, y))
        
        X_temp, X_test = X[train_idx], X[test_idx]
        y_temp, y_test = y[train_idx], y[test_idx]
        
        # Further split for validation
        sss_val = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, val_idx = next(sss_val.split(X_temp, y_temp))
        
        X_train, X_val = X_temp[train_idx], X_temp[val_idx]
        y_train, y_val = y_temp[train_idx], y_temp[val_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)
        
        # Convert to tensors
        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
        
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
        
        # Train model
        logger.info("Training improved classification model...")
        model, train_losses, val_losses = train_improved_model(
            X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor,
            input_size=X_train_scaled.shape[1], num_classes=3
        )
        
        # Evaluate model
        results = evaluate_classification_model(model, X_test_tensor, y_test_tensor)
        
        # Analyze stance effectiveness
        stance_analysis = analyze_stance_performance(df)
        
        if not stance_analysis.empty:
            logger.info("\nStance Analysis Results:")
            print(stance_analysis.round(4).to_string(index=False))
        
        # Print cluster insights
        logger.info("\nCluster Insights:")
        for cluster_id, stats in cluster_stats.items():
            logger.info(f"\nCluster {cluster_id}: {stats['size']} pitches")
            logger.info(f"  Avg foot separation: {stats['means']['avg_foot_sep']:.2f}")
            logger.info(f"  Avg stance angle: {stats['means']['avg_stance_angle']:.2f}°")
            logger.info(f"  Handedness mix: {stats['means']['bat_side_enc']:.2f}")
        
        # Generate recommendations
        recommendations = generate_recommendations(stance_analysis)
        logger.info("\nActionable Recommendations:")
        for category, recs in recommendations.items():
            logger.info(f"\n{category}:")
            for rec in recs:
                logger.info(f"  • {rec}")
        
        return system, model, results, df, stance_analysis
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

def generate_recommendations(stance_analysis: pd.DataFrame) -> Dict[str, List[str]]:
    """Generate actionable recommendations based on stance analysis"""
    
    if stance_analysis.empty:
        return {"No Analysis": ["Insufficient stance data for recommendations"]}
    
    recommendations = {
        "Pitching Strategy": [],
        "Hitting Adjustments": [],
        "Coaching Insights": []
    }
    
    # Sort clusters by pitcher success rate
    stance_analysis_sorted = stance_analysis.sort_values('pitcher_win_rate', ascending=False)
    
    most_vulnerable = stance_analysis_sorted.iloc[0]  # Highest pitcher win rate
    least_vulnerable = stance_analysis_sorted.iloc[-1]  # Lowest pitcher win rate
    
    # Pitching recommendations
    recommendations["Pitching Strategy"].extend([
        f"Target Cluster {int(most_vulnerable['stance_cluster'])} stances (foot sep: {most_vulnerable['avg_foot_separation']:.1f}, "
        f"angle: {most_vulnerable['avg_stance_angle']:.1f}°) - {most_vulnerable['pitcher_win_rate']:.1%} success rate",
        
        f"Avoid attacking Cluster {int(least_vulnerable['stance_cluster'])} stances directly - "
        f"only {least_vulnerable['pitcher_win_rate']:.1%} pitcher success rate",
    ])
    
    # Add pitch-specific recommendations if available
    for _, row in stance_analysis.iterrows():
        cluster = int(row['stance_cluster'])
        if 'best_pitch_vs_stance' in row and pd.notna(row['best_pitch_vs_stance']):
            best_pitch = row['best_pitch_vs_stance']
            success_rate = row['best_pitch_success_rate']
            recommendations["Pitching Strategy"].append(
                f"Use {best_pitch} vs Cluster {cluster} - {success_rate:.1%} effectiveness"
            )
    
    # Hitting recommendations
    recommendations["Hitting Adjustments"].extend([
        f"Players with narrow stances should consider widening (avg foot sep < 27)",
        f"Extremely closed stances (< -20°) may benefit from slight opening",
        f"Wide stances (> 35 foot separation) show good overall performance"
    ])
    
    # Add specific stance recommendations
    for _, row in stance_analysis.iterrows():
        cluster = int(row['stance_cluster'])
        if row['hitter_win_rate'] < 0.25:  # Low hitter success
            angle = row['avg_stance_angle']
            foot_sep = row['avg_foot_separation']
            recommendations["Hitting Adjustments"].append(
                f"Cluster {cluster} hitters: Consider adjusting from {angle:.1f}° angle, "
                f"{foot_sep:.1f} foot separation for better outcomes"
            )
    
    # Coaching insights
    recommendations["Coaching Insights"].extend([
        f"Stance clustering reveals {len(stance_analysis)} distinct batting approaches",
        f"Best performing stance: {most_vulnerable['avg_foot_separation']:.1f} foot separation, "
        f"{most_vulnerable['avg_stance_angle']:.1f}° angle",
        f"Success rate variance across stances: "
        f"{stance_analysis['pitcher_win_rate'].std():.3f} (higher = more differentiation)"
    ])
    
    return recommendations

def plot_results(train_losses: List[float], val_losses: List[float], 
                results: Dict, stance_analysis: pd.DataFrame):
    """Create visualization of results"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training curves
    axes[0, 0].plot(train_losses, label='Training Loss')
    axes[0, 0].plot(val_losses, label='Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Curves')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(results['actuals'], results['predictions'])
    sns.heatmap(cm, annot=True, fmt='d', ax=axes[0, 1], 
                xticklabels=['Pitcher Win', 'Neutral', 'Hitter Win'],
                yticklabels=['Pitcher Win', 'Neutral', 'Hitter Win'])
    axes[0, 1].set_title('Confusion Matrix')
    
    # Stance cluster performance
    if not stance_analysis.empty:
        x_pos = range(len(stance_analysis))
        axes[1, 0].bar(x_pos, stance_analysis['pitcher_win_rate'], 
                      color='lightcoral', alpha=0.7)
        axes[1, 0].set_xlabel('Stance Cluster')
        axes[1, 0].set_ylabel('Pitcher Win Rate')
        axes[1, 0].set_title('Pitcher Success by Stance Cluster')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels([f"Cluster {int(c)}" for c in stance_analysis['stance_cluster']])
        axes[1, 0].grid(True, alpha=0.3)
        
        # Stance characteristics
        scatter = axes[1, 1].scatter(stance_analysis['avg_stance_angle'], 
                                   stance_analysis['avg_foot_separation'],
                                   c=stance_analysis['pitcher_win_rate'],
                                   s=stance_analysis['total_pitches']/100,
                                   alpha=0.7, cmap='RdYlBu_r')
        axes[1, 1].set_xlabel('Avg Stance Angle (degrees)')
        axes[1, 1].set_ylabel('Avg Foot Separation')
        axes[1, 1].set_title('Stance Characteristics vs Performance')
        plt.colorbar(scatter, ax=axes[1, 1], label='Pitcher Win Rate')
        
        # Add cluster labels
        for _, row in stance_analysis.iterrows():
            axes[1, 1].annotate(f"C{int(row['stance_cluster'])}", 
                              (row['avg_stance_angle'], row['avg_foot_separation']),
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('baseball_analysis_results.png', dpi=300, bbox_inches='tight')
    logger.info("Results visualization saved as 'baseball_analysis_results.png'")
    plt.show()

# Additional utility functions

def get_player_stance_recommendation(player_name: str, df: pd.DataFrame, 
                                   stance_analysis: pd.DataFrame) -> str:
    """Get specific recommendation for a player"""
    
    player_data = df[df['name'] == player_name]
    
    if player_data.empty:
        return f"No data found for player: {player_name}"
    
    if 'stance_cluster' not in player_data.columns:
        return "No stance cluster data available"
    
    player_cluster = player_data['stance_cluster'].mode()
    if player_cluster.empty:
        return "No stance cluster determined for this player"
    
    cluster = int(player_cluster.iloc[0])
    cluster_info = stance_analysis[stance_analysis['stance_cluster'] == cluster]
    
    if cluster_info.empty:
        return f"No analysis available for stance cluster {cluster}"
    
    cluster_row = cluster_info.iloc[0]
    
    recommendation = f"""
Player: {player_name}
Stance Cluster: {cluster}
Cluster Performance: {cluster_row['pitcher_win_rate']:.1%} pitcher success rate

Current Stance Characteristics:
- Foot Separation: {cluster_row['avg_foot_separation']:.1f}
- Stance Angle: {cluster_row['avg_stance_angle']:.1f}°

Recommendations:
"""
    
    if cluster_row['pitcher_win_rate'] > 0.4:  # High pitcher success = bad for hitter
        recommendation += "- Consider stance adjustments - current stance is vulnerable\n"
        recommendation += "- Work on timing vs dominant pitch types in this cluster\n"
    else:
        recommendation += "- Maintain current stance approach - shows good results\n"
        recommendation += "- Focus on consistency rather than major changes\n"
    
    if 'best_pitch_vs_stance' in cluster_row and pd.notna(cluster_row['best_pitch_vs_stance']):
        recommendation += f"- Be especially alert for {cluster_row['best_pitch_vs_stance']} pitches\n"
    
    return recommendation

def export_results(results: Dict, stance_analysis: pd.DataFrame, 
                  cluster_stats: Dict, filename: str = "baseball_analysis_results.json"):
    """Export comprehensive results to JSON"""
    
    import json
    
    export_data = {
        "model_performance": {
            "accuracy": float(results['accuracy']),
            "f1_weighted": float(results['f1_weighted']),
            "classification_report": results['classification_report']
        },
        "stance_analysis": stance_analysis.to_dict('records') if not stance_analysis.empty else [],
        "cluster_statistics": cluster_stats,
        "summary": {
            "total_clusters": len(cluster_stats),
            "best_performing_cluster": int(stance_analysis.loc[stance_analysis['pitcher_win_rate'].idxmax(), 'stance_cluster']) if not stance_analysis.empty else None,
            "performance_range": {
                "min_pitcher_success": float(stance_analysis['pitcher_win_rate'].min()) if not stance_analysis.empty else None,
                "max_pitcher_success": float(stance_analysis['pitcher_win_rate'].max()) if not stance_analysis.empty else None
            }
        }
    }
    
    with open(filename, 'w') as f:
        json.dump(export_data, f, indent=2)
    
    logger.info(f"Results exported to {filename}")

if __name__ == "__main__":
    system, model, results, data, stance_analysis = main()
    
    # Optional: Create visualizations
    # plot_results(train_losses, val_losses, results, stance_analysis)
    
    # Optional: Export results
    # export_results(results, stance_analysis, system.stance_clusters['stats'])
    
    # Optional: Get recommendation for specific player
    # print(get_player_stance_recommendation("Olson, Matt", data, stance_analysis))