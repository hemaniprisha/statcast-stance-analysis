"""
Advanced Baseball Analytics System: Pitch Outcome Prediction & Batting Stance Classification

This module implements a comprehensive machine learning system for predicting MLB pitch outcomes
and identifying strategic batting stance vulnerabilities using Statcast data and biomechanical
batting stance measurements.

Key Features:
- Neural network classification of pitch outcomes (Pitcher Win, Neutral, Hitter Win)
- K-means clustering for batting stance archetype identification
- Advanced feature engineering combining pitch physics and stance biomechanics
- Temporal data matching between pitch events and stance measurements
- Actionable strategic insights for players, coaches, and analysts

Technical Approach:
- Feedforward neural network with batch normalization and dropout regularization
- Multi-level data imputation strategy with player/pitch-type/global fallbacks
- Robust data preprocessing pipeline with comprehensive error handling
- Statistical validation ensuring reproducibility across dataset sizes

Author: Prisha Hemani
Version: 2.0
Last Updated: 07/09/2025
"""

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

# Configure logging and suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class ImprovedBaseballSystem:
    """
    Main system class for baseball analytics with enhanced data processing and modeling capabilities.
    
    This class encapsulates the entire pipeline from raw data loading through model training
    and strategic insight generation. Designed for production use with robust error handling
    and comprehensive logging.
    
    Attributes:
        data_dir (str): Path to directory containing input CSV files
        scaler (StandardScaler): Feature normalization object (set during training)
        label_encoder (LabelEncoder): Target variable encoder (currently unused)
        model (ImprovedPitchClassifier): Trained neural network model
        stance_clusters (Dict): Clustering results and metadata
        outcome_mapping (Dict): Maps pitch descriptions to 3-class outcomes
    """
    
    def __init__(self, data_dir: str = "../data"):
        """
        Initialize the baseball analytics system.
        
        Args:
            data_dir (str): Path to directory containing batting-stance.csv and statcast data
        """
        self.data_dir = data_dir
        self.scaler = None
        self.label_encoder = None
        self.model = None
        self.stance_clusters = None
        
        # Outcome mapping strategy: Convert descriptive pitch outcomes to strategic categories
        # 0 = Pitcher Win (strikes, swinging strikes)
        # 1 = Neutral (fouls, blocked balls, pitchouts - extend at-bat)
        # 2 = Hitter Win (balls, HBP, contact opportunities)
        self.outcome_mapping = {
            'swinging_strike': 0,        # Clear pitcher advantage
            'swinging_strike_blocked': 0,
            'called_strike': 0,
            'missed_bunt': 0,
            'pitchout': 1,               # Strategic neutral outcome
            'blocked_ball': 1,           # Prevents advancement but no strike
            'foul': 1,                   # Extends at-bat, neutral value
            'bunt_foul_tip': 1,
            'foul_bunt': 1,
            'foul_tip': 0,               # Counts as strike
            'ball': 2,                   # Hitter advantage (improves count)
            'hit_by_pitch': 2,           # Free base for hitter
            'hit_into_play': 2           # Contact opportunity for hitter
        }
        
    def load_and_preprocess_data(self) -> pd.DataFrame:
        """
        Load and preprocess pitch and stance data with comprehensive error handling.
        
        This method handles the complex task of combining temporal pitch data with
        biomechanical stance measurements, including data validation, temporal matching,
        and feature engineering.
        
        Returns:
            pd.DataFrame: Merged dataset with pitch events and corresponding stance data
            
        Raises:
            FileNotFoundError: If required CSV files are not found in data directory
            ValueError: If data formats are incompatible or contain critical errors
        """
        logger.info("Loading and preprocessing data...")
        
        try:
            # Load primary datasets with explicit error handling
            pitch_df = pd.read_csv(os.path.join(self.data_dir, "sample_statcast.csv"))
            stance_df = pd.read_csv(os.path.join(self.data_dir, "batting-stance.csv"))
            
            logger.info(f"Loaded {len(pitch_df)} pitch records and {len(stance_df)} stance records")
            
            # Standardize player name format for consistent merging
            # Convert "First Last" to "Last, First" format used in stance data
            pitch_df['name'] = pitch_df['player_name'].apply(
                lambda x: ', '.join(x.strip().split()[::-1]) if isinstance(x, str) and x.strip() else x
            )
            
            # Enhanced date processing with comprehensive validation
            pitch_df['game_date'] = pd.to_datetime(pitch_df['game_date'], errors='coerce')
            pitch_df = pitch_df.dropna(subset=['game_date'])  # Remove invalid dates
            
            # Create temporal features for potential seasonal analysis
            pitch_df['year'] = pitch_df['game_date'].dt.year
            pitch_df['month'] = pitch_df['game_date'].dt.month
            pitch_df['day_of_year'] = pitch_df['game_date'].dt.dayofyear
            
            # Process stance measurement dates (monthly aggregations)
            # Stance data is aggregated by month, so we create first-of-month dates
            stance_df['stance_date'] = pd.to_datetime(
                stance_df['year'].astype(str) + '-' + 
                stance_df['api_game_date_month_mm'].astype(str).str.zfill(2) + '-01',
                errors='coerce'
            )
            stance_df = stance_df.dropna(subset=['stance_date'])
            
            # Perform sophisticated temporal matching between datasets
            merged_df = self._improved_temporal_merge(pitch_df, stance_df)
            
            # Generate comprehensive engineered features
            merged_df = self._create_enhanced_features(merged_df)
            
            return merged_df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _improved_temporal_merge(self, pitch_df: pd.DataFrame, stance_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform sophisticated temporal matching between pitch events and stance measurements.
        
        This is a critical component that matches individual pitch events with the closest
        available batting stance measurements in time. The challenge is that stance data
        is measured monthly while pitch data is per-event.
        
        Algorithm:
        1. Merge datasets on player name
        2. Calculate temporal distance between pitch date and stance measurement date
        3. For each pitch, select the stance measurement with minimum temporal distance
        4. Apply reasonable temporal constraints (within 90 days)
        
        Args:
            pitch_df (pd.DataFrame): Individual pitch event data
            stance_df (pd.DataFrame): Monthly aggregated stance measurement data
            
        Returns:
            pd.DataFrame: Pitch data with matched stance measurements
        """
        logger.info("Performing improved temporal matching...")
        
        # Preserve original pitch indices for accurate matching
        pitch_df_reset = pitch_df.reset_index().rename(columns={'index': 'pitch_index'})
        
        # Perform left join to keep all pitches, add stance data where available
        merged = pitch_df_reset.merge(stance_df, on='name', how='left', suffixes=('', '_stance'))
        
        # Filter to only pitches with available stance data
        merged = merged.dropna(subset=['stance_date', 'game_date'])
        
        # Calculate temporal distance in days for precise matching
        merged['date_diff'] = (merged['stance_date'] - merged['game_date']).abs().dt.days
        
        # Apply reasonable temporal constraint: stance data within 90 days of pitch
        # This prevents matching pitches with stance data from different seasons
        merged = merged[merged['date_diff'] <= 90]
        
        # For each pitch, select the stance measurement with minimum temporal distance
        idx = merged.groupby('pitch_index')['date_diff'].idxmin()
        closest_stance = merged.loc[idx].set_index('pitch_index')
        
        # Define stance features to retain in final dataset
        stance_cols = [
            'avg_batter_y_position',      # Vertical position in batter's box
            'avg_batter_x_position',      # Horizontal position in batter's box  
            'avg_foot_sep',               # Distance between feet (stance width)
            'avg_stance_angle',           # Angle of stance (closed/open)
            'avg_intercept_y_vs_batter',  # Biomechanical timing metric
            'avg_intercept_y_vs_plate',   # Position relative to home plate
            'bat_side',                   # Left/Right handed batter
            'stance_date'                 # Date of stance measurement
        ]
        
        # Merge stance data back to original pitch dataframe structure
        result_df = pitch_df.reset_index(drop=True).join(closest_stance[stance_cols])
        
        return result_df
    
    def _create_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate comprehensive engineered features combining pitch physics and stance biomechanics.
        
        This method creates 40+ features that capture the complex interactions between
        pitch characteristics and batter positioning. Features are designed based on
        baseball physics principles and strategic considerations.
        
        Feature Categories:
        1. Pitch Location: Strike zone analysis, distance metrics
        2. Pitch Movement: Horizontal/vertical break, total movement
        3. Release Point: Pitcher mechanics and deception metrics
        4. Count Situation: Strategic context features
        5. Stance Characteristics: Biomechanical positioning features
        6. Pitch Classification: Type groupings and relative metrics
        
        Args:
            df (pd.DataFrame): Merged pitch and stance data
            
        Returns:
            pd.DataFrame: Dataset with comprehensive engineered features
        """
        logger.info("Creating enhanced features...")
        
        # === PITCH LOCATION FEATURES ===
        # Distance from center of strike zone (plate center is 0,0, vertical center ~2.5ft)
        df['distance_from_center'] = np.sqrt(df['plate_x']**2 + (df['plate_z'] - 2.5)**2)
        
        # Binary indicator for pitches in official strike zone
        # Strike zone: ±0.83ft horizontally, 1.5-3.5ft vertically (approximate MLB zone)
        df['in_strike_zone'] = ((df['plate_x'].abs() <= 0.83) & 
                               (df['plate_z'] >= 1.5) & (df['plate_z'] <= 3.5)).astype(int)
        
        # Edge of zone indicator for borderline calls (critical for umpire decisions)
        df['edge_of_zone'] = (((df['plate_x'].abs() > 0.7) & (df['plate_x'].abs() <= 1.0)) |
                             ((df['plate_z'] < 1.7) & (df['plate_z'] >= 1.3)) |
                             ((df['plate_z'] > 3.3) & (df['plate_z'] <= 3.7))).astype(int)
        
        # === PITCH MOVEMENT FEATURES ===
        # Total break/movement magnitude (inches)
        df['total_movement'] = np.sqrt(df['pfx_x']**2 + df['pfx_z']**2)
        
        # Horizontal movement magnitude (slider/cutter break)
        df['horizontal_movement'] = df['pfx_x'].abs()
        
        # Vertical movement (gravity-adjusted drop/rise)
        df['vertical_movement'] = df['pfx_z']
        
        # === RELEASE POINT FEATURES ===
        # Effective velocity accounts for extension (closer release = faster perceived speed)
        df['effective_velocity'] = df['release_speed'] * (df['release_extension'] / 6.0)
        
        # Release height relative to pitcher's mound (6ft standard)
        df['release_height_normalized'] = df['release_pos_z'] - 6.0
        
        # === COUNT SITUATION FEATURES ===
        # These features capture the strategic context of each pitch
        if 'balls' in df.columns and 'strikes' in df.columns:
            df['count_pressure'] = df['strikes'] - df['balls']  # Positive = pitcher ahead
            df['pitcher_ahead'] = (df['strikes'] > df['balls']).astype(int)
            df['hitter_ahead'] = (df['balls'] > df['strikes']).astype(int)
            df['two_strike_count'] = (df['strikes'] == 2).astype(int)  # Defensive hitting
            df['three_ball_count'] = (df['balls'] == 3).astype(int)   # Must throw strike
            df['full_count'] = ((df['balls'] == 3) & (df['strikes'] == 2)).astype(int)
        else:
            # Create dummy features when count data unavailable
            df['count_pressure'] = 0
            df['pitcher_ahead'] = 0
            df['hitter_ahead'] = 0
            df['two_strike_count'] = 0
            df['three_ball_count'] = 0
            df['full_count'] = 0
        
        # === STANCE-DERIVED FEATURES ===
        # Advanced biomechanical features based on batting stance measurements
        if 'avg_foot_sep' in df.columns and 'avg_stance_angle' in df.columns:
            # Stance openness (absolute angle regardless of direction)
            df['stance_openness'] = df['avg_stance_angle'].abs()
            
            # Width-to-depth ratio (stability vs. mobility tradeoff)
            df['stance_width_ratio'] = df['avg_foot_sep'] / (df['avg_batter_y_position'] + 1e-6)
            
            # Categorical stance characteristics for strategic analysis
            df['stance_closed'] = (df['avg_stance_angle'] < -15).astype(int)   # Very closed
            df['stance_open'] = (df['avg_stance_angle'] > 15).astype(int)      # Very open
            df['stance_wide'] = (df['avg_foot_sep'] > 35).astype(int)          # Power stance
            df['stance_narrow'] = (df['avg_foot_sep'] < 25).astype(int)        # Contact stance
        
        # === PITCH TYPE CLASSIFICATION FEATURES ===
        # Group pitch types by strategic similarity
        fastball_types = ['FF', 'FA', 'FT', 'FC', 'SI']  # Velocity-based pitches
        breaking_types = ['SL', 'CU', 'KC', 'SV', 'ST']  # Sharp break pitches
        offspeed_types = ['CH', 'FS', 'FO', 'SC']        # Change of pace pitches
        
        df['is_fastball'] = df['pitch_type'].isin(fastball_types).astype(int)
        df['is_breaking'] = df['pitch_type'].isin(breaking_types).astype(int)
        df['is_offspeed'] = df['pitch_type'].isin(offspeed_types).astype(int)
        
        # Velocity relative to pitch type expectation (deception metric)
        pitch_type_avg_speed = df.groupby('pitch_type')['release_speed'].transform('mean')
        df['speed_vs_pitch_type_avg'] = df['release_speed'] - pitch_type_avg_speed
        
        return df
    
    def advanced_imputation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Implement sophisticated multi-level imputation strategy for missing data.
        
        Missing data is inevitable in baseball analytics due to equipment failures,
        measurement errors, and data collection gaps. This method implements a
        hierarchical imputation strategy that preserves statistical relationships.
        
        Imputation Strategy:
        1. Player-specific patterns (if player has other measurements)
        2. Group-specific patterns (handedness, pitch type)
        3. Global statistical measures (overall median/mode)
        
        Args:
            df (pd.DataFrame): Dataset with potential missing values
            
        Returns:
            pd.DataFrame: Dataset with imputed missing values
        """
        logger.info("Performing advanced imputation...")
        
        # Define feature categories for targeted imputation strategies
        stance_features = [
            'avg_batter_y_position', 'avg_batter_x_position', 'avg_foot_sep',
            'avg_stance_angle', 'avg_intercept_y_vs_batter', 'avg_intercept_y_vs_plate'
        ]
        
        pitch_features = [
            'release_speed', 'release_pos_x', 'release_pos_z',
            'plate_x', 'plate_z', 'pfx_x', 'pfx_z',
            'zone', 'release_spin_rate', 'release_extension'
        ]
        
        # === STANCE FEATURE IMPUTATION ===
        # Multi-level approach: player → handedness → global
        for col in stance_features:
            if col in df.columns:
                # Level 1: Player-specific median (player consistency)
                df[col] = df.groupby('name')[col].transform(
                    lambda x: x.fillna(x.median())
                )
                # Level 2: Handedness-specific median (biomechanical similarity)
                df[col] = df.groupby('stand')[col].transform(
                    lambda x: x.fillna(x.median())
                )
                # Level 3: Global median (population average)
                df[col] = df[col].fillna(df[col].median())
        
        # === PITCH FEATURE IMPUTATION ===
        # Multi-level approach: pitch type → global
        for col in pitch_features:
            if col in df.columns:
                # Level 1: Pitch-type specific median (physics consistency)
                df[col] = df.groupby('pitch_type')[col].transform(
                    lambda x: x.fillna(x.median())
                )
                # Level 2: Global median fallback
                df[col] = df[col].fillna(df[col].median())
        
        # === CATEGORICAL VARIABLE IMPUTATION ===
        # Use mode (most frequent value) for categorical features
        for col in ['bat_side', 'stand', 'pitch_type']:
            if col in df.columns:
                mode_val = df[col].mode()
                if len(mode_val) > 0:
                    df[col] = df[col].fillna(mode_val.iloc[0])
        
        return df
    
    def perform_stance_clustering(self, df: pd.DataFrame, n_clusters: int = 5) -> Dict:
        """
        Identify distinct batting stance archetypes using unsupervised K-means clustering.
        
        This method discovers natural groupings in batting stance characteristics,
        enabling strategic analysis of stance-based vulnerabilities and strengths.
        The number of clusters (5) was chosen based on baseball domain knowledge
        and statistical validation.
        
        Clustering Features:
        - Batter positioning (x, y coordinates)
        - Stance geometry (foot separation, angle)
        - Biomechanical timing (intercept metrics)
        - Handedness encoding
        
        Args:
            df (pd.DataFrame): Dataset with stance measurements
            n_clusters (int): Number of stance archetypes to identify (default: 5)
            
        Returns:
            Dict: Clustering results including model, statistics, and metadata
        """
        logger.info(f"Performing stance clustering with {n_clusters} clusters...")
        
        # Encode categorical handedness for clustering
        df['bat_side_enc'] = df['bat_side'].map({'R': 0, 'L': 1}).fillna(0.5)
        
        # Define features for clustering (exclude non-numeric and derived features)
        stance_features = [
            'avg_batter_y_position',      # Depth in batter's box
            'avg_batter_x_position',      # Distance from plate
            'avg_foot_sep',               # Stance width
            'avg_stance_angle',           # Open/closed orientation
            'avg_intercept_y_vs_batter',  # Biomechanical timing
            'avg_intercept_y_vs_plate',   # Plate coverage metric
            'bat_side_enc'                # Handedness (numeric)
        ]
        
        # Extract clean data for clustering (no missing values)
        stance_data = df[stance_features].dropna()
        
        if len(stance_data) == 0:
            logger.warning("No valid stance data for clustering")
            return {}
        
        # Use RobustScaler to handle outliers in biomechanical measurements
        scaler = RobustScaler()
        stance_scaled = scaler.fit_transform(stance_data)
        
        # Perform K-means clustering with multiple random initializations
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(stance_scaled)
        
        # Add cluster assignments back to original dataframe
        df['stance_cluster'] = np.nan
        df.loc[stance_data.index, 'stance_cluster'] = clusters
        
        # Calculate comprehensive statistics for each cluster
        cluster_stats = {}
        for i in range(n_clusters):
            mask = df['stance_cluster'] == i
            if mask.sum() > 0:
                cluster_data = df[mask][stance_features]
                cluster_stats[i] = {
                    'size': mask.sum(),                    # Number of pitches in cluster
                    'means': cluster_data.mean().to_dict(), # Average characteristics
                    'stds': cluster_data.std().to_dict()    # Variability measures
                }
        
        # Store clustering metadata for later use
        self.stance_clusters = {
            'model': kmeans,           # Fitted K-means model
            'scaler': scaler,          # Feature scaling parameters
            'features': stance_features, # Features used for clustering
            'stats': cluster_stats     # Statistical summaries
        }
        
        return cluster_stats


class ImprovedPitchClassifier(nn.Module):
    """
    Feedforward neural network for pitch outcome classification.
    
    This model predicts pitch outcomes using a simplified but effective architecture
    designed to prevent overfitting while capturing complex feature interactions.
    The architecture uses batch normalization and dropout for regularization.
    
    Architecture:
    - Input normalization layer
    - 3 hidden layers with decreasing size (128 → 64 → 32 neurons)
    - Batch normalization and dropout after each layer
    - Final classification layer (3 outputs: Pitcher Win, Neutral, Hitter Win)
    
    Design Principles:
    - Batch normalization for stable training and faster convergence
    - Dropout for preventing overfitting to training data
    - ReLU activation for handling sparse features effectively
    - Progressive dimensionality reduction for feature hierarchy
    """
    
    def __init__(self, input_size: int, hidden_size: int = 128, 
                 num_classes: int = 3, dropout: float = 0.4):
        """
        Initialize the neural network architecture.
        
        Args:
            input_size (int): Number of input features (40 in our case)
            hidden_size (int): Size of first hidden layer (default: 128)
            num_classes (int): Number of output classes (3: Pitcher/Neutral/Hitter Win)
            dropout (float): Dropout probability for regularization (default: 0.4)
        """
        super().__init__()
        
        # Input normalization to stabilize training
        self.input_norm = nn.BatchNorm1d(input_size)
        
        # Sequential architecture with progressive size reduction
        self.layers = nn.Sequential(
            # First hidden layer: Full feature representation
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Second hidden layer: Feature abstraction
            nn.Linear(hidden_size, hidden_size // 2),
            nn.BatchNorm1d(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Third hidden layer: High-level patterns
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.BatchNorm1d(hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Output layer: Classification predictions
            nn.Linear(hidden_size // 4, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input features, shape (batch_size, input_size) or
                            (batch_size, seq_len, features) for sequence data
                            
        Returns:
            torch.Tensor: Raw logits for each class, shape (batch_size, num_classes)
        """
        # Handle sequence data by flattening if necessary
        if len(x.shape) == 3:  # Convert sequences to single pitch features
            batch_size, seq_len, features = x.shape
            x = x.reshape(batch_size, seq_len * features)
        
        # Apply input normalization and forward pass
        x = self.input_norm(x)
        return self.layers(x)


def create_pitch_sequences(df: pd.DataFrame, feature_cols: List[str], 
                          label_col: str, seq_len: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequential pitch data for temporal modeling (currently unused in main pipeline).
    
    This function creates sequences of consecutive pitches to the same batter,
    which could be used for LSTM/RNN modeling to capture pitcher-hitter dynamics
    over the course of an at-bat.
    
    Args:
        df (pd.DataFrame): Pitch data with temporal ordering
        feature_cols (List[str]): Column names to use as features
        label_col (str): Target variable column name
        seq_len (int): Length of sequences to create (default: 3)
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (sequences, labels) for training
    """
    logger.info(f"Creating sequences of length {seq_len}...")
    
    sequences = []
    labels = []
    
    # Group pitches by game and batter for temporal continuity
    grouped = df.groupby(['game_date', 'name'])
    
    for (date, player), group in grouped:
        if len(group) < seq_len:
            continue  # Skip groups with insufficient data
            
        # Sort by temporal indicators for proper sequence ordering
        if 'inning' in group.columns and 'at_bat_number' in group.columns:
            group = group.sort_values(['inning', 'at_bat_number'])
        else:
            group = group.sort_index()  # Fallback to index ordering
        
        features = group[feature_cols].values
        targets = group[label_col].values
        
        # Create overlapping sequences (sliding window approach)
        for i in range(len(group) - seq_len + 1):
            seq_features = features[i:i+seq_len]
            seq_label = targets[i+seq_len-1]  # Predict outcome of final pitch in sequence
            
            # Quality check: ensure no missing values in sequence
            if not (np.isnan(seq_features).any() or np.isnan(seq_label)):
                sequences.append(seq_features)
                labels.append(seq_label)
    
    return np.array(sequences), np.array(labels)


def create_single_pitch_features(df: pd.DataFrame, feature_cols: List[str], 
                                label_col: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract single pitch features and targets from dataset (main approach used).
    
    This function creates the feature matrix and target vector used for training
    the feedforward neural network. Each row represents a single pitch with
    associated stance characteristics.
    
    Args:
        df (pd.DataFrame): Complete dataset with features and targets
        feature_cols (List[str]): Column names to use as model features
        label_col (str): Target variable column name
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: (features, targets) ready for model training
    """
    logger.info("Creating single pitch features...")
    
    # Remove rows with any missing values in features or target
    clean_df = df[feature_cols + [label_col]].dropna()
    
    # Extract feature matrix and target vector
    X = clean_df[feature_cols].values
    y = clean_df[label_col].values
    
    return X, y


def train_improved_model(X_train: torch.Tensor, y_train: torch.Tensor,
                        X_val: torch.Tensor, y_val: torch.Tensor,
                        input_size: int, num_classes: int = 3,
                        epochs: int = 100, batch_size: int = 1024,
                        learning_rate: float = 0.001) -> Tuple[ImprovedPitchClassifier, List, List]:
    """
    Train the neural network with advanced techniques for optimal performance.
    
    This function implements a comprehensive training pipeline with:
    - Class balancing through weighted sampling
    - Early stopping to prevent overfitting
    - Learning rate scheduling for convergence
    - Gradient clipping for training stability
    
    Args:
        X_train, y_train: Training data and labels
        X_val, y_val: Validation data and labels  
        input_size (int): Number of input features
        num_classes (int): Number of output classes
        epochs (int): Maximum training epochs
        batch_size (int): Mini-batch size for training
        learning_rate (float): Initial learning rate
        
    Returns:
        Tuple containing:
        - Trained model (ImprovedPitchClassifier)
        - Training loss history (List[float])
        - Validation loss history (List[float])
    """
    
    # === CLASS BALANCING SETUP ===
    # Handle imbalanced classes by computing sample weights
    class_counts = Counter(y_train.numpy())
    total_samples = len(y_train)
    class_weights = {cls: total_samples / (len(class_counts) * count) 
                    for cls, count in class_counts.items()}
    
    # Create weighted sampler to balance training batches
    weights = torch.tensor([class_weights[int(y.item())] for y in y_train], dtype=torch.float32)
    sampler = WeightedRandomSampler(weights, len(weights))
    
    # === MODEL AND OPTIMIZER SETUP ===
    model = ImprovedPitchClassifier(input_size, num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()  # Appropriate for multi-class classification
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
    """Comprehensively evaluate the trained classification model on a held-out test set.

    Args:
        model (ImprovedPitchClassifier): Trained neural network classifier.
        X_test (torch.Tensor): Scaled test features of shape (n_samples, n_features).
        y_test (torch.Tensor): Ground-truth class labels for the test set.

    Returns:
        Dict: Dictionary containing:
            - accuracy (float): Overall accuracy on the test set.
            - f1_weighted (float): Weighted F1 score across classes.
            - classification_report (dict): Per-class precision/recall/F1 support metrics.
            - predictions (np.ndarray): Predicted class indices for each sample.
            - probabilities (np.ndarray): Softmax probabilities per class, per sample.
            - actuals (np.ndarray): True labels for comparison.

    Notes:
        - The function logs summary metrics and prints a formatted classification report.
        - Softmax is applied to logits to expose calibrated class probabilities.
    """
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
    """Compute stance-cluster performance metrics and pitch-type insights.

    This aggregates pitch outcomes within each discovered stance cluster and
    derives summary statistics useful for strategy (e.g., pitcher win rate,
    hitter win rate), as well as the most and least effective pitch types
    against each stance (subject to a minimum sample size threshold).

    Args:
        df (pd.DataFrame): Dataset containing stance_cluster assignments and
            outcome_category labels.

    Returns:
        pd.DataFrame: One row per stance cluster with aggregated metrics such as
            total_pitches, pitcher_win_rate, neutral_rate, hitter_win_rate,
            average stance characteristics, and best/worst pitch types when
            available.
    """
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
    """Run the end-to-end pipeline: data prep, clustering, training, evaluation, insights.

    Steps:
        1) Load raw pitch and stance data, engineer features, and impute missing values.
        2) Map raw Statcast descriptions to 3-class outcome categories.
        3) Perform K-means stance clustering to discover stance archetypes.
        4) Build train/val/test splits with stratification and scale features.
        5) Train the neural network with class-balancing and early stopping.
        6) Evaluate on the held-out test set and log metrics.
        7) Analyze stance-cluster performance and generate recommendations.

    Returns:
        Tuple: (system, model, results, df, stance_analysis)
            - system: ImprovedBaseballSystem instance with clustering metadata.
            - model: Trained ImprovedPitchClassifier.
            - results: Dict with evaluation metrics and predictions.
            - df: Processed DataFrame used for modeling and analysis.
            - stance_analysis: Per-cluster performance summary DataFrame.
    """
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
    """Translate stance cluster metrics into concise, actionable guidance.

    Args:
        stance_analysis (pd.DataFrame): Output of analyze_stance_performance with
            per-cluster success rates and optional best/worst pitch types.

    Returns:
        Dict[str, List[str]]: Buckets of recommendations for pitching strategy,
            hitting adjustments, and coaching insights.
    """
    
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
    """Create a 2x2 dashboard of training curves, confusion matrix, and stance insights.

    Args:
        train_losses (List[float]): Per-epoch training loss values.
        val_losses (List[float]): Per-epoch validation loss values.
        results (Dict): Output from evaluate_classification_model containing
            predictions and actuals for confusion matrix.
        stance_analysis (pd.DataFrame): Aggregated stance performance metrics for
            bar and scatter plots.

    Notes:
        - Saves a high-resolution image to 'baseball_analysis_results.png'.
        - Displays the figure for interactive review.
    """
    
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
    """Generate a player-centric summary and guidance from their dominant stance cluster.

    Args:
        player_name (str): Name formatted to match df['name'] (e.g., 'Olson, Matt').
        df (pd.DataFrame): Processed dataset containing 'name' and 'stance_cluster'.
        stance_analysis (pd.DataFrame): Cluster-level performance summary.

    Returns:
        str: Readable multi-line recommendation with stance characteristics and
            high-level advice based on cluster performance.
    """
    
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
