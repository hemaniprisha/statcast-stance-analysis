# Advanced Baseball Analytics: Pitch Outcome Prediction & Batting Stance Classification

This project implements a comprehensive machine learning system that predicts MLB pitch outcomes and identifies strategic batting stance vulnerabilities using Statcast data and biomechanical batting stance measurements.


Update: Organized project into intuitive file structure, added visualization module (09/07/2025) 

Note: This project was recently updated with improved modeling and results. Earlier versions achieved different performance metrics and a different technical framework (see commit history), but the current version reflects the most accurate implementation.

## Project Overview

**Key Achievements:**
- **55.8% prediction accuracy** on MLB pitch outcomes (3-class classification)
- **Discovered 5 distinct batting stance archetypes** with measurable performance differences
- **Analyzed 675,811+ pitch records** with statistical significance
- **Generated actionable insights** for pitchers, hitters, and coaches

**Technical Approach:**
- **Feedforward Neural Network** with batch normalization and dropout
- **40 engineered features** combining pitch physics and stance biomechanics  
- **K-means clustering** for stance classification
- **Advanced data pipeline** with temporal matching and imputation

## Key Findings

### Batting Stance Performance Rankings
1. **Closed Stance** (27.3% pitcher success) - Most effective for hitters
2. **Balanced Stance** (27.4% pitcher success) - Nearly tied for best  
3. **Wide/Compact Stances** (27.9% pitcher success) - Moderate effectiveness
4. **Standard Stance** (28.5% pitcher success) - Most vulnerable (48.6% of players)

### Strategic Insights
- **Standard stance users** should adjust toward balanced characteristics (+1.1% improvement)
- **Forkballs most effective** against compact & standard stances (61% of hitters)
- **Knuckleballs consistently ineffective** across all stance types
- **Stance classification enables targeted scouting** and pitch selection

## Project Structure

```
baseball-analytics/
├── README.md                    # Project documentation
├── requirements.txt             # Python dependencies
├── main.py                     # Main execution script
├── config.py                   # Configuration and constants
├── data/                       # Input data directory
│   ├── sample_statcast.csv     # Pitch-by-pitch data
│   └── batting-stance.csv      # Batting stance measurements
└── src/                        # Core functionality
    ├── __init__.py            # Package initialization
    ├── data_processing.py     # Data loading and preprocessing
    ├── models.py              # Neural network models
    ├── analysis.py            # Clustering and performance analysis
    └── visualization.py       # Plotting and export functions```
```
## Data Sources

### **batting-stance.csv** (~343 KB)
- **2,267 batting stance records** with biomechanical measurements
- Features: foot separation, stance angle, batter positioning
- Temporal data for player stance evolution

### **sample_statcast.csv** (~8.8 MB) 
- **14,100 pitch records** for rapid testing and development
- Statcast features: velocity, movement, location, spin rate
- Perfect for initial model validation

### **combined_statcast_data.csv** (~422 MB) - *Download Required*
- **675,811+ complete pitch records** for full analysis
- Download from [Baseball Savant Statcast Search](https://baseballsavant.mlb.com/statcast_search)
- Required for production-level results and statistical significance

> **Note:** By default, the system uses `sample_statcast.csv` for faster development. Switch to the full dataset for comprehensive analysis and production deployment.

## Data Sources & Requirements
Statcast Data (sample_statcast.csv)
Required columns:

Identifiers: player_name, game_date, description
Physics: release_speed, release_pos_x/z, plate_x/z, pfx_x/z
Context: pitch_type, zone, release_spin_rate, release_extension
Situational: balls, strikes (optional but recommended)

Stance Data (batting-stance.csv)
Required columns:

Identity: name (format: "Last, First")
Temporal: year, api_game_date_month_mm
Positioning: avg_batter_y_position, avg_batter_x_position
Geometry: avg_foot_sep, avg_stance_angle
Biomechanics: avg_intercept_y_vs_batter, avg_intercept_y_vs_plate
Handedness: bat_side




### 1. Environment Setup
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/baseball-analytics.git
cd baseball-analytics

# Create virtual environment
python -m venv .venv
source .venv/bin/activate   # Mac/Linux
.venv\Scripts\activate      # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Analysis
```bash
# Quick test with sample data
python main.py

# Full analysis (requires complete dataset)
# First download combined_statcast_data.csv to data/ directory
# Change line 75 in data_processing.py to "combined_statcast_data.csv"
python main.py
```
This executes the full pipeline:

1. Data loading and temporal matching
2. Feature engineering (40+ features)
3. Advanced imputation and preprocessing
4. Stance clustering (5 archetypes)
4. Neural network training with early stopping
6. Model evaluation and performance metrics
7. Strategic analysis and recommendations
8. Visualization generation and results export


```bash
# Advanced Usage 
from src import ImprovedBaseballSystem
from src.analysis import get_player_stance_recommendation

# Initialize system
system = ImprovedBaseballSystem()

# Run full pipeline
system, model, results, data, stance_analysis = main()

# Get player-specific recommendations
recommendation = get_player_stance_recommendation("Olson, Matt", data, stance_analysis)
print(recommendation)

# Access trained model and scaler
predictions = system.model(new_data_tensor)
scaled_features = system.scaler.transform(raw_features)

### 3. Expected Output
```
INFO: Model Performance:
INFO: Test Accuracy: 0.5580
INFO: Weighted F1: 0.5746

INFO: Stance Analysis Results:
 stance_cluster  pitcher_win_rate  avg_foot_separation  avg_stance_angle
              0            0.2727                 26.78            -30.20
              1            0.2787                 39.99             -7.38
              ...
```

## Technical Implementation

### **Machine Learning Architecture**
- **Model Type**: Feedforward Neural Network 
- **Architecture**: 3-layer with batch normalization
- **Input**: 40 engineered features per pitch
- **Output**: 3-class classification (Pitcher Win, Neutral, Hitter Win)
- **Training**: Weighted sampling for class balance, early stopping

### **Feature Engineering Pipeline**
- **Pitch Physics**: Velocity, movement, release point metrics
- **Location Features**: Strike zone analysis, edge detection
- **Stance Biomechanics**: Foot separation, angle, positioning
- **Situational Context**: Count pressure, handedness, pitch sequencing
- **Derived Metrics**: Effective velocity, stance ratios, movement combinations

### **Data Processing**
- **Temporal Matching**: Algorithm matches pitch events to stance measurements using temporal matching
- **Advanced Imputation**: Multi-level fallback strategy (player → pitch-type → global)
- **Robust Validation**: Stratified splits, cross-validation, reproducibility testing

## Business Applications

### **For Professional Teams**
- **Scouting Enhancement**: Classify opponents into 5 stance categories
- **Pitch Selection**: Data-driven recommendations for specific batters
- **Player Development**: Optimize batting stances based on outcome data

### **For Coaches & Players**
- **Stance Analysis**: Identify personal vulnerabilities and strengths
- **Training Focus**: Prioritize practice areas based on stance type
- **Strategic Planning**: Exploit opponent weaknesses in game situations

### **For Analytics Departments**
- **Performance Modeling**: Extend predictions to additional outcomes
- **Real-time Integration**: Deploy decision support tools for dugout staff
- **Competitive Intelligence**: First systematic analysis of stance effectiveness

## Dependencies

### **Core Requirements**
```
pandas>=1.5.0          # Data manipulation and analysis
numpy>=1.21.0          # Numerical computing
scikit-learn>=1.1.0    # Machine learning utilities
torch>=1.12.0          # Neural network framework
scipy>=1.9.0           # Statistical functions
matplotlib>=3.5.0      # Data visualization
seaborn>=0.11.0        # Statistical plotting
```

### **Installation**
```bash
pip install -r requirements.txt
```

## Model Performance Metrics

### **Classification Results**
- **Overall Accuracy**: 55.8% (vs ~33% random chance)
- **Weighted F1 Score**: 57.5%
- **Training Samples**: 571,351 pitch-stance combinations

### **Class-Specific Performance**
| Outcome Type | Precision | Recall | F1-Score |
|--------------|-----------|---------|----------|
| Pitcher Win  | 51%       | 59%     | 55%      |
| Neutral      | 35%       | 58%     | 44%      |
| Hitter Win   | 82%       | 53%     | 64%      |

### **Statistical Validation**
- **Reproducibility**: 98.7% consistency between sample and full dataset
- **Sample Size**: 571K+ training samples (statistically robust)
- **Cross-Validation**: Stratified sampling maintains class balance

## Key Insights & Recommendations

### **Immediate Action Items**
1. **Target Standard Stance hitters** (48.6% of players) - highest pitcher success rate
2. **Develop forkball usage** - most effective against 61% of batting stances
3. **Eliminate knuckleball usage** - consistently worst pitch across stance types
4. **Implement stance classification** in scouting reports

### **Player Development Priorities**
1. **Standard stance users**: Adjust toward balanced stance metrics (+1.1% improvement)
2. **Wide stance players**: Focus on breaking ball recognition training
3. **Closed stance players**: Maintain approach but improve plate coverage

## Future Enhancements

### **Short-term** (3-6 months)
- Real-time stance classification from video feeds
- Integration with existing TrackMan/Statcast systems
- Mobile app for coaches with stance analysis tools

### **Long-term** (6-12 months)
- Expand predictions to swing decisions and contact outcomes
- Develop pitcher fatigue and sequence modeling
- Create proprietary stance modification protocols

## Contributing

Contributions welcome! Areas for improvement:
- Additional feature engineering approaches
- Alternative clustering algorithms for stance classification
- Integration with video analysis systems
- Expansion to other baseball metrics

## Acknowledgments

- **Baseball Savant** for providing comprehensive Statcast data
- **MLB Advanced Media** for stance measurement datasets  
- **PyTorch community** for neural network framework
- **Baseball analytics community** for inspiration and validation

---

**Note**: This project represents a systematic analysis of batting stance impact on pitch outcomes in professional baseball. Results are based on rigorous statistical methodology and provide genuine competitive insights for baseball organizations.

*For questions, suggestions, or collaboration opportunities, please open an issue or contact hemaniprisha1@gmail.com*
