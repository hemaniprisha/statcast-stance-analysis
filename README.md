# Batting Stance & Pitch Outcome Analysis

This project analyzes how batting stances interact with pitch data using machine learning models. It uses Statcast pitch data and custom batting stance features to train and evaluate predictive models.

## Project Structure
- `src/your_notebook.ipynb` — main Jupyter notebook with code & analysis
- `requirements.txt` — Python dependencies
- `data/` — (optional) CSV datasets

## Data

- **batting-stance.csv** (~343 KB): contains batting stance info for all players.
- **sample_statcast.csv** (~8.8 MB): a small sample of Statcast pitch-level data for testing and running notebooks quickly.
- **combined_statcast_data.csv** (~422 MB, not included): full Statcast dataset. You can download it from [Baseball Savant Statcast Search](https://baseballsavant.mlb.com/statcast_search) and save it as `data/combined_statcast_data.csv` if you want to run the full analysis.

> By default, notebooks use `sample_statcast.csv` for faster testing. Swap in `combined_statcast_data.csv` for full analyses.

## Setup

1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
   cd YOUR_REPO
2. Create and activate a virtual environment (recommended):
    python -m venv .venv
    source .venv/bin/activate   # Mac/Linux
    .venv\Scripts\activate      # Windows
3. Install dependencies: 
    pip install -r requirements.txt
4. Launch Jupyter: 
    jupyter notebook
5. Open VS Code or Jupyter Lab and run src/batting_stance_vs_pitch_nn.ipynb


## Dependencies:
pandas
numpy
scikit-learn
torch
scipy
jupyter
Install via:
pip install -r requirements.txt

## Notes
The full dataset is not stored in this repo (too large + license).
Use the sample dataset for quick tests, or regenerate your own Statcast dataset for full-scale analysis.
Contributions and improvements welcome!