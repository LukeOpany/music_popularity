# Music Popularity Prediction

## Summary

This project explores whether Spotify-style audio features can predict a song's popularity score. The analysis is built in a Jupyter notebook and focuses on clean preprocessing, fair model comparison, and honest evaluation against a baseline.

## Project Goal / Why This Project Exists

The goal of this project is to test how much signal exists in audio and basic track metadata when predicting music popularity. Popularity is influenced by many external factors such as artist reputation, playlist placement, marketing, region, and social trends, so this project treats the results as exploratory rather than fully conclusive.

## Tech Stack

- Python
- Jupyter Notebook
- pandas
- NumPy
- scikit-learn
- Matplotlib
- Seaborn

## Architecture / Workflow

1. Load the music popularity dataset from `music_popularity.csv`.
2. Clean the dataset by removing duplicate tracks, dropping unnecessary columns, parsing release dates, and repairing text encoding issues.
3. Create derived features such as release year, track age, duration in minutes, and explicit-content flags.
4. Compare core audio features against an expanded feature set.
5. Evaluate baseline and regression models using 5-fold cross-validation.
6. Tune a Random Forest model with a modest hyperparameter search.
7. Compare the best model against the baseline and summarize whether the result is meaningful.

## Features / Analysis

- Removes duplicate tracks using `Track ID`.
- Cleans mojibake text issues in track, artist, and album names.
- Uses `DummyRegressor` as a mean baseline.
- Compares Linear Regression, Ridge Regression, Random Forest, Gradient Boosting, and tuned Random Forest models.
- Reports cross-validated `R2`, `MAE`, and `RMSE` metrics.
- Includes an actual-vs-predicted plot for the best model.
- Concludes that the available CSV features provide some predictive signal, but the outcome remains exploratory due to dataset size and missing external popularity drivers.

## Installation & Usage

Clone the repository and open the project folder:

```bash
git clone https://github.com/LukeOpany/music_popularity.git
cd music_popularity
```

Create and activate a virtual environment:

```bash
python3 -m venv env
source env/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Open the notebook:

```bash
jupyter notebook music.ipynb
```

In VS Code, open `music.ipynb`, select the project virtual environment as the kernel, and run all cells from top to bottom.

## Future Improvements

- Collect more data to improve model reliability.
- Add external features such as artist popularity, playlist count, genre, region, and social media trends.
- Convert the notebook workflow into a reusable Python pipeline.
- Add automated tests for data cleaning and feature engineering.
- Experiment with additional models and feature selection techniques.
