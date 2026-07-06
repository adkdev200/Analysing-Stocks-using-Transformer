# Stock Prediction using Transformers

This project leverages deep learning, specifically a custom Transformer model built with PyTorch, to perform stock price prediction. The repository focuses on exploring and utilizing time-series forecasting techniques for stock market data, including a specific case study on Apple Inc. (AAPL).

## Features

- **Custom Transformer Architecture**: Implements a modular Transformer Encoder in PyTorch (`transformer.py`) tailored for time-series/stock data, featuring:
  - Custom Feature Embeddings
  - Positional Encoding
  - Multi-Head Attention Blocks
  - Feed-Forward Blocks with Residual Connections and Layer Normalization
- **Data Gathering**: Utilizes `yfinance` to directly download and update historical stock data.
- **Model Training & Evaluation**: Jupyter notebooks (`main.ipynb`, `apple.ipynb`) provide an end-to-end pipeline:
  - Data preprocessing using `scikit-learn` and `pandas`.
  - Batching sequences using PyTorch DataLoaders.
  - Training loop with metrics visualization (`matplotlib` and `tqdm` for progress tracking).
  - Model evaluation with MSE, MAE, R-squared, and confusion matrices.
- **Visualizations**: Plots the loss curves, stock predictions versus actual prices, and classification confusion matrices.

## Getting Started

### Prerequisites

You need Python 3.8 or higher. The required packages are listed in `requirements.txt`.

### Installation

1. Clone the repository and navigate to the project directory:
   ```bash
   git clone https://github.com/adkdev200-ops/Analysing-Stocks-using-Transformer.git
   cd Analysing-Stocks-using-Transformer
   ```

2. (Optional but recommended) Create and activate a virtual environment:
   ```bash
   python -m venv myenv
   source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

You can run the prediction models interactively through the provided Jupyter notebooks:

- **`main.ipynb`**: General exploratory data analysis, transformer testing, and broad stock prediction logic.
- **`apple.ipynb`**: Focused prediction specifically on Apple Inc. stock data.

To start Jupyter, run:
```bash
jupyter notebook
```
Then open `apple.ipynb` or `main.ipynb` in your browser.

## Project Structure

- `transformer.py`: Contains the PyTorch implementation of the Transformer components.
- `main.ipynb`: Core notebook for training and evaluating models.
- `apple.ipynb`: Case study notebook specifically forecasting Apple's stock prices.
- `requirements.txt`: Python package dependencies.
- `.gitignore`: Files and directories to be ignored by git.
