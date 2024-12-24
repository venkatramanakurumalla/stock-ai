import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from tkinter import *
from tkinter import ttk, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from datetime import datetime

class StockPredictor:
    def fetch_data(self, ticker):
        """
        Fetch historical stock data from Yahoo Finance.
        """
        data = yf.download(ticker, start="2020-01-01", end=datetime.now().strftime("%Y-%m-%d"))
        if data.empty:
            raise ValueError("No data available for the given ticker.")
        return data

    def compute_indicators(self, df):
        """
        Compute RSI, SMA, EMA, MACD, and Bollinger Bands.
        """
        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Simple Moving Average (SMA)
        df['SMA'] = df['Close'].rolling(window=14).mean()

        # Exponential Moving Average (EMA)
        df['EMA'] = df['Close'].ewm(span=14, adjust=False).mean()

        # MACD
        df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()

        # Bollinger Bands
        df['BB_Upper'] = df['Close'].rolling(window=20).mean() + 2 * df['Close'].rolling(window=20).std()
        df['BB_Lower'] = df['Close'].rolling(window=20).mean() - 2 * df['Close'].rolling(window=20).std()

        df.fillna(0, inplace=True)
        return df

    def predict(self, df):
        """
        Train a Random Forest model and generate buy/sell recommendation.
        """
        features = ['RSI', 'SMA', 'EMA', 'MACD', 'BB_Upper', 'BB_Lower']
        df = self.compute_indicators(df)

        # Prepare feature matrix and target variable
        X = df[features].values
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)  # 1 for price increase, 0 for decrease
        y = df['Target'].fillna(0).values

        # Train Random Forest model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X[:-1], y[:-1])  # Exclude last row

        # Predict recommendation for the latest row
        latest_features = X[-1].reshape(1, -1)
        prediction = model.predict(latest_features)[0]
        probability = model.predict_proba(latest_features)[0]

        recommendation = "BUY" if prediction == 1 else "SELL"
        confidence = max(probability) * 100
        return recommendation, confidence

def plot_indicators(df):
    """
    Plot stock indicators and price chart.
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Price and Moving Averages
    axs[0].plot(df['Close'], label='Close Price', color='blue')
    axs[0].plot(df['SMA'], label='SMA (14)', color='orange', linestyle='--')
    axs[0].plot(df['EMA'], label='EMA (14)', color='green', linestyle='--')
    axs[0].fill_between(df.index, df['BB_Upper'], df['BB_Lower'], color='grey', alpha=0.3, label='Bollinger Bands')
    axs[0].set_title('Price and Indicators')
    axs[0].legend()

    # RSI and MACD
    axs[1].plot(df['RSI'], label='RSI', color='purple')
    axs[1].axhline(70, color='red', linestyle='--', linewidth=0.7, label='Overbought')
    axs[1].axhline(30, color='green', linestyle='--', linewidth=0.7, label='Oversold')
    axs[1].plot(df['MACD'], label='MACD', color='brown')
    axs[1].set_title('RSI and MACD')
    axs[1].legend()

    fig.tight_layout()
    return fig

def on_predict():
    """
    Handle Predict button click.
    """
    ticker = ticker_entry.get().strip()
    if not ticker:
        messagebox.showwarning("Warning", "Please enter a ticker symbol!")
        return

    predictor = StockPredictor()
    try:
        df = predictor.fetch_data(ticker)
        df = predictor.compute_indicators(df)
        recommendation, confidence = predictor.predict(df)

        # Update recommendation label
        result_label.config(text=f"Recommendation: {recommendation}\nConfidence: {confidence:.2f}%")

        # Display charts
        fig = plot_indicators(df)
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=True)
        canvas.draw()
    except Exception as e:
        messagebox.showerror("Error", str(e))

# GUI Setup
root = Tk()
root.title("Stock Prediction Tool")

# UI Layout
frame = Frame(root, padx=10, pady=10)
frame.pack(side=LEFT, fill=BOTH, expand=True)

Label(frame, text="Stock Prediction Tool", font=("Arial", 16, "bold")).grid(row=0, column=0, columnspan=2, pady=10)

Label(frame, text="Enter Ticker Symbol:").grid(row=1, column=0, sticky=W, padx=5, pady=5)
ticker_entry = Entry(frame, width=20)
ticker_entry.grid(row=1, column=1, padx=5, pady=5)

predict_button = Button(frame, text="Predict", command=on_predict)
predict_button.grid(row=2, column=0, columnspan=2, pady=10)

result_label = Label(frame, text="", font=("Arial", 12), fg="blue")
result_label.grid(row=3, column=0, columnspan=2, pady=10)

# Chart Frame
chart_frame = Frame(root, padx=10, pady=10, bg="white", relief=RIDGE)
chart_frame.pack(side=RIGHT, fill=BOTH, expand=True)

# Run the GUI
root.mainloop()
