import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Fetch historical gold price data
def fetch_gold_data():
    gold_data = yf.download("GC=F", start="2010-01-01", end=pd.Timestamp.now().strftime('%Y-%m-%d'))
    
    if gold_data.empty:
        st.error("Failed to retrieve gold data.")
        return None
    
    if 'Adj Close' in gold_data.columns:
        df = gold_data.reset_index()[["Date", "Adj Close"]]
    else:
        df = gold_data.reset_index()[["Date", "Close"]]

    df.rename(columns={"Close": "GoldPrice", "Adj Close": "GoldPrice"}, inplace=True)
    return df

# Preprocessing function
def preprocess_data(df, window_size=10):
    df["Date"] = pd.to_datetime(df["Date"])
    df["OrdinalDate"] = df["Date"].map(pd.Timestamp.toordinal)
    
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[["OrdinalDate", "GoldPrice"]])

    X, y = [], []
    for i in range(len(df_scaled) - window_size):
        X.append(df_scaled[i:i+window_size, 1])  # Use GoldPrice only
        y.append(df_scaled[i+window_size, 1])  # Predict next day's price
    return np.array(X), np.array(y), scaler

# Define the LSTM model
class GoldPriceLSTM(nn.Module):
    def __init__(self):
        super(GoldPriceLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=3, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Take last output
        return x

# Train the model
def train_model(X_train, y_train, model, epochs=100, batch_size=64):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    dataset = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1),
                                             torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for X_batch, y_batch in dataloader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

# Predict function
def predict_gold_rate(model, scaler, df, date, window_size=10):
    date_ordinal = pd.Timestamp(date).toordinal()
    
    # Prepare last sequence for prediction
    recent_data = df.iloc[-window_size:][["GoldPrice"]].values
    recent_data_scaled = scaler.transform(np.hstack((df.iloc[-window_size:][["OrdinalDate"]].values, recent_data)))[:, 1]
    input_tensor = torch.tensor(recent_data_scaled.reshape(1, window_size, 1), dtype=torch.float32)

    # Predict
    predicted_scaled = model(input_tensor).item()

    # Fix for inverse_transform shape
    predicted_original = np.array([[date_ordinal, predicted_scaled]])
    predicted_price = scaler.inverse_transform(predicted_original)[:, 1][0]

    return predicted_price

# Streamlit UI
st.title("Gold Price Prediction using LSTM")
st.write("Enter a date to predict the gold price and view past 5 days' trend.")

df = fetch_gold_data()
if df is not None:
    X, y, scaler = preprocess_data(df)
    
    if len(X) > 1:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = GoldPriceLSTM()
        train_model(X_train, y_train, model)

        # Calculate Model Accuracy
        y_pred_test = model(torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)).detach().numpy().flatten()
        y_test_original = scaler.inverse_transform(np.column_stack((np.zeros(len(y_test)), y_test)))[:, 1]
        y_pred_original = scaler.inverse_transform(np.column_stack((np.zeros(len(y_pred_test)), y_pred_test)))[:, 1]

        mae = mean_absolute_error(y_test_original, y_pred_original)
        r2 = r2_score(y_test_original, y_pred_original)
        
        st.write(f"**Model Accuracy:**")
        st.write(f"📉 Mean Absolute Error (MAE): **{mae:.2f}**")
        st.write(f"📊 R-Squared Score: **{r2:.4f}**")

        # Date input
        date_to_predict = st.text_input("Enter a date (YYYY-MM-DD):")

        if date_to_predict:
            try:
                predicted_price = predict_gold_rate(model, scaler, df, date_to_predict)
                st.success(f"Predicted Gold Price on {date_to_predict}: **${predicted_price:.2f}**")

                # Show past 5 days' gold prices
                entered_date = pd.Timestamp(date_to_predict).toordinal()
                past_5_days = df[df["OrdinalDate"] < entered_date].tail(5)

                if not past_5_days.empty:
                    st.write("📌 Past 5 days' gold prices:")
                    st.dataframe(past_5_days[["Date", "GoldPrice"]])

                    # Plot the trend
                    plt.figure(figsize=(8, 5))
                    plt.plot(past_5_days["Date"], past_5_days["GoldPrice"], marker="o", linestyle="-", label="Past 5 Days")
                    plt.axhline(y=predicted_price, color="r", linestyle="--", label="Predicted Price")
                    plt.xlabel("Date")
                    plt.ylabel("Gold Price (USD)")
                    plt.title("📈 Gold Price Trend")
                    plt.xticks(rotation=45)
                    plt.legend()
                    st.pyplot(plt)

            except Exception as e:
                st.error(f"❌ Error in prediction: {e}")
