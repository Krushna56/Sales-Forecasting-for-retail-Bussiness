# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

# Load dataset
file_path = "sales_data_sample.csv"  # Change this to your file path
df = pd.read_csv(file_path, encoding="ISO-8859-1")

# Convert ORDERDATE to datetime format
df["ORDERDATE"] = pd.to_datetime(df["ORDERDATE"], errors="coerce")

# Drop rows with invalid dates
df = df.dropna(subset=["ORDERDATE"])

# Aggregate sales by date
daily_sales = df.groupby("ORDERDATE")["SALES"].sum().reset_index()

# Rename columns for Prophet compatibility
daily_sales.columns = ["ds", "y"]  # Prophet requires "ds" (date) and "y" (value)

# Handle missing dates by filling with 0 sales
all_dates = pd.date_range(start=daily_sales["ds"].min(), end=daily_sales["ds"].max())
daily_sales = daily_sales.set_index("ds").reindex(all_dates, fill_value=0).reset_index()
daily_sales.columns = ["ds", "y"]

# Train the Prophet model
model = Prophet()
model.fit(daily_sales)

# Create future dataframe for predictions (next 90 days)
future = model.make_future_dataframe(periods=90)
forecast = model.predict(future)

# Plot the forecast
fig = model.plot(forecast)
plt.title("Sales Forecast for Retail Business")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.show()

# Plot forecast components (trend, seasonality)
fig2 = model.plot_components(forecast)
plt.show()

# Display forecasted values
print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10))