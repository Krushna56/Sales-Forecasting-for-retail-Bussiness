import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import os

# Set correct file path for Windows
file_path = os.path.join(os.path.dirname(__file__), "sales_data_sample.csv")

try:
    # Load dataset with error handling
    df = pd.read_csv(file_path, encoding="ISO-8859-1")
    
    # Verify required columns exist
    required_columns = ["ORDERDATE", "SALES"]
    if not all(col in df.columns for col in required_columns):
        raise KeyError(f"Missing one of required columns: {required_columns}")
    
    # Convert ORDERDATE to datetime format
    df["ORDERDATE"] = pd.to_datetime(df["ORDERDATE"], errors="coerce")
    
    # Remove rows with invalid dates
    df = df.dropna(subset=["ORDERDATE"])
    
    if len(df) == 0:
        raise ValueError("No valid data remaining after cleaning")
    
    # Aggregate sales by date
    daily_sales = df.groupby("ORDERDATE")["SALES"].sum().reset_index()
    
    # Rename columns for Prophet compatibility
    daily_sales.columns = ["ds", "y"]
    
    # Handle missing dates and interpolate
    all_dates = pd.date_range(start=daily_sales["ds"].min(), end=daily_sales["ds"].max())
    daily_sales = daily_sales.set_index("ds").reindex(all_dates).reset_index()
    daily_sales.columns = ["ds", "y"]
    daily_sales["y"] = daily_sales["y"].interpolate()
    
    # Create and fit the Prophet model
    model = Prophet(
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.05,
        yearly_seasonality=True,
        weekly_seasonality=True
    )
    
    model.fit(daily_sales)
    
    # Make predictions
    future = model.make_future_dataframe(periods=90)
    forecast = model.predict(future)
    
    # Create plots
    plt.figure(figsize=(12, 6))
    fig = model.plot(forecast)
    plt.title("Sales Forecast for Retail Business")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.savefig("forecast.png")  # Save plot before showing
    plt.show()
    
    plt.figure(figsize=(12, 8))
    fig2 = model.plot_components(forecast)
    plt.savefig("components.png")  # Save plot before showing
    plt.show()
    
    # Display forecasted values
    print("\nForecasted Values (last 10 days):")
    print(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10))

except FileNotFoundError:
    print(f"Error: Could not find the file {file_path}")
    print("Please ensure the sales_data_sample.csv file is in the same directory as the script")
except KeyError as e:
    print(f"Error: {e}")
    print("Please check your CSV file contains the required columns: ORDERDATE and SALES")
except Exception as e:
    print(f"An error occurred: {e}")

    