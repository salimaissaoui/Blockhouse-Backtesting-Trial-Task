import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1. Generate Synthetic Market Data
def generate_synthetic_data(num_points=100, initial_price=100.0, volatility=0.02):
    """
    Generate synthetic price data using a simple random walk model.
    """
    np.random.seed(42)
    timestamps = pd.date_range(start="2023-01-01", periods=num_points, freq="T")
    
    prices = [initial_price]
    volumes = []
    
    for i in range(1, num_points):
        price_change = np.random.randn() * volatility
        new_price = prices[-1] * (1 + price_change)
        prices.append(max(1, new_price))  
        volumes.append(np.random.randint(10, 1000))
    
    prices = np.array(prices)
    volumes = np.array([np.random.randint(10, 1000) for _ in range(num_points)])
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "price": prices,
        "volume": volumes
    })
    df.set_index("timestamp", inplace=True)
    return df

# 2. TWAP Strategy
def twap_execution(data, total_quantity, start_time, end_time, freq="5T"):
    """
    Executes a TWAP strategy by slicing orders equally over the time period.
    
    :param data: DataFrame with 'price' column indexed by timestamp
    :param total_quantity: total quantity to be traded
    :param start_time: start of trading (datetime)
    :param end_time: end of trading (datetime)
    :param freq: frequency of order slices, e.g., "5T" for 5 minutes
    :return: A DataFrame of executions with columns [exec_timestamp, exec_price, exec_quantity].
    """
    trade_data = data.loc[start_time:end_time]
    if len(trade_data) == 0:
        raise ValueError("No data in the specified trading window.")
    
    slice_times = pd.date_range(start=start_time, end=end_time, freq=freq)
    num_slices = len(slice_times)
    slice_quantity = total_quantity / num_slices if num_slices > 0 else total_quantity
    
    executions = []
    
    for t in slice_times:
        if t in trade_data.index:
            current_price = trade_data.loc[t, "price"]
        else:
            current_price = trade_data.iloc[(trade_data.index.get_loc(t, method='nearest'))]["price"]
        
        executions.append({
            "exec_timestamp": t,
            "exec_price": current_price,
            "exec_quantity": slice_quantity
        })
        
    exec_df = pd.DataFrame(executions)
    return exec_df

# 3. VWAP Strategy with Dynamic Rebalancing
def vwap_execution_dynamic(data, total_quantity, start_time, end_time, rebalance_freq="5T"):
    """
    Executes a VWAP-like strategy with dynamic rebalancing intervals.
    At each rebalance interval, the strategy:
      - Observes the realized volume so far
      - Estimates the remaining volume for the rest of the window
      - Reallocates the remaining order quantity accordingly
    
    :param data: DataFrame with 'price' and 'volume' indexed by timestamp
    :param total_quantity: total quantity to be traded
    :param start_time: start of trading (datetime)
    :param end_time: end of trading (datetime)
    :param rebalance_freq: frequency at which the strategy recalculates slice sizes
    :return: DataFrame of executions with columns [exec_timestamp, exec_price, exec_quantity].
    """
    trade_data = data.loc[start_time:end_time]
    if len(trade_data) == 0:
        raise ValueError("No data in the specified trading window.")
    
    rebalance_times = pd.date_range(start=start_time, end=end_time, freq=rebalance_freq)
    exec_records = []
    
    remaining_quantity = total_quantity
    previous_time = start_time
    
    for i, current_time in enumerate(rebalance_times):
        if i == 0:
            # no trades yet, just move to next iteration
            continue
        
        # Time window: previous_time -> current_time
        interval_data = trade_data.loc[previous_time:current_time]
        if len(interval_data) == 0:
            continue
        
        # Observed volume in this interval
        observed_interval_volume = interval_data["volume"].sum()
        
        # Remaining time intervals
        intervals_left = len(rebalance_times) - i
        
        # Estimate total future volume by naive assumption:
        # For simplicity, let's do average volume so far:
        if i == 1:
            average_volume_so_far = observed_interval_volume
        else:
            # All intervals up to i
            observed_data_so_far = trade_data.loc[start_time:current_time]
            average_volume_so_far = observed_data_so_far["volume"].sum() / i
        
        estimated_future_volume = average_volume_so_far * intervals_left
        total_estimated_volume = observed_interval_volume + estimated_future_volume
        
        # Fraction of total trades allocated to this interval
        # based on the fraction of interval volume within the total estimated volume
        if total_estimated_volume <= 0:
            continue
        fraction_of_interval = observed_interval_volume / total_estimated_volume
        
        # Assign fraction_of_interval * remaining_quantity to this interval
        quantity_to_execute = remaining_quantity * fraction_of_interval
        
        # For simplicity, assume we execute at the average price within this interval
        avg_price_interval = np.average(interval_data["price"], weights=interval_data["volume"])
        
        # Record a single execution for this interval (or break down further as needed)
        exec_records.append({
            "exec_timestamp": current_time,
            "exec_price": avg_price_interval,
            "exec_quantity": quantity_to_execute
        })
        
        # Update remaining quantity
        remaining_quantity -= quantity_to_execute
        previous_time = current_time
        
        if remaining_quantity <= 0:
            break
    
    # If there's leftover quantity after final rebalance_time, allocate at the final time
    if remaining_quantity > 0:
        last_time = rebalance_times[-1]
        final_data = trade_data.loc[last_time:end_time]
        if not final_data.empty:
            final_price = np.average(final_data["price"], weights=final_data["volume"])
        else:
            final_price = trade_data.iloc[-1]["price"]  # fallback
        exec_records.append({
            "exec_timestamp": end_time,
            "exec_price": final_price,
            "exec_quantity": remaining_quantity
        })
    
    exec_df = pd.DataFrame(exec_records)
    return exec_df

# 4. Reinforcement-Learning-Based Routing (Conceptual)
def rl_smart_routing(data, total_quantity, start_time, end_time):
    """
    A minimal conceptual example of using RL to decide whether to execute or hold at each time step.
    For demonstration only. This is NOT a production RL solution.
    
    :param data: DataFrame with 'price' column indexed by timestamp
    :param total_quantity: total quantity to be traded
    :param start_time: start of trading (datetime)
    :param end_time: end of trading (datetime)
    :return: A DataFrame with RL-based execution records.
    """
    trade_data = data.loc[start_time:end_time]
    if len(trade_data) == 0:
        raise ValueError("No data in the specified trading window.")
    
    timestamps = trade_data.index
    exec_records = []
    remaining_quantity = total_quantity
    
    # Minimal RL placeholders:
    # state = [time_step, remaining_quantity, price_trend, etc.]
    # action = [0: do nothing, 1: execute a partial/ full slice, ...]
    
    # For demonstration, let's assume a naive 'policy' that tries to learn to buy only when
    # the price is below some running average.
    running_avg_price = trade_data["price"].expanding().mean()
    
    for i, t in enumerate(timestamps):
        if remaining_quantity <= 0:
            break
        
        current_price = trade_data.loc[t, "price"]
        avg_price_so_far = running_avg_price.iloc[i]
        
        # RL 'policy': if current_price < avg_price_so_far => buy a slice
        # else do nothing
        # In real RL: we'd have Q-values or a neural net to decide action
        if current_price < avg_price_so_far:
            # Decide how much to buy; here we do a fixed fraction
            slice_quantity = remaining_quantity * 0.1  # 10% of what's left
            exec_records.append({
                "exec_timestamp": t,
                "exec_price": current_price,
                "exec_quantity": slice_quantity
            })
            remaining_quantity -= slice_quantity
    
    # If anything remains at end_time, do a final fill
    if remaining_quantity > 0:
        final_price = trade_data.iloc[-1]["price"]
        exec_records.append({
            "exec_timestamp": end_time,
            "exec_price": final_price,
            "exec_quantity": remaining_quantity
        })
        remaining_quantity = 0
    
    exec_df = pd.DataFrame(exec_records)
    return exec_df

# 5. Calculate Metrics (Execution Cost, Slippage)
def calculate_metrics(exec_df, market_data, benchmark="VWAP"):
    """
    Calculate various performance metrics:
      - Execution Cost relative to a benchmark (e.g., VWAP)
      - Slippage: difference between expected (arrival) price and actual fill prices
    """
    if len(exec_df) == 0:
        print("No executions found. Cannot calculate metrics.")
        return {}
    
    if benchmark == "VWAP":
        total_volume = (market_data["price"] * market_data["volume"]).sum()
        total_shares = market_data["volume"].sum()
        if total_shares == 0:
            overall_vwap = market_data["price"].mean()
        else:
            overall_vwap = total_volume / total_shares
        
        avg_exec_price = np.average(exec_df["exec_price"], weights=exec_df["exec_quantity"])
        execution_cost = avg_exec_price - overall_vwap
        
        # Slippage: difference between the arrival price (price at first execution) and actual execution
        expected_price = exec_df.loc[0, "exec_price"] if not exec_df.empty else overall_vwap
        slippage = avg_exec_price - expected_price
        
        return {
            "Benchmark": benchmark,
            "Overall_VWAP": overall_vwap,
            "Avg_Exec_Price": avg_exec_price,
            "Execution_Cost": execution_cost,
            "Slippage": slippage
        }
    else:
        # Implement other benchmarks as needed
        return {}

# 6. Main Simulation
if __name__ == "__main__":
    # Generate synthetic data
    df_market = generate_synthetic_data(num_points=200, initial_price=100, volatility=0.01)
    
    # Parameters
    total_quantity = 1000
    start_time = df_market.index[0]
    end_time = df_market.index[-1]
    
    # A. Basic TWAP Execution
    exec_twap = twap_execution(df_market, total_quantity, start_time, end_time, freq="5T")
    metrics_twap = calculate_metrics(exec_twap, df_market, benchmark="VWAP")
    print("=== TWAP Strategy Results ===")
    for k, v in metrics_twap.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    
    # B. VWAP with Dynamic Rebalancing
    exec_vwap_dynamic = vwap_execution_dynamic(df_market, total_quantity, start_time, end_time, rebalance_freq="10T")
    metrics_vwap_dynamic = calculate_metrics(exec_vwap_dynamic, df_market, benchmark="VWAP")
    print("\n=== VWAP (Dynamic) Strategy Results ===")
    for k, v in metrics_vwap_dynamic.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    
    # C. Reinforcement-Learning-Based Routing (Conceptual Example)
    exec_rl = rl_smart_routing(df_market, total_quantity, start_time, end_time)
    metrics_rl = calculate_metrics(exec_rl, df_market, benchmark="VWAP")
    print("\n=== RL-Based Strategy Results (Conceptual) ===")
    for k, v in metrics_rl.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    
    # Plot results for visual inspection
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # Plot: TWAP
    axes[0].plot(df_market.index, df_market["price"], label="Price")
    axes[0].scatter(exec_twap["exec_timestamp"], exec_twap["exec_price"], color="red", marker="x", label="TWAP Exec")
    axes[0].set_title("TWAP Executions")
    axes[0].legend()
    
    # Plot: VWAP Dynamic
    axes[1].plot(df_market.index, df_market["price"], label="Price")
    axes[1].scatter(exec_vwap_dynamic["exec_timestamp"], exec_vwap_dynamic["exec_price"], color="purple", marker="o", label="VWAP (Dynamic) Exec")
    axes[1].set_title("VWAP (Dynamic) Executions")
    axes[1].legend()
    
    # Plot: RL-based
    axes[2].plot(df_market.index, df_market["price"], label="Price")
    axes[2].scatter(exec_rl["exec_timestamp"], exec_rl["exec_price"], color="green", marker="D", label="RL Exec")
    axes[2].set_title("RL-Based Executions (Conceptual)")
    axes[2].legend()
    
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.show()
