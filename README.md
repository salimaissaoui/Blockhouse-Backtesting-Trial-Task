
# Blockhouse Backtesting Trial Task

This project implements a backtesting framework for simulating various trading strategies. The framework includes TWAP (Time-Weighted Average Price), VWAP (Volume-Weighted Average Price with dynamic rebalancing), and a conceptual RL (Reinforcement Learning)-based execution strategy. The goal is to validate algorithmic performance and provide execution metrics such as execution cost and slippage.

---

## Features

1. **Synthetic Data Generation**
   - Simulates realistic market data, including prices and volumes, using a random walk model.

2. **Trading Strategies**
   - **TWAP**: Executes trades at fixed intervals over a given time window.
   - **VWAP (Dynamic)**: Allocates order quantities dynamically based on observed and forecasted market volumes.
   - **RL-Based (Conceptual)**: A reinforcement learning-inspired strategy that adjusts execution based on price trends.

3. **Metrics Calculation**
   - Calculates execution cost, slippage, and performance benchmarks (e.g., VWAP).

4. **Visualizations**
   - Plots the execution results for each strategy over time.

---

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Strategies](#strategies)
4. [Metrics](#metrics)
5. [Results](#results)
6. [Future Improvements](#future-improvements)

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/blockhouse-backtesting-trial-task.git
   cd blockhouse-backtesting-trial-task
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # For Linux/Mac
   .\.venv\Scripts\activate   # For Windows
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

1. Run the main script:
   ```bash
   python script.py
   ```

2. Outputs:
   - Prints metrics for each strategy to the console.
   - Generates plots visualizing the executions.

---

## Strategies

### 1. Time-Weighted Average Price (TWAP)
Splits a total order into equal slices and executes them at regular intervals.

### 2. Volume-Weighted Average Price (VWAP)
Dynamically adjusts order quantities based on observed and estimated market volumes.

### 3. Reinforcement Learning-Based Execution (Conceptual)
Uses a simple RL-inspired policy to decide execution actions based on price trends.

---

## Metrics

1. **Overall VWAP**: Benchmark price calculated based on total traded volume.
2. **Average Execution Price**: Weighted average of execution prices.
3. **Execution Cost**: Difference between average execution price and VWAP.
4. **Slippage**: Difference between arrival price and average execution price.

---

## Results

### TWAP Strategy
- **Execution Cost**: `-0.1293`
- **Slippage**: `-4.2379`

### VWAP (Dynamic) Strategy
- **Execution Cost**: `-0.3638`
- **Slippage**: `-5.0162`

### RL-Based Strategy (Conceptual)
- **Execution Cost**: `2.6858`
- **Slippage**: `-0.6647`

The results highlight the efficiency and adaptability of each strategy under simulated market conditions.

---

## Future Improvements

1. **Advanced RL Algorithms**
   - Implement reinforcement learning models like DDPG or PPO for dynamic execution optimization.

2. **Multi-Venue Support**
   - Simulate routing across multiple market venues with varying liquidity and fees.

3. **Improved Market Models**
   - Use historical data or stochastic processes for more realistic market simulations.

4. **Real-Time Simulations**
   - Extend the framework to handle real-time streaming data.

---

## Contributing

Contributions are welcome! If you'd like to improve this project, please fork the repository and create a pull request.

---

## Contact

For any questions or feedback, please reach out to [saiss048@uottawa.ca](mailto:saiss048@uottawa.ca).

