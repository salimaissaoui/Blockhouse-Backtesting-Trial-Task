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
