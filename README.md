# Loss of Plasticity (LoP) Experiment

PyTorch implementation for investigating loss of plasticity in continual learning with Gaussian Mixture Model (GMM) tasks.

## Overview

This experiment trains a multi-layer perceptron (MLP) on a sequence of binary classification tasks to study how neural networks lose plasticity over time. Each task is generated from a GMM distribution with configurable parameters.

## Key Metrics

- **Cross-Entropy Loss**: Min and final loss per task
- **Saturation**: Fraction of pre-activations with |z| > 2
- **Gradient Norms**: Hidden layer gradient magnitudes
- **Effective Rank**: Dimensionality of layer representations
- **Feature Duplication**: Correlation-based redundancy in activations

## Installation

```bash
pip install torch numpy matplotlib pandas
```

## Usage

Basic run with default parameters:
```bash
python lop_fixed_regime.py
```

Custom configuration:
```bash
python lop_fixed_regime.py --tasks 200 --h 100 --L 5 --lr 0.6 --steps 40
```

**Example configuration that induces loss of plasticity:**
```bash
python lop_fixed_regime.py --steps 500 --lr 0.3 --tasks 300 --delta 0.0 --spread 1. --sigma 0.2 --k 4 --d 2 --L 10 --fresh_steps 500
```
*Note: This configuration demonstrates clear loss of plasticity behavior with high task count, deep network (10 layers), and overlapping task distributions (delta=0.0).*

## Key Parameters

- `--tasks`: Number of sequential tasks (default: 100)
- `--h`: Hidden layer dimension (default: 100)
- `--L`: Number of hidden layers (default: 5)
- `--lr`: Learning rate (default: 0.6)
- `--steps`: Training steps per task (default: 40)
- `--delta`: Task cluster separation (default: 2.8)
- `--fresh_baseline`: Fresh network comparison tasks (default: 20)

## Output

- **CSV**: `lop_fixed_regime_results.csv` - Per-task metrics
- **Plots**:
  - `lop_minCE.png` - Minimum cross-entropy
  - `lop_finalCE.png` - Final cross-entropy
  - `lop_sat.png` - Saturation over time
  - `lop_grad.png` - Gradient norms
  - `lop_effrank.png` - Effective rank by layer
  - `lop_dup_frac.png` - Feature duplication by layer

## Architecture

- **Input**: d-dimensional (default: 4)
- **Hidden**: L layers of h units with tanh activation (default: 5 layers, 100 units)
- **Output**: Single sigmoid unit for binary classification
- **Optimizer**: SGD with fixed learning rate
- **Initialization**: Xavier (Glorot) normal
