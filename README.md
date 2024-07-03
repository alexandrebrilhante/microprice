# microprice

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/microprice)
![GitHub](https://img.shields.io/github/license/alexandrebrilhante/quantnet)

A Jax implementation implementation of [Micro-Price: A High Frequency Estimator of Future Prices](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694).

## Installation
```bash
pip install microprice
```

## Usage
```python
from microprice import MicroPriceEstimator, load_sample_data

mpe = MicroPriceEstimator(symbol="AAPL",
                          n_imbalances=10,
                          n_spread=2,
                          dt=1)

T = mpe.load(load_sample_data())

mpe.fit(T, iterations(6))
```