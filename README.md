# Lat-PFN: A Joint Embedding Predictive Architecture for In-context Time-series Forecasting

The official repository of LatentTimePFN (LaT-PFN) [available as preprint on Arxiv](https://arxiv.org/abs/2405.10093)

This work introduces LaT-PFN, a novel time series model that combines PFN and JEPA frameworks to generate zero-shot forecasts efficiently, using a versatile latent space that enables adaptable time granularity and superior predictive performance.


![forecast fits of the LaT-PFN model](https://wair.ai/wp-content/uploads/2024/06/forecast-paper.png)

See also [our](https://wair.ai/forecasting-of-product-sales-patterns/) blogpost with some practical examples

## Abstract

```text
We introduce LatentTimePFN (LaT-PFN), a foundational Time Series model with a strong embedding
space that enables zero-shot forecasting. To achieve this, we perform in-context learning in
latent space utilizing a novel integration of the Prior-data Fitted Networks (PFN) and Joint
Embedding Predictive Architecture (JEPA) frameworks. We leverage the JEPA framework to create
a prediction-optimized latent representation of the underlying stochastic process that generates
time series and combines it with contextual learning, using a PFN. Furthermore, we improve on
preceding works by utilizing related time series as a context and introducing an normalized
abstract time axis. This reduces training time and increases the versatility of the model by
allowing any time granularity and forecast horizon. We show that this results in superior
zero-shot predictions compared to established baselines. We also demonstrate our latent space
produces informative embeddings of both individual time steps and fixed-length summaries of
entire series. Finally, we observe the emergence of multi-step patch embeddings without explicit
training, suggesting the model actively learns discrete tokens that encode local structures
in the data, analogous to vision transformers.
```


## Installation

This repository uses Python 3.10. Please ensure you have the correct version installed.

To install the required packages, first create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Then install the requirements file:

```bash
pip install -r requirements.txt
```
We have provided a copy of our real datasets for evaluation. The zip file is named:
`datasets.tar.gz`

After unzipping it, please ensure you place the resulting datasets folder in the root of the repository.

```bash
tar -xvf datasets.tar.gz
```

> NOTE: This model requires a machine with GPU for training.

### Trained Weights

Download trained model weights [here](https://drive.google.com/drive/folders/11dC1tbj0Vafr1Iqnk-IMBOpkI3-imYwL?usp=drive_link) 

### Train

We have trained LaT-PFN on a single NVIDIA A10G Tensor Core GPU, for 24 hours.

To train the model, run:

```bash
python train.py
```
The `config.py` file contains the hyperparameters used for training. Note some seeds can have different effects on the final model stability.

### Eval

To evaluate the model, run:

```bash
python -m evaluation.run_evals
```
The details of the contexts and held-out series are in the `evaluation/real_evals.py` file. 

Please note that these contexts have been curated for optimal performance, changing them may result in suboptimal performance.
Similarly the normalization functions used to map both the time and value axis to their respective normalized spaces have also been selected for optimal performance. Once again, changing these may result in suboptimal performance.

### Your Inference Script

Our model implements two functions for inference:
1. `create_forecast`
2. `create_embeddings`

See examples in `example.ipynb` for how to use these functions

To maximise the forecasting performance of the model, please consider dedicating some time to curating your contexts and normalisation functions.

### Tune

To tune the model, run:

```bash
python tune.py
```
To change the parameters to tune, please modify the `objective` function in `tune.py`.

Multiple tuning process can work together by all referencing the `STUDY_NAME`

# References

Please include to the following reference when using or building upon this work:

```bibtex
    @article{verdenius2024lat,
        title={LaT-PFN: A Joint Embedding Predictive Architecture for In-context Time-series Forecasting},
        author={Verdenius, Stijn and Zerio, Andrea and Wang, Roy LM},
        journal={arXiv preprint arXiv:2405.10093},
        year={2024}
    }
```

# WAIR Forecasting

See the website of [WAIR AI](https://wair.ai/) for automated retail forecasting solutions
