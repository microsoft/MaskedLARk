This folder is intended to be a demonstration of how to run training using the MaskedLARk protocol. It depends on the MaskedLark package, located in ../MaskedLARkRequests.

The data used is the [Wisconsin Breast Cancer Dataset](https://archive-beta.ics.uci.edu/ml/datasets/breast+cancer+wisconsin+diagnostic), which we automatically download using sklearn.

## Running the Demo

Once at least two helper services are running, either remotely or locally, run the training script in /training/sklearn_trainer_mlark.py .

This script requires the helper names (for proper identification) and the endpoint locations of the helpers.

### Example Command

```buildoutcfg
python training/sklearn_trainer_mlark.py --helpernames helpername0 helpername1 --helperendpoints http://localhost:7071/api/GradientHelper http://localhost:7072/api/GradientHelper
```


### Dependencies

