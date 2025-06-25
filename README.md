# StatAIT

An empirical privacy test for attribute inference based on statistical inference

This is a preliminary repository. The complete code will be published in a future update. 
# Installation 

Before installing the package, make sure you have the dependencies listed in `requirements.txt` installed. 

To install, run the following command 
```
pip install .
```

# Usage
The real train and test data and the synthetic data need to convert to a binary matrix containing the records attributes as rows before usage. 
Then, the binary attribute matrices should saved as `.npy` files in the `data_dir` directory. 

To run StatAIT, use the command
```
python -m stat_ait.run \
    --data_dir <data_dir> \
    --output_dir <output_dir> \
    --seed <seed> \
    > "stat_ait.log"
```
This will create a `.csv` file containing the pandas data frame with the privacy evaluation results. 

In case you also want to run the standard empirical privacy tests, use the command
```
python -m stat_ait.run_standard \
    --data_dir <data_dir> \
    --output_dir <output_dir> \
    --num_known <num_known> \
    --seed 12 
```
The argument `num_known` sets the number of known parameter for the standard attribute inference tests.

# Licence 
StatAIT is licensed under the MIT License. See LICENSE for more information.