# Seishimi



## Info

Some scripts for analysing seismicity dynamics. Processes in our interest include clustering of events, aftershocks decay, background events activity.

These are very incomplete, and the repo need to be replenished as my experience and knowledge in this the theories and practices of this field grow. :)



## Dependencies

### Python

Any version of python, e.g., Python 3.6+

### emcee (the mcmc hammer)

The code with fitting aftershocks decay required a MCMC python package. Check it out here: https://emcee.readthedocs.io/en/stable/

#### Installation

```bash
conda update conda
conda install -c conda-forge emcee
```

