import numpy as np

from scipy import special
from scipy import stats


# --------------------------------------------------------------------------------
# continuous distributions
# --------------------------------------------------------------------------------
def dagum(x, a, b, p):
    return ( 1+(x/b)**-a )**-p
