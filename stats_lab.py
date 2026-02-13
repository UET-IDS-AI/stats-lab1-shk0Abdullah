import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------
# Question 1 – Generate & Plot Histograms (and return data)
# -----------------------------------

# 1. Normal(0,1)
def normal_histogram(n):
    data = np.random.normal(0, 1, n)
    
    plt.figure()
    plt.hist(data, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Normal(0,1) Distribution")
    plt.show()
    
    return data


# 2. Uniform(0,10)
def uniform_histogram(n):
    data = np.random.uniform(0, 10, n)
    
    plt.figure()
    plt.hist(data, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Uniform(0,10) Distribution")
    plt.show()
    
    return data


# 3. Bernoulli(0.5)
def bernoulli_histogram(n):
    data = np.random.binomial(1, 0.5, n)
    
    plt.figure()
    plt.hist(data, bins=10)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Bernoulli(0.5) Distribution")
    plt.show()
    
    return data
# -----------------------------------
# Question 2 – Sample Mean & Variance
# -----------------------------------

def sample_mean(data):
    """
    Compute sample mean.
    """
    data = np.array(data)
    return np.sum(data) / len(data)


def sample_variance(data):
    """
    Compute sample variance using n-1 denominator.
    """
    data = np.array(data)
    n = len(data)
    mean = sample_mean(data)
    return np.sum((data - mean) ** 2) / (n - 1)


# -----------------------------------
# Question 3 – Order Statistics
# -----------------------------------

def order_statistics(data):
    """
    Return:
    - min
    - max
    - median
    - 25th percentile (Q1)
    - 75th percentile (Q3)

    Uses Tukey's hinges (median included in both halves when n is odd).
    """
    data = sorted(data)
    n = len(data)
    
    # Min & Max
    minimum = data[0]
    maximum = data[-1]
    
    # Median
    if n % 2 == 1:
        median = data[n // 2]
        lower_half = data[: n // 2 + 1]   # include median
        upper_half = data[n // 2 :]       # include median
    else:
        median = (data[n // 2 - 1] + data[n // 2]) / 2
        lower_half = data[: n // 2]
        upper_half = data[n // 2 :]
    
    # Q1 (median of lower half)
    m = len(lower_half)
    if m % 2 == 1:
        q1 = lower_half[m // 2]
    else:
        q1 = (lower_half[m // 2 - 1] + lower_half[m // 2]) / 2
    
    # Q3 (median of upper half)
    m = len(upper_half)
    if m % 2 == 1:
        q3 = upper_half[m // 2]
    else:
        q3 = (upper_half[m // 2 - 1] + upper_half[m // 2]) / 2
    
    return (minimum, maximum, median, q1, q3)
# -----------------------------------
# Question 4 – Sample Covariance
# -----------------------------------

def sample_covariance(x, y):
    """
    Compute sample covariance using n-1 denominator.
    """
    x = np.array(x)
    y = np.array(y)
    
    if len(x) != len(y):
        raise ValueError("x and y must have the same length")
    
    n = len(x)
    mean_x = np.sum(x) / n
    mean_y = np.sum(y) / n
    
    covariance = np.sum((x - mean_x) * (y - mean_y)) / (n - 1)
    
    return covariance

# -----------------------------------
# Question 5 – Covariance Matrix
# -----------------------------------
def covariance_matrix(x, y):
    """
    Return 2x2 covariance matrix:
        [[var(x), cov(x,y)],
         [cov(x,y), var(y)]]
    """
    var_x = sample_variance(x)
    var_y = sample_variance(y)
    cov_xy = sample_covariance(x, y)
    
    return np.array([
        [var_x, cov_xy],
        [cov_xy, var_y]
    ])
