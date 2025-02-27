# Kernel Density Estimation
In statistics, kernel density estimation is application of `kernel smoothing` for `probility density estimation`. It is also termed as `Parzen–Rosenblatt window` . One of its famous application is its usage in a `Naive Bayes Classifier`.

To build a distribution, instead of putting price changes into small bins like a histogram or building a normal distribution, where you would need to assume data is normal. Kernel density estimation is a more nuanced approach.

In the finance world, it is essential for density estimation of returns for a specific asset, portfolio and trading strategy. The market returns are not normal. There are multiple `black swans` but I don't see the opposite way occurred.

## Definition
$\hat{f}_h(x) = \frac{1}{n} \sum_{i=1}^{n} K_h(x - x_i) = \frac{1}{nh} \sum_{i=1}^{n} K\left(\frac{x - x_i}{h}\right)$

Kernel Density Estimation works by placing a K (kernel, a smooth, non-negative, continuous function) (on your choice) at each data point. Then, it sums up these kernel with a subscript h (called bandwidth). It forms a smooth curve that represents the `overall probability density function of that sample data`.

Kernel Density Estimation: Parameter 1 `Kernel` Function:
A range of kernel functions are commonly used: uniform, triangular, biweight, triweight, Epanechnikov (parabolic), normal, and others. The Epanechnikov kernel is optimal in a mean square error sense though the loss of efficiency is small for the kernels listed previously.

1. Gaussian Kernel
2. Epanechnikov kernel
3. Uniform kernel
4. Triangular kernel
5. Time-varying kernel

![KDEs in different Kernel Function](assets\KDE-Figure1.png)

Although the kernel function is important for the smoothness and shape of the density estimate. As the size of sample data increases, the impact of different kernel function decreases.

Kernel Density Estimation: Parameter 2 `Bandwidth` (h)
It controls the width of the kernel function and influences the smoothness of the resulting density curve. A smaller bandwidth capture more noise(underfitting). A bigger bandwidth tends to be smoother but obscure important information(overfitting). Selecting the appropriate bandwidth is essential.

## Example
The script below plot the density estimate of the weekly returns of Bitcoin using different library available including `scipy` and `sklearn`. Noted that for 1-D array `bw_method` in `gaussian_kde` and `bandwidth` in `KernelDensity` is not the same [1](https://stackoverflow.com/questions/68396403/kernel-density-estimation-using-scipys-gaussian-kde-and-sklearns-kerneldensity). `gaussian_kde.bw_method = KernelDensity.bandwidth * np.std(data)`.

```python
# Scipy gaussian kernel
from scipy.stats import gaussian_kde, norm
from sklearn.neighbors import KernelDensity

x_grid = np.linspace(df['Change%'].min(), df['Change%'].max(), 200)

BANDWIDTH = 1

# Scipy Gaussian KDE
kernel = gaussian_kde(df['Change%'], bw_method=BANDWIDTH)
kde_values = kernel(x_grid)

# Noted that gaussian_kde.bw_method = KernelDensity.bandwidth * standard deviation of data

# Sklearn Epanechnikov KDE
kernel = KernelDensity(kernel='gaussian', bandwidth=BANDWIDTH * np.std(df['Change%'])).fit(df['Change%'].to_numpy().reshape(-1, 1))
log_kde_values2 = kernel.score_samples(x_grid.reshape(-1, 1))
kde_values2 = np.exp(log_kde_values2)

# Sklearn Epanechnikov KDE
kernel = KernelDensity(kernel='epanechnikov', bandwidth=BANDWIDTH * np.std(df['Change%'])).fit(df['Change%'].to_numpy().reshape(-1, 1))
log_kde_values2 = kernel.score_samples(x_grid.reshape(-1, 1))
kde_values2 = np.exp(log_kde_values2)

# Normal PDF
mean, std = np.mean(df['Change%']), np.std(df['Change%'])
normal_values = norm.pdf(x_grid, mean, std)
```

Other than estimating the return of portfolio or an asset, it is great for analyzing your strategy future return.

Ps: Although Gaussian kernel density estimation and normal distribution uses the same Probability Distribution Function. They have different formula. On a concept level, normal distribution assume that the data is normally distributed while Gaussian kernel density estimation does not.

## Challenges:
### Boundary Bias
Since there's no data beyond the boundary, this results in an underestimation of the density near the edge. Mirroring data is one of the solution for boundary bias [2](https://medium.com/illumination/kernel-density-estimate-964fd46d54df#:~:text=Let's%20have%20an%20example,). Using boundary kernel is also one solution for this, however, I have not found any resources yet.
### Bandwidth Selection
Choosing an appropriate bandwidth is crucial for an accurate density estimation. Various methods such as rule-of-thumb (Silverman) methods, cross-validation and plug-in methods can help the issue.
### Computation Cost
Kernel density estimation can be computationally intensive for large datasets or high dimensional data. This is because it involves calculating up the kernel function for each data point and summing them.

## Useful Resources
García-Portugués, E. (2025). _2.5 Practical issues | Notes for Nonparametric Statistics_. [online] Bookdown.org. Available at: https://bookdown.org/egarpor/NP-UC3M/kde-i-prac.html [Accessed 27 Feb. 2025].

‌Wikipedia Contributors (2019). _Kernel density estimation_. [online] Wikipedia. Available at: https://en.wikipedia.org/wiki/Kernel_density_estimation.

‌Wikipedia Contributors (2024). _Kernel (statistics)_. Wikipedia.
