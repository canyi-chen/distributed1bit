% -------------------------------------------------------------------------
This is a Matlab package for Distributed Decoding from Heterogeneous 1-bit 
Compressive Measurements. 
(version 1.0). 
This package is  maintained by [name removed]
% -------------------------------------------------------------------------
All files for reproducing the numerical results in the paper 
"Distributed Decoding from Heterogeneous 1-bit Compressive Measurements" are relegated to simulations subdirectory. We assume the root folder is the same folder the README.md resides in. Before you run each script to reproduce the numerical results in the paper, you shall add all the sub-folders into the search path by executing in Matlab

```matlab
addpath(genpath('.'));
```
The mehod KSW depends on cvx. Make sure you install *cvx* according to the guidance,

```html
http://cvxr.com/cvx/doc/install.html
```

All methods are implemented with Matlab (Version R2020a) and conducted on a server with 64-core Intel(R) Xeon(R) Gold 6148 CPU (2.40GHz) and 252 GB RAM. We implement the distributed algorithms in a fully synchronized distributed setting.  

  Figure 1 by compare_iterations.m
  Figure 2 by compare_local_sample_size.m
  Figure 3 by compare_total_sample_size.m
  Figure 4 by compare_heterogeneity_sigma.m
  Table 1  by compare_sparsity.m



To facilitate your reproducible run, please run

```shell
bash main.sh
```



% -------------------------------------------------------------------------
If you have any questions or find any buggs please contact [email removed]. Thank you.



