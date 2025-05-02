Extended Fiducial Inference for Individual Treatment Effects via Deep Neural Networks
===============

This code implements the Double Neural Network (Double-NN) method for individual treatment effect (ITE) estimation within the framework of Extended Fiducial Inference (EFI), developed by Sehwan Kim and Faming Liang. The method uses deep neural networks to model both treatment and control outcome functions, with an additional network for parameter estimation. Leveraging the EFI framework enables principled uncertainty quantification without requiring reference distributions. Numerical experiments show that the Double-NN method outperforms conformal quantile regression (CQR) in ITE estimation tasks, offering both improved accuracy and a rigorous foundation for inference in deep learning-based models.

## Related Publication

Sehwan Kim, and Faming Liang (2025+), [Extended Fiducial Inference for Individual Treatment Effects via Deep Neural Networks](https://arxiv.org/abs/2407.21622), accepted by *Statistics and Computing*


## Description: ITE for nonlinear control and nonlinear treatment effect.

Causal inference is a fundamental problem in many disciplines such as medicine, econometrics, and social science. Formally, let $\{(y_1,x_1,t_1), (y_2,x_2,t_2),\ldots, (y_n,x_n,t_n)\}$ denote a set of observations drawn from the following data-generating equations: 

$$
y_i=c(x_i) +\tau(x_i) t_i+\sigma z_i,  \quad i=1,2,\ldots,n,
$$
where $\bx_i \in \mathbb{R}^d$ represents a vector of covariates of subject $i$, 
$t_i \in \{0,1\}$ represents the treatment assignment to subject $i$; 
$c(\cdot)$ represents the expected outcome of 
subject $i$ if assigned to the control group (with $t_i=0)$, and $\tau(\bx_i)$ is the 
expected treatment effect of subject $i$ if assigned to the treatment group (with $t_i=1$); $\sigma>0$ is the standard deviation,  and $z_i$ represent a standardized  random error that is not necessarily Gaussian.


The ITE is often defined as the conditional average treatment effect (CATE):   
\begin{equation} \label{CATEeq}
\tau(\bx)=\mathbb{E}(Y|T=1,\bx)-\mathbb{E}(Y|T=0,\bx),
\end{equation}
see e.g., \cite{shalit2017pehe} and \cite{Lu2018EstimatingIT}. 
Recently, \cite{lei2021ite} proposed to make 
predictive inference of the ITE by quantifying the uncertainty of 
\begin{equation} \label{ITEeq}
\tilde{\tau}_i:=Y(T=1,\bx_i)-Y(T=0,\bx_i):=Y_i(1)-Y_i(0),
\end{equation}
where $Y_i(t_i)$ denotes the potential outcome of subject  
$i$ with treatment assignment $t_i \in \{0,1\}$. Henceforth, we will call $\tilde{\tau}_i$ the predictive ITE.  



<p align="center">
    <img src="img/LR_example.png" width=600>
</p>

The above figure presents the results of EFI on Linear Regression: (left) a scatter plot of $\hat{z}_{i}$ (y-axis) versus $z_i$ (x-axis), (middle) a Q-Q plot of $\hat{z}_i$ and $z_i$, and (right) confidence intervals of $\beta_1$ produced by EFI and OLS.

The left panel shows that the imputed random error is quite similar to the true unknown random error. The middle panel demonstrates that the imputed random error exhibits similar distributional behavior to the true random errors. The right panel indicates that the inference from EFI is comparable to that of MLE, Bayes (with objective prior), and Generalized Fiducial Inference.

