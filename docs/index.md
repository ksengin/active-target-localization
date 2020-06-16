This page contains codes and information for the paper "[Active Localization of Multiple Targets from Noisy Relative Measurements](https://ksengin.github.io/papers/wafr2020active.pdf)". The code is available [here](https://github.com/ksengin/active-target-localization/tree/master/target_localization).


## Abstract
Consider a mobile robot tasked with localizing targets at unknown locations by obtaining relative measurements. The observations can be bearing or range measurements. How should the robot move so as to localize the targets and minimize the uncertainty in their locations as quickly as possible? This is a difficult optimization problem for which existing approaches are either greedy in nature or they rely on accurate initial estimates.

We formulate this path planning problem as an unsupervised learning problem where the measurements are aggregated using a Bayesian histogram filter. The robot learns to minimize the total uncertainty of each target in the shortest amount of time using the current measurement and an aggregate representation of the current belief state. We analyze our method in a series of experiments where we show that our method outperforms a standard greedy approach. In addition, its performance is comparable to that of an offline algorithm which has access to the true location of the targets.


## Paper
View the paper [here](https://ksengin.github.io/papers/wafr2020active.pdf).
```
@inproceedings{
  engin2020active,
  title={Active localization of multiple targets from noisy relative measurements},
  author={Selim Engin and Volkan Isler},
  booktitle={The 14th International Workshop on the Algorithmic Foundations of Robotics (WAFR)},
  year={2020},
  url={https://arxiv.org/abs/2002.09850}
}
```


## Results
The following two visuals illustrate the active localization of 8 targets using bearing and range measurements, respectively.
<!-- ![alt text](https://github.com/ksengin/active-target-localization/blob/master/visuals/atl_bearing.gif?raw=true)
![alt text](https://github.com/ksengin/active-target-localization/blob/master/visuals/atl_range.gif?raw=true) -->


<center>
<iframe src="https://drive.google.com/file/d/1rrkkmvIhP80OuivfAPDipD9i71otX6fL/preview" width="360" height="360"></iframe>
<!-- </center>
<center> -->
<iframe src="https://drive.google.com/file/d/1H3HgEA7MIXLEwgV4ZN9g-M9TaBKa8XpB/preview" width="360" height="360"></iframe>
</center>



## Contact
Correspondence to: engin003 [at] umn [dot] edu
