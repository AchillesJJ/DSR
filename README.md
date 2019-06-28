
# Differential Sharpe Ratio (DSR)
---------------------------
## Object function: Sharpe Ratio
Typical RL task tries to optimize the object function

$$U_T=U(R_1, R_2, ..., R_T)$$

where $R_i$ is the instaneous reward at time $t$. For general $Q$-learning or policy gradient algorithm, the object function is

$$U_T=\sum_{i=1}^{T}\gamma^{i-1}\cdot R_i$$

which is a discounted cunmulated sum of all one-step reward. However, such object function does not take market risk into consideration. The simplest benchmark to measure risk and return is the Sharpe ratio, i.e.

$$U_T=\frac{E(R)}{\text{std}(R)}$$

with $E(R)=(1/T)\sum_{i=1}^{T}R_i$ and std is the standard deviation. However, such object function is not additive, which prohibits us to use the old wisdom such as $Q$-learning and vanilla policy gradient methods. 

J.Moody propose the idea of DSR, which can turn $U$ above into an additive sum of single step reward.

First, for given step n, the Sharpe ratio $U_n$ can be estimated by

$$U_n=\frac{A_n}{K_n(B_n-A_n^2)^{1/2}}$$

with

$$A_n=\frac{1}{n}\sum_{i=1}^n R_i~~\text{and}~~B_n=\frac{1}{n}\sum_{i=1}^n R_i^2 ~~ K_n=(\frac{n}{n-1})^{1/2}$$

both $A$ and $B$ satisfy the following recurrent relation

$$A_n=\frac{1}{n}R_n+\frac{n-1}{n}A_{n-1}~~\text{and}~~B_n=\frac{1}{n}R_n^2+\frac{n-1}{n}B_{n-1}$$

Now we can extend such formulism to an exponential moving average Sharpe ratio on time scale $\eta^{-1}$ by making use of the EMA of $A$ and $B$,

$$S_t=\frac{A_t}{K_{\eta}(B_t-A_t^2)^{1/2}}$$

with

$$A_t=\eta R_t+(1-\eta)A_{t-1}=A_{t-1}+\eta\Delta A_t$$

$$B_t=\eta R_t^2 +(1-\eta)B_{t-1}=B_{t-1}+\eta\Delta B_t$$

$$K_{\eta}=(\frac{1-\eta/2}{1-\eta})^{1/2}$$

initialized with $A(0)=B(0)=0$.
Now we can write for small $\eta$ that

$$U_t=U_{t-1}+\eta \frac{\partial U_t}{\partial \eta}+O(\eta^2)$$

where we define the DSR as

$$D_t=\frac{\partial U_t}{\partial \eta}=\frac{B_{t-1}\Delta A_t-A_{t-1}\Delta B_t/2}{(B_{t-1}-A_{t-1}^2)^{3/2}}$$

Now if we expand the whole $U_T$ to first order of $\eta$, we have

$$U_T\simeq \eta \sum_{i=1}^T D_i + O(\eta^2)$$

Now the original object function of Sharpe ratio becomes totally additive to first order. One step reward is now replaced by $D_t$. It means that we can use $D_t$ as reward and leave all others of RL framework unchanged.

## Numerical simulation of DSR
Here use S&P500 daily close price to illustrate the idea of DSR. We note that the initial condition $A(0)=B(0)=0$ is singular for the definition of DSR. Thus we in practice, we shall use the former 200 days to calculate a base SR as the initial value of $A$ and $B$ and then begin to calculate $D_t$ in the following iteration.

```python
# encoding: utf-8
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt

dt = pd.read_csv('SP500.csv', index_col=0)['close']
pct = dt.pct_change().ffill().fillna(0.0)
pct = pct.values

def sharpe(ls):
    return np.mean(ls)/np.std(ls)

# ls1 contains the true SR values
ls1 = []
sr0 = sharpe(pct[:200])
for i in range(500):
    sr = sharpe(pct[:200+i+1])
    ls1.append(sr-sr0)
    sr0 = sr

# ls2 uses cumulated DSR to approximate SR 
ls2 = []
eta = 0.004
# use the first 200 days to set an initial value of SR
sr = sharpe(pct[:200]) 
for i in range(500):
    A = np.mean(pct[:200+i])
    B = np.mean(pct[:200+i]**2)
    delta_A = pct[200+i+1] - A
    delta_B = pct[200+i+1]**2 - B
    Dt = (B*delta_A - 0.5*A*delta_B) / (B-A**2)**(3/2)
    sr += eta * Dt
    ls2.append(Dt*eta)
```
The comparison between SR and cumulated DSR is presented a

![DSR](DSR.jpg)
























