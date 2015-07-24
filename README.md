## Incremental Approaches for Matrix Approximation 

Experimental code and results of:

T. Kitazawa, T. Matsuo, **Incremental Approaches for Matrix Approximation: Performance Evaluations and Their Possible Applications**, (in Japanese), *The Japanese Society for Artificial Intelligence SIG-FPAI-B501*, Aug. 2015.

### 1) Brute Force

Running Time: around 4000 sec.

### 2) Incremental SVD (iSVD)

Running Time: around 750 sec.

|  k  |  Covariance Error | Projection Error |
|----:|------------------:|-----------------:|
|   2 |   0.0657172694621 |              1.0 |
|   4 |   0.0491418902379 |              1.0 |
|   8 |   0.0256682270686 |              1.0 |
|  16 |   0.0122495458957 |              1.0 |
|  32 |   0.0052127371377 |              1.0 |
|  64 |  0.00168276318593 |              1.0 |
| 128 | 0.000446738422159 |              1.0 |
| 256 | 6.82607484728e-13 |    5228.83140371 |

### 3) Frequent Directions

|ell|Projection Error|Running Time (sec.)|
|--:|--:|--:|
|4| 1.9043975495| 0.591689|
|8| 1.53392089235| 0.877754|
|16| 1.14943840055| 2.401179|
|32| 1.00044963427| 6.108395|
|64| 1.00000728168| 17.863258|
|128| 1.00000006729| 84.104562|
|256| 1.0| 304.967199|