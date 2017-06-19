```bash
# to be able to install scipy in Ubuntu
sudo apt-get install gfortran libopenblas-dev liblapack-dev
```

Useful links:
---
http://stats.stackexchange.com/questions/152644/what-algorithm-should-i-use-to-detect-anomalies-on-time-series
http://www.slideshare.net/Anodot/analytics-for-largescale-time-series-and-event-data
https://machinelearningmastery.com/machine-learning-with-python/
http://stats.stackexchange.com/questions/81538/is-it-possible-to-train-a-one-class-svm-to-have-zero-training-error
http://stats.stackexchange.com/questions/96922/one-class-classifier-cross-validation
http://stackoverflow.com/questions/24078301/custom-cross-validation-split-sklearn
http://stackoverflow.com/questions/11116697/how-to-get-most-informative-features-for-scikit-learn-classifiers

##### Netflix:
http://probdist.com/netflix-anomaly-detection/

http://www.covert.io/    # various papers
http://www.mlsecproject.org/
http://www.secrepo.com/

Data sets:
*http://mcfp.weebly.com/the-ctu-13-dataset-a-labeled-dataset-with-botnet-normal-and-background-traffic.html (2011)
*http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
*http://kdd.ics.uci.edu/databases/kddcup99/kddcup.names
*http://digitalcorpora.org/

\*: http://www.secrepo.com/

Poseidon (uses Deep Learning)
https://gab41.lab41.org/machine-learning-for-network-security-theres-an-app-for-that-e9bc01139f19#.aboloxed9
https://github.com/Lab41/poseidon

Null-hypothesis
----------------
A general statement or default position that there is no relationship between
two measured phenomena, or no association among groups

P-Value
---------------
The p-value is defined as the probability of obtaining a result equal to or
"more extreme" than what was actually observed, when the null hypothesis is true.

T-Test
----
(source: https://www.socialresearchmethods.net/kb/stat_t.php)
The t-test assesses whether the means of two groups are statistically different from each other.
This analysis is appropriate whenever you want to compare the means of two groups,
and especially appropriate as the analysis for the posttest-only two-group
randomized experimental design.

Student's t-distribution
----
tbd



Replicator Neural Networks - RNN
-----------------------------------
Replicator neural networks squeeze the data through a hidden layer that uses
a staircase-like activation function. The staircase-like activation function
makes the network compress the data by assigning it to a certain number of clusters
(depending on the number of neurons and number of steps).


Principal Component Analysis - PCA
-------------------------------------
http://setosa.io/ev/principal-component-analysis/
https://georgemdallas.wordpress.com/2013/10/30/principal-component-analysis-4-dummies-eigenvectors-eigenvalues-and-dimension-reduction/

So getting the eigenvectors gets you from one set of axes to another.
These axes are much more intuitive to the shape of the data now.
These directions are where there is most variation, and that is where there is
more information (think about this the reverse way round. If there was no
variation in the data [e.g. everything was equal to 1] there would be no information,
it’s a very boring statistic – in this scenario the eigenvalue for that dimension would equal zero,
because there is no variation).


Neural Networks
----------------
http://www.eis.mdx.ac.uk/staffpages/rvb/teaching/BIS3226/hand11.pdf


Area Under Curve - AUC
---
Consider a plot of the true positive rate vs the false positive rate as the
threshold value for classifying an item as 0 or is increased from 0 to 1: if the
classifier is very good, the true positive rate will increase quickly and the area
under the curve will be close to 1. If the classifier is no better than random guessing,
the true positive rate will increase linearly with the false positive rate and
the area under the curve will be around 0.5.


Softmax function or normalized exponential function:
------------------------------------------------------
A generalization of the logistic function that "squashes" a K-dimensional vector
z of arbitrary real values to a K-dimensional vector σ(z) of real values
in the range (0, 1) that add up to 1.

In probability theory, the output of the softmax function is used to represent a
categorical distribution - that is, a probability distribution over K different
possible outcomes.
The softmax function is the gradient-log-normalizer of the categorical probability distribution
In the field of reinforcement learning, a softmax function can be used to convert values into action probabilities.
In machine-learned neural networks, the softmax function is often implemented at
the final layer of a network used for classification. Such networks are then trained
under a log loss (or cross-entropy) regime, giving a non-linear variant of
multinomial logistic regression.
Sigmoidal or Softmax normalization is a way of reducing the influence of extreme
values or outliers in the data without removing them from the dataset.
It is useful given outlier data, which we wish to include in the dataset while
still preserving the significance of data within a standard deviation of the mean.
The data are nonlinearly transformed using one of the sigmoidal functions.


Markov Chain
-------------------------------------------------------
In probability theory and statistics, a Markov chain or Markoff chain, named
after the Russian mathematician Andrey Markov, is a stochastic process that satisfies
the Markov property (usually characterized as "memorylessness"). Loosely speaking,
a process satisfies the Markov property if one can make predictions for the future
of the process based solely on its present state just as well as one could knowing
the process's full history. i.e., conditional on the present state of the system,
its future and past are independent.

In discrete time,
the process is known as a discrete-time Markov chain. It undergoes transitions
from one state to another on a state space, with the probability distribution of
the next state depending only on the current state and not on the sequence of events that preceded it.

In continuous time,
the process is known as a Continuous-time Markov chain. It takes values in some
finite state space and the time spent in each state takes non-negative real values
and has an exponential distribution. Future behaviour of the model (both remaining
time in current state and next state) depends only on the current state of the model
and not on historical behaviour.


Deep Learning vs SVMs and RFs
---
As a rule of thumb, SVMs are great for relatively small data sets with *fewer outliers*.
Random forests may require more data but they almost always come up with a pretty robust model.
And deep learning algorithms, they require "relatively" large datasets to work well,
and you also need the infrastructure to train them in reasonable time.
Also, deep learning algorithms require much more experience: Setting up a neural
network using deep learning algorithms is much more tedious than using an off-the-shelf
classifiers such as random forests and SVMs.
(http://www.kdnuggets.com/2016/04/deep-learning-vs-svm-random-forest.html)


Novelty and Outlier Detection
--
http://scikit-learn.org/stable/modules/outlier_detection.html


One class SVM for outlier detection
---
http://scikit-learn.org/stable/auto_examples/svm/plot_oneclass.html
http://scikit-learn.org/stable/modules/outlier_detection.html


Isolation Forest
---
The path of the tree is a measure of irregularity
Based on assumption that malicious actions are few and different
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html#sklearn.ensemble.IsolationForest


Graph based anomaly detection:
----------------------------------------
- Temporal Data:
https://github.com/SocioPatterns/neo4j-dynagraph/wiki/Representing-time-dependent-graphs-in-Neo4j

- Spatial Data:
https://neo4j.com/blog/outliers-opportunities-graph-spatial/
https://www.linkedin.com/pulse/finding-valuable-outliers-opportunities-using-graph-spatial-urich

https://groups.google.com/forum/#!topic/neo4j/gnlOWYG9fcU

- OutRank: A GRAPH-BASED OUTLIER DETECTION FRAMEWOR USING RANDOM WALK
http://www.cse.msu.edu/~ptan/papers/IJAIT.pdf

- Outlier Detection for Graph Data
https://www.microsoft.com/en-us/research/wp-content/uploads/2013/01/gupta13b_asonam.pdf

- Outlier Detection in Graph Streams
http://www.charuaggarwal.net/ICDE11_conf_full_357.pdf

- Focused Clustering and Outlier Detection in Large Attributed Graphs
http://www.perozzi.net/projects/focused-clustering/
http://www.perozzi.net/projects/deepwalk/

- Graph-based Anomaly Detection and Description: A Survey
https://arxiv.org/pdf/1404.4679.pdf

- Anomaly Detection in Large Graphs
http://www.andrew.cmu.edu/user/lakoglu/pubs/oddball_TR.pdf


Autoencoder
------------------------------------------------------------------
https://en.wikipedia.org/wiki/Autoencoder
https://en.wikipedia.org/wiki/Feature_learning
The aim of an autoencoder is to learn a representation (encoding)
for a set of data, typically for the purpose of dimensionality reduction.
Recently, the autoencoder concept has become more widely used for
learning generative models of data
Architecturally, the simplest form of an autoencoder is a feedforward, non-recurrent
neural network very similar to the multilayer perceptron (MLP) – having an input
layer, an output layer and one or more hidden layers connecting them –, but with
the output layer having the same number of nodes as the input layer, and with
the purpose of reconstructing its own inputs
(instead of predicting the target value Y given inputs X).
Therefore, autoencoders are unsupervised learning models.


PCA:
-------------------------------------------------------------------
https://coolstatsblog.com/2015/03/21/principal-component-analysis-for-dummies/
PCA performs orthogonal transformation to the points (projection of the vertical line on an axis)
and then takes the projected points in order to reduce dimensionality.


Mean:
-------------------------------------------------------------------
a = [0,1,2,3,1,4]
sorted(a)
median  = a[len(a)/2]  # the number in the middle
The median better represents the typical example and is less susceptible to outliers

Mode:
---------------------------------------------------------------------
The most common value in a data set
Only works in discrete data

Histogram:
---------------------------------------------------------------------
Look up the bucket of a given value and the shape will tell you how likely to have that value.
The shape of the histogram can explain the variance

Variance (σ^2):
---------------------------------------------------------------------
The average of the squared differences from the mean
sum(((x-mean)^2 for x in a))/ len(a)

Standard Deviation (sqrt(variance) == sqrt(σ^2))
----------------------------------------------------------------------
A way to identify outliers.
sqrt(sum(((x-mean)^2 for x in a))/ len(a))
"Data points that lie more than one standard deviation from the mean can be considered unusual"

Population vs Sample
---------------------------------------------------------------------
If we are working with a sample of the dataset (population), we want to use the
"sample variance" instead of "population variance"

For N samples, you divide the sqared varuances by N-1 instead of N
sum(((x-mean)^2 for x in a))/ (len(a)-1)

Probability Density Functions
---------------------------------------------------------------------
Used for continuous data
np.random.uniform(-10.0, 10.0, 100000)
|
|
|---------------------------|
|                           |
|___________________________|
-10           0             10


Probability Mass Function
---------------------------------------------------------------------
Used for discrete dataset
The probabilities of discrete values occuring in a dataset

Normal Distribution
---------------------------------------------------------------------
'mu' is the desired mean
'sigma' is the standard deviation

mu = 5.0
sigma = 2.0
values  = np.random.normal(mu, sigma, 10000)
plt.hist(values, 50)
plt.show()

Exponential PDF (Probability Density Function)/ "Power Law"
--------------------------------------------------------------------
```python
# Many things in nature have an Exponential behaviour
from scipy.stats import expon
import matplotlib as plt

x = np.arrange(0, 10, 0.001)
plt.plot(x, expon.pdf(x))
```

|
|\
| \
|   \_______________
|___________________________________


Binomial Probability Mass Function
----------------------------------------------------------------------
```python
from scipy.stats import binom
import matplotlib as plt

n, p = 10, 0.5
x = np.arrange(0, 10, 0.001)
plt.plot(x, binom.pmf(x, n, p)
```

|           |
|           |   |
|       |   |   |
|       |   |   |
|   |   |   |   |   |
|___|___|___|___|___|___


Poisson Probability Mass Function
----------------------------------------------------------------------
Example: My website gets on average 500 visits per day. What are the odds of getting 550?

```python
from scipy import poisson
import matplotlib as plt

mu = 500
x = np.arange(400, 600, 0.5)
plt.plot(x, poisson.pmf(x, mu))  # pmf = probability mass function
```

What is the equivalent of a probability distribution function when using discrete instead of continuous data?
A probability mass function.


Percentiles
--------------------------------------------------------------------------
What's the point at which x% of the values are less than that value?

```python
mu = .0
sigma = .5
values  = np.random.normal(mu, sigma, 10000)
plt.hist(values, 50)
plt.show()

np.percentile(values, 50)
```

Moments
--------------------------------------------------------------------------
Used to measure the shape of a data distribution (of a pdf)
- The first moment is the mean/ avg
- The second moment is the variance
- The 3rd moment is called skew - how lopsided is the distribution
negative -> towards right
positive -> towards left
- The 4th moment is the kurtosis : how thick is the tail and how sharp is the peak
compared to a normal distribution. E.x. higher peaks have thinner tails

```python
import scipy.stats as sp
import numpy as np
import matplotlin.pyplot as plt

values = np.random.normal(0, 0.5, 10000)
plt.hist(vals, 50)
plt.show()

# calculate the moments
np.mean(values)  # the data should average at about zero since normal
np.var(values)
sp.skew(values)  # should be around zero since we have a normal distribution
sp.kurtosis(values) # same here
```

Matplotlib
=========================================================================
- Draw a line graph
```python
from scipy.stats import norm
import maptlotlib.pyplot as plt

x = np.arange(-3, 3, 0.001)
plt.plot(x, norm.pdf(x))
plt.show()
- Multiple plots on One Graph
plt.plot(x, norm.pdf(x))
plt.plot(x, norm.pdf(x, 1.0, 0.5))
plt.show()
```

- Save to File
```python
plt.savefig('path', format='png')
```

- Adjusting the axes
```python
axes = plt.axes()
axes.set_xlim([-5, 5])
axes.set_ylim([0, 1.0])
axes.set_xticks([-5, -4, -3, -2, -1, 0 , 1, 2, 3, 4, 5])
axes.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
plt.plot(x, norm.pdf(x))
plt.plot(x, norm.pdf(x, 1.0, 0.5))
plt.show()
```

- Add a Grid
same as above but call
`axes.grid()` before plotting

- Change Line Types and Colors
```python
axes = plt.axes()
axes.set_xlim([-5, 5])
axes.set_ylim([0, 1.0])
axes.set_xticks([-5, -4, -3, -2, -1, 0 , 1, 2, 3, 4, 5])
axes.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
axes.grid()
plt.plot(x, norm.pdf(x), 'b-')  # blue solid line
plt.plot(x, norm.pdf(x, 1.0, 0.5), 'r:') # red vertical hashes
plt.show()
```

'--' dashed
'-.'
etc

- Labelling Axes and Adding a Legend

```python
axes = plt.axes()
axes.set_xlim([-5, 5])
axes.set_ylim([0, 1.0])
axes.set_xticks([-5, -4, -3, -2, -1, 0 , 1, 2, 3, 4, 5])
axes.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
axes.grid()
plt.plot(x, norm.pdf(x), 'b-')  # blue solid line
plt.plot(x, norm.pdf(x, 1.0, 0.5), 'r:') # red vertical hashes
plt.legend(["Name of first line", "Name of secind line"], loc=4) # location in chart
plt.show()
```

- XKCD Style!!! :) <3
```python
plt.xkcd()
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines('right').set_color('none')
ax.spines('top').set_color('none')
plt.xticks([])
plt.yticks([])

ax.set_ylim([-10, 10])

data = np.ones(100)
data[70:] -= np.arange(30) # subtract a large value from data after 70th
plt.annotate(
    'THE DAY I REALIZED\n I COULD COOK BACON\nWHENEVER I WANTED',
    xy=(70, 1), arrowprops=dict(arrowstyle='->'), xytext=(15, -10))
)

plt.plot(data)
plt.xlabel('time')
plt.ylabel('my overall health')
```

- Pie Chart

```python
# remove XKCD mode
plt.rcdefaults()

values = [12, 55, 4, 32, 14]
colors = ['r', 'g', 'b', 'c', 'm']
explode = [0, 0, 0.2, 0, 0]  # make one slice stand out by 20%
labels = ["India", "USA", "Russia", "China", "Europe"]
plt.pie(values, colors=colors, labels=labels, explode=explode)
plt.show()
```

- Bar Chart
```python
plt.bar(range(0, 5), values, colors=colors)
plt.show()
```
- Scatter plot

```python
x = randn(500)
y = randn(500)
plt.scatter(X, Y)
plt.show()
```

- Histogram
```python
incomes = np.random.normal(2700, 15000, 10000)
plt.hist(incomes, 50)
plt.show()
```

- Box & Whisker Plot
Useful for visualizing the spread & skew of data
The red line represents the median of the data and
the box represents the bounds of the 1st and 3rd quartiles
Half the data exists within the box
The dotted line "whiskers" indicate the range of the data
- except for ouliers which ar plotted outside the whiskers.
Outliers are 1.5x or more the interquartile range

```python
uniform_skewed = np.random.rand(100) * 100 - 40  # uniformely distributed numbers between -40 and 60
high_outliers = np.random.rand(10) * 50 + 100    # a few outliers above 100
low_outliers = np.random.rand(10) * -50 -100     # a few outliers below 100

data = np.concatenate((uniform_skewed, high_outliers, low_ourliers))

plt.boxplot(data)
plt.show()
```

Covariance and Correlation
======================================================================
Weather two different attributes/ variables are related to each other
Can be hard to interpret it - how large is large?

Covariance
---------------------------------------------------------------------
Measures how to variables vary in tandem from their means
How much two variables seem to depend on one another.

To compute covariance:
- Two variables --> high-dimensional vectors
- Convert the vectors to variances from the mean
- Take the dot product (cosine of the angle between them) of the two vectors
- Divide by the sample size

Correlation
--------------------------------------------------------------------
![xkcd:correlation](https://imgs.xkcd.com/comics/correlation.png)

Correlation is to solve the problem of how large is large
Correlation does NOT imply causation

Divide covariance by the standard deviations of both variables to normalize things
Correlation of -1 means perfect inverse correlation
Correlation of 0 means no correlation
Correlation of 1 means perfect correlation

Use correlation to decide about what experiments to run

#### the hard way

```python
def deviations_mean(x):
    xmean = mean(x)
    return [xi-mean for xi in x]

def covariance(x, y):
    n = len(x)
    return dot(deviations_mean(x), deviations_mean(y)) / (n-1)

page_speeds = np.random.normal(3.0, 1.0, 1000)
purchase_amount = np.random.normal(50.0, 10.0, 1000)

scatter(page_speeds, purchase_amount)
covariance(page_speeds, purchace_amount) # no actual correlation since data are random with normal distribution

# to create a fake correlation just for the exercise's purpose
purchase_amount /= page_speeds
scatter(page_speeds, purchase_amount)
covariance(page_speeds, purchace_amount)
```

The covariance is sensitive to the units used in the variables - which makes it difficult to interpret

```python
def correlation(x, y):
    std_dev_x = x.std()
    std_dev_y = y.std()

    if std_dev_x == 0 or std_dev_y == 0:
        return 0
    return covariance(x, y)/ std_dev_x / std_dev_y
```

or the numpy way:

```python
np.corrcoef(page_seeds, purchase_amount)

# to create a PERFECT fake correlation just for the exercise's purpose
purchase_amount = 100 + purchace_amount * 3  # linear dependency
scatter(page_speeds, purchase_amount)
covariance(page_speeds, purchace_amount)
```

Conditional probability
------------------------------------------------------------------------
Measure the relationship between two events happening
If I have two events that depend on each other, what's the probability that both will occur?

`P(B|A) = P(A,B)/ P(A)`

`P(A, B)`: The probability of A and B happening independently


Bayes Theorem
-----------------------------------------------------------------------
`P(A|B) = (P(A)P(B|A)) / P(B)`
The probability of A given B is the probabilty of A times the probability of B given A over the probability of B.
Which means that the probability of something that depends on B depends on the probability of B given that someting.


A/B Testing
------------------------------------------------------------------------
A Control set of people --> orange button
A Test set of people --> blue button
--> Measure the difference!
1. Identify the metric you wan to optimize for:
- Order amounts
- Profit
- Ad clicks
- Order quantity
2. Variance is your enemy
Running an experiment that collects the test users' data for an x amount of time
does not mean that you can safely decide whether the change has a positive effect or a negative effect since
the x amount of time can be very misleading and sales have great Variance.

Use conversion metrics with less variance, e.g. order quantities vs $ amounts.

T-Tests and P-Values
-------------------------------------------------------------------------
T-Tests / T- Statistic
To know if a result is real or not

- A measure of the difference between the two sets expressed in units of standard error
- The size of the difference relative to the variance in the data
- A HIGH T-value means there's probably a real difference between the two datasets
- Asumes a normal distribution of behaviour
    - which is a good assumption if measuring revenue as conversion
    - Fisher's exact test (clickthrough rates)
    - E-test (transactions per user)
    - chi-squared test (for product quantities purchased)

The P-Value
The probability of A and B satisfying the "null hypothesis" - that there is no real difference between A and B
A low p-value implies significance.
It is the probability of an observation lying at an extreme t-value assuming the null hypothesis

- Chose some threshold for significance before the experiment:
    e.g. 1% or 5% - it depends on the tolerance you have for the experiment to prove wrong
- When your experiment is over:
    - Measure the P-Value
    - If it is less than your significance threshold, then you can reject the null hypothesis
        - if it is a positive change, roll it out
        - if it is a negative change, discard it.
```python
import numpy as np
from scipy import stats

A = np.random.normal(25., 5., 10000)
B = np.random.normal(26., 5., 10000)  # the old situation was better than A - negative results expected

stats.ttest_ind(A, B) # negative t-value and very negative pvalue which means
#that this result is not because of random variance, thus the negative t-value is unfortunately correct

B = np.random.normal(25., 5., 10000)  # the old the same as A
stats.ttest_ind(A, B) # negative t-value and a positive p-value put small e.g. 0.3 - pretty high compared to the acceptable 1-5%

# increase the sample size
A = np.random.normal(25., 5., 100000)
B = np.random.normal(26., 5., 100000)
stats.ttest_ind(A, B) # negative t-value and a positive p-value put small e.g. 0.2 - still pretty high compared to the acceptable 1-5% but a bit better


# sanity check
stats.ttest_ind(A, A) --> t-value=0.0 and pvalue=1.0 because there are no differences
```

How long Do I Run an Experiment
-------------------------------------------------------------------
- If we have achieved a significant result (positive or negative) then stop
- If you no longer observe any meaningful trends over time in your p-value
- You reach a flat line - pre-established upper bound on time.

A/B Gotchas
--------------------------------------------------------------------
- Correlation does not imply causation!!! - this cannot be stressed enough!
Even a low p-value in well designed experiment does not imply causation
    - it could be a random change
    - other factors can play a big role
    - it should be clear in any decision making
Novelty Effects
    Re-run experiments much later and validate their impact to understand if it was a temporary
    effect due to the novelty of A
Seasonal Effects
    E.g. consumption near Christmas
    An experiment near Christmas will probably not represent the behavior during the rest of the year.
Selection Bias
    - E.g. select customers based somehow on customer ID - but customers with low id are older and probably loyal
    and customers with large ID are newer - this may lead to ultimately test old versus new which is not
    the desired result and can have very misleading results.
    - Run an A/A test periodically to check - there should not be any change in behavior
    - Audit your segment assignment algorithms
Data Polution
    - E.g. when there is a robot - crawler - good reason to measure conversion based on something that requires real money
    (e.g. use google analytics to identify useful and not)
    - In general: Are outliers skewing the result?
Attribution Errors
    - How you are actually counting the conversions
    - Multiple experiments at once? Can they interfere with each other?


Dealing with outliers
--------------------------------------------------------------------
Throw out outliers only in the case where they are not consistent with your use case, what you are trying to model.
E.g. do not throw out D. Trump when calculating the mean income just because you want to but
a single user who rates thousands of movies may be appropriate to leave aside.
How:
    - Standard Deviation - An outlier is a point outside one or more standard deviations
    - Look out the data and understand what they are

Apache Spark
-----------------------------------------------------------------------
DAG engine: Directed Acyclic Graph  // Apache Flink can be cyclic
DAG vertices are the Stages - there is a visualization tool to see how the stages and tasks are run.

Spark GraphX is only Scala for now.


StandAlone deployment:
SparkContext < --- >  Spark Master <----> Worker Node 1...n
            ^-------------------------------/^
In a cluster set up Spark Master == Mesos || Hadoop YARN Resource Manager


# How to set up on linux!
https://www.santoshsrinivas.com/installing-apache-spark-on-ubuntu-16-04/
https://www.youtube.com/watch?v=pZQsDloGB4w
https://www.ardentex.com/publications/RDDs-DataFrames-and-Datasets-in-Apache-Spark.pdf
https://www.youtube.com/watch?v=1a4pgYzeFwE
https://github.com/mahmoudparsian/pyspark-tutorial


https://databricks.gitbooks.io/databricks-spark-reference-applications/content/logs_analyzer/chapter3/save_an_rdd_to_a_database.html

MySQL
https://www.percona.com/blog/2016/08/17/apache-spark-makes-slow-mysql-queries-10x-faster/
https://www.percona.com/blog/2015/10/07/using-apache-spark-mysql-data-analysis/

mongodb
https://docs.mongodb.com/spark-connector/current/python-api/
https://github.com/mongodb/mongo-hadoop/blob/master/spark/src/main/python/README.rst

# packages
https://spark-packages.org/package/freeman-lab/thunder

Books:
---
http://isbn.directory/book/9781783288519
http://it-ebooks.info/book/5767/
http://isbn.directory/book/9781784392574

https://www.amazon.com/Learning-Spark/dp/1449358624/ref=cm_cr_dp_d_rvw_txt?ie=UTF8
https://www.amazon.com/dp/1491912766?ref=emc_b_5_i
https://www.amazon.com/Learning-Spark-Streaming-Practices-Optimizing/dp/1491944242/ref=pd_sbs_14_7?_encoding=UTF8&pd_rd_i=1491944242&pd_rd_r=WJFEBT2REQDYCDEK7GP4&pd_rd_w=Uvaxf&pd_rd_wg=S4hfH&psc=1&refRID=WJFEBT2REQDYCDEK7GP4


https://www.amazon.com/Pro-Spark-Streaming-Real-Time-Analytics/dp/1484214803%3FSubscriptionId%3DAKIAIVTZXR6R55637OFA%26tag%3Disbndir-20%26linkCode%3Dsp1%26camp%3D2025%26creative%3D165953%26creativeASIN%3D1484214803

About Spark and training: https://sparkhub.databricks.com/resources/ out-of date and not active...

These look good and up to date:

https://www.amazon.com/Fast-Data-Processing-Spark-Third/dp/1785889273/ref=mt_paperback?_encoding=UTF8&me=

https://www.amazon.com/Spark-Science-Cookbook-Padma-Chitturi/dp/1785880101/ref=pd_sbs_14_6?_encoding=UTF8&pd_rd_i=1785880101&pd_rd_r=F5PGRRS0RF7K0HM3FJKS&pd_rd_w=lkLBT&pd_rd_wg=uFeka&psc=1&refRID=F5PGRRS0RF7K0HM3FJKS

https://www.amazon.com/Apache-Spark-Beginners-Rajanarayanan-Thottuvaikkatumana/dp/1785885006/ref=pd_sbs_14_1?_encoding=UTF8&pd_rd_i=1785885006&pd_rd_r=F5PGRRS0RF7K0HM3FJKS&pd_rd_w=lkLBT&pd_rd_wg=uFeka&psc=1&refRID=F5PGRRS0RF7K0HM3FJKS

https://www.amazon.com/Spark-Data-Science-Srinivas-Duvvuri/dp/1785885650/ref=pd_bxgy_14_img_2?_encoding=UTF8&pd_rd_i=1785885650&pd_rd_r=NAG6DNRYXF88TA5W39M3&pd_rd_w=XmdxV&pd_rd_wg=WqSxw&psc=1&refRID=NAG6DNRYXF88TA5W39M3


Spark has to serialize data ... a lot

- RDD Resilient Distributed Dataset
  - compile time type-safe
  - lazy --> collect forces action
  - They're slow on non-JVM languages like Python.
  - It's too easy to build an inefficient RDD transformation chain, e.g. do a reduceBy before a filter.

- DataFrame: Conceptually like a Pandas DataFrame but not mutable, the handling is much more difficult.
  - It has Schemas
  - We can think about it like relational db tables and sql queries.
  - More readable than rdds.
  - DataFrame queries are optimized like in a RDBMS
  - `collect()` returns a **Row** that isn't type-safe.

- Parquet files: a very efficient column based storage.
- Catalyst: sql/ df queries optimization (with some reservations...)

- Datasets:
  - Good luck googling "Datasets" and getting the results you need...
  - An extension to the DataFrame API
  - Conceptually similar to RDDs.
  - Tend to use less memory
  - Use Tungsten's fast in-memory encoding
    (as opposed to JVM objects or serialized objects on the heap)
  - Expose expressions and fields to the DataFrame query planner, where the optimizer can use them to make decisions.
    (This can't happen with RDDs.)
    They have functionality similar to in-memory assembly to generate code on the fly
    and convert back and forth from compiler agnostic type to Row and
    vice versa so as to facilitate the query planner.
  - Dataset[Person] = df.as[Person] - well, in scala at least...
  - Spark understands the structure of data in Datasets, because they're typed.
  - Spark uses encoders to translate between these types ("domain objects") and
    Spark's compact internal Tungsten data format
  - It generates these encoders via runtime code-generation. The generated code
    can operate directly on the Tungsten compact format
  - Memory is conserved, because of the compact format. Speed is improved by
    custom code-generation

- Python and Scala API is kept the same, with the first one being left slightly behind.
- Databricks is the official support for Spark.



TF-IDF Term Frequency and Inverse Document Frequency
---
Term Frequency:
How often a word occurs in a document: A word that comes up frequently is probably important to the meaning.
Document Frequency:
How often a word occurs in an entire set of documents: e.g. "the", "and" etc,
which means that they occur frequently and might not play a very crucial role in the body of documents

TFIDF: Term Frequency / Document Frequency or Term Frequency * Inverse Document Frequency (or 1/DF)
We use the log instead of the actual value.
Also TFIDF assumes that a document is just a BoW - no relationships between the words
Words can be represented as a hash value for efficiency
What about synonyms, tenses, abbreviations, capitalizations, misspellings?

A very simple search algorithm:
Compute the TFIDF for every word in a corpus
For a given search word, sort the documents by their TFIDF score for that word
Display results


On Kernels: https://help.pythonanywhere.com/pages/IPythonNotebookVirtualenvs/

Basics:
---
#### The Derivative:
The derivative of a function of a real variable measures the sensitivity to change
of the function value with respect to a change in its argument (input value).
Derivatives are a fundamental tool of calculus. For example, the derivative of
the position of a moving object with respect to time is the object's velocity:
this measures how quickly the position of the object changes when time advances.

Useful Advice from KazAnova:
----
http://blog.hackerearth.com/winning-tips-machine-learning-competitions-kazanova-current-kaggle-3?stc_status=success&stc_hash=f5072850af1b19e186c5311f9043d92f
http://blog.kaggle.com/2016/02/10/profiling-top-kagglers-kazanova-new-1-in-the-world/

Feature Transformations:
---
- Log, Sqrt: smoothens the variables
- Dummies for categorical variables
- Sparse matrices to be able to compress data
- 1st derivatives to smoothen data
- Weights of evidence (transfoming variables while using information of the target variable) ??? needs xplaining
- Unsupervised methods that reduce dimensionality (SVD, PCA, ISOMAP, KDTREE, clustering)

Scaling:
---
- Max Scaler: Divide each feature with highest absolute value
- Normalization: Subtract the mean and divide with the standard deviation
- Conditional Scaling: scale under certain conditions, e.g. in medicine scale per subject to make features comparable ??? needs xplaining

Feature Extraction:
---
- Text classification: get the corpus of words and produce the TFIDF
- Sounds: convert sounds to frequencies through Fourier Transformations
- Images: convolution e.g. break down an image to pixels and extract the different parts of the image.
- Interactions: Show e.g. how an item is popular AND the customers like it through a boolean one-hot-encoded feature
- Other cases that make sense:
    - similarity features,
    - dimensionality reduction,
    - reduction features
    - even predictions from other models as features
    etc

    1. What are the steps you follow for solving a ML problem? Please describe from scratch.

    A. Understand the data - After you download the data, start exploring features. Look at data types. Check variable classes. Create some univariate - bivariate plots to understand the nature of variables.
    
    B. Understand the metric to optimize - Every problem comes with a unique evaluation metric. It's imperative for you to understand it, specially how does it change with target variable.
    
    C. Decide cross validation strategy - To avoid overfitting, make sure you've set up a cross validation strategy in early stages. A nice CV strategy will help you get reliable score on leaderboard.
    
    D. Start hyper parameter tuning - Once CV is at place, try improving model's accuracy using hyper parameter tuning. It further includes the following steps:
        - Data transformations: It involve steps like scaling, removing outliers, treating null values,  transform categorical variables, do feature selections, create interactions etc.
        - Choosing algorithms and tuning their hyper parameters: Try multiple algorithms to understand how model performance changes.
        - Saving results: From all the models trained above, make sure you save their predictions. They will be useful for ensembling.
    
    E. Combining models: At last, ensemble the models, possibly on multiple levels. Make sure the models are correlated for best results.

    2. What are the model selection and data manipulation techniques you follow to solve a problem?
      - Time series: I use GARCH, ARCH, regression, ARIMA models etc.
      - Image classification: I use deep learning (convolutional nets) in python.
      - Sound Classification : Common neural networks
      - High cardinality categorical (like text data): I use linear models, FTRL, Vowpal wabbit, LibFFM, libFM, SVD etc.

        For everything else, Gradient boosting machines (like  XGBoost and LightGBM) and deep learning (like keras, Lasagne, caffe, Cxxnet).
        Decide what model to keep/drop in Meta modelling with feature selection techniques.
        Some of the feature selection techniques used:

        - Forward (cv or not) - Start from null model. Add one feature at a time and check CV accuracy. If it improves keep the variable, else discard.
        - Backward (cv or not) - Start from full model and remove variables one by one. If CV accuracy improves by removing any variable, discard it.
        - Mixed (or stepwise) - Use a mix of above to techniques.
        - Permutations
        - Using feature importance - Use random forest, gbm, xgboost feature selection feature.
        - Apply some stats’ logic such as chi-square test, anova.

        Data manipulation could be different for every problem :
          - Time series : You can calculate moving averages, derivatives. Remove outliers.
          - Text : Useful techniques are tfidf, countvectorizers, word2vec, svd (dimensionality reduction).
                   Stemming, spell checking, sparse matrices, likelihood encoding, one hot encoding (or dummies), hashing.
          - Image classification: Here you can do scaling, resizing, removing noise (smoothening), annotating etc
          - Sounds : Calculate Furrier Transforms , MFCC (Mel frequency cepstral coefficients), Low pass filters etc
          - Everything else : Univariate feature transformations (like log +1 for numerical data),
                              feature selections, treating null values, removing outliers, converting categorical variables to numeric.

    3. Can you elaborate cross validation strategy?

    Cross validation means that from my main set, I create RANDOMLY 2 sets. I built (train) my algorithm with the first one (let’s call it training set) and score the other (let’s call it validation set). I repeat this process multiple times and always check how my model performs on the test set in respect to the metric I want to optimize.

    The process may look like:
      - For 10 (you choose how many X) times
      - Split the set in training (50%-90% of the original data)
      - And validation (50%-10%  of the original data)
      - Then fit the algorithm on the training set
      - Score the validation set.
      - Save the result of that scoring in respect to the chosen metric.
      - Calculate the average of these 10 (X) times. That how much you expect this score in real life and is generally a good estimate.
      - Remember to use a SEED to be able to replicate these X splits
      - Other things to consider is Kfold and stratified KFold . Read here. For time sensitive data,
        make certain you always the rule of having past predicting future when testing’s.

    4. Can you please explain some techniques used for cross validation?

    - Kfold
    - Stratified Kfold
    - Random X% split
    - Time based split
    - For large data, just one validation set could suffice (like 20% of the data – you don’t need to do multiple times).

    14. What techniques perform best on large data sets on Kaggle and in general ? How to tackle memory issues ?

    Big data sets with high cardinality can be tackled well with linear models. Consider sparse models. Tools like vowpal wabbit. FTRL , libfm, libffm, liblinear are good tools matrices in python (things like csr matrices). Consider ensembling (like combining) models trained on smaller parts of the data.

    15. What is the SDLC (Sofware Development Life Cycle) of projects involving Machine Learning ?

    Give a walk-through on an industrial project and steps involved, so that we can get an idea how they are used. Basically, I am in learning phase and would expect to get an industry level exposure.
    Business questions: How to recommend products online to increase purchases.
    Translate this into an ml problem. Try to predict what the customer will buy in the future given some data available at the time the customer is likely to make the click/purchase, given some historical exposures to recommendations
    Establish a test /validation framework.
    Find best solutions to predict best what customer chose.
    Consider time/cost efficiency as well as performance
    Export model parameters/pipeline settings
    Apply these in an online environment. Expose some customers but NOT all. Keep test and control groups
    Assess how well the algorithm is doing and make adjustments over time.

# Overfitting in Regression
---
- When we have few observations (N small) - in contrast to when we have many observations (N big),
where it is very hard to follow all the points.
We need to have lots of examples of every possible value for a features - which is very hard, and it
gets a lot harder as the # of features increases.

# The Ridge Objective
---
Modify the cost function to balance:
- How well a function fits the data
- The magnitude of the coefficients

Total cost = measure of fit + measure of the magnitude of the coefficients.
measure of fit --> small # = good fit to training data
momoc -- >  small # = not overfit.

`#todo: better represent Math...`

RSS(w) + λ ||w||2 ^2
λ = tuning parameter
if λ=0:
   reduces to minimizing RSS(w), w^ LS (least squares)

if λ=infinity:
   for solutions were w^ != 0, then the total cost is infinite
   if w^ = 0, then the total cost = RSS(0) --> solution is w^ = 0

if 0<λ<infinity:
    then 0<= ||w^||2^2 <= ||w^LS||2^2

This is referred to as Ridge Regression a.k.a. L2 normalization

Leave One Out Cross Validation - To minimize λ
-------
Create as many folds for cv as the # of data points
For each value of L2_penalty, fit a model for each fold and compute the average MSE
Save the squared error in a list of MSE for each L2_penalty.
Keep the best penalty - where MSE is the minimum

Plot the coefficients w^vj (y axis) by λ (x axis) to see how the coefficients behave.
As λ -> infinity our solution w^ -> 0
and vice versa (w^LS)
the solution is somewhere in between.


Computing the gradient of the Ridge Regression Objective:
---

## Gradient Boosting
http://blog.kaggle.com/2017/01/23/a-kaggle-master-explains-gradient-boosting/
The objective is to minimize squared error:
Initial idea:
  Fit a model to the data, F_1(x) = y
  Fit a model to the residuals, h_1(x) = y - F_1(x)
  Create a new model: F_2(x) = F_1(x) + h_1(x)
In general insert more models that correct the errors of the previous model.

h_m(x) is just a model (in practice h_m is almost always a tree learner).
So Gradient boosting is just a framework to iteratively improve any weak learner.
F_0(x) = \underset{\gamma}{\arg\min} \sum_{i=1}^n L(y_i, \gamma) = \underset{\gamma}{\arg\min} \sum_{i=1}^n (\gamma - y_i)^2 = {\displaystyle {\frac {1}{n}}\sum _{i=1}^{n}y_{i}.}

## Boosting - in general

Boosting in general means using many different weaker models to improve the accuracy of the core model.
It works by weighing the models and adjusting the weights accordingly to how good - certain - and how bad - uncertain a model is.

### Ada Boost
For a good classifier (e.g. with weighted error 0.01) it gives high weight, so it trusts its predictions. For an uncertain classifier (e.g. with weighted error 0.5) it gives zero weight, so it ignores its prediction. For a bad classifier (e.g. with weighted error 0.99) it gives negative weight, so it does the opposite than the predicted.
To re-compute weights αi, the concept is that if we made a mistake we increase the weight so as to increase the importance of this data point, and be more careful. So, the next classify is going to pay much more attention to this particular data point because it was a mistaken one. If it is correct then we decrease the importance of this data point.


ROC Curve (Receiver Operating Characteristics):
---
http://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
ROC curves typically feature true positive rate on the Y axis, and false positive
rate on the X axis. This means that the top left corner of the plot is the “ideal” point
- a false positive rate of zero, and a true positive rate of one.
This is not very realistic, but it does mean that a larger area under the curve (AUC) is usually better.


# Deep Learning:
---
Speech:
  audio --> phonemes --> transcript
  audio ---------------> transcript (DL approach - train a NN)

Image:
  image -(\*)-> bone lengths --> age
  image -------------------> age (not very suitable for DL, not enough data of child hand xrays for example)
  \* You can use DL to find where the hands and the bones are.

  image --> cars --> trajectory --> steering
      \--> pedestrians_/^
  image --------------------------> steering (Much more difficult, not sure it would work)


Goal: Build human-level speech system
---

```
________________________________________________________________
|  train                             | dev / validation| test  |
----------------------------------------------------------------
```
Human level error   1%  --> The Bias            | 1%                  | 1%  -> This is a high Bias
Training Set error  5%  --> of the algorithm    | 2% -> We have a     | 5%  -> and High Variance
Dev Set error       6%                          | 6% -> Variance Issue| 10% -> issue


Training error high? -- Y -- > try train longer or a  new model architecture or train a bigger model
          |
          N
          |
          V
      Dev error High? -- Y --> More Data, Regularization, New Model Architecture.
          |
          N
          |
          V
        Done!

### Automatic Data Synthesis
- OCR
- Speech Recognition
- NLP
- Video Games (RL)


Tips:
Make sure Dev and test set are from the **same distribution**

```
  50k  or not really relative data                 40 relevant data
__________________________________________        _____________
| Train                      | Train-Dev  |       | Dev |Test |
-------------------------------------------       -------------
                                                Aggregate Error
Human-level:    1% -> Bias                            1%
Training set : 10% -> --//                            2%
Training-dev:  10% -> Train / Test mismatch           
Dev:           10% -> -- // --                       21%
Test:          10% -> Overfit of Dev                 10%
```

So,
```
Training error high? -- Y -- > try train longer or a  new model architecture or train a bigger model
          |
          N
          |
          V
  Training-Dev error High? -- Y --> More Data, Regularization, New Model Architecture.
          |
          N
          |
          V
  Dev error high?  -- Y -- > More data similar to test, data synthesis, new architecture (Hail Mary sort of)
          |
          N
          |
          V
  Test error high?  -- Y --> Get more Dev data!
          |
          N
          |
          V
        Done!

```
Human level performance
```
        |---------------------------- There is an optimal error rate (Bayes rate)
        |             ________
        |- - - - - - /- - - - - - - -   accuracy
        |           /
        |         /
        |_______/____________________
```

Human error Medical example:
--
1. Typical Human 3%
2. Typical Doctor 1%
3. Expert Doctor 0.7%
4. Team of expert Doctors 0.5%

When labeling you can have layers where if a typical doctor for example is unsure, you get the expert to label it also
and trust the expert more etc.
However, for a measure of how good is your system and for the metric of Human Level Performance,
with which you will compare your system's performance to, one would and should get the
0.5% - the best there is.


What can AI/ DL do?
---
1) Anything that a person can do in < 1 sec
2) Predicting the outcome of the next in a sequence of events.

How to become better:
Read a lot of papers, replicate results and do the dirty work!


