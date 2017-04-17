Greetings to everyone who lived to see part five of our coursed!

The course has already brought together more than 1,000 participants, of which 520, 450 and 360 persons respectively completed first 3 homework assignments. <img src="https://habrastorage.org/files/1f8/981/50a/1f898150a862463bae79b42e3da32beb.jpg" align="right" width="400" /> About 200 participants have so far the maximum score. The outflow is much lower than in MOOCs, even though our articles are large. 

This session will be devoted to simple methods of composition: bagging and random forest. You will learn how you can get the distribution of the mean of the general population, if we have information only on a small part of it. We'll look how to reduce the variance and thus improve the accuracy of the model using compositions of algorithms. We'll also figure out what a random forest is, what parameters in it need to be "tuned up" and how to find the most important feature. We'll focus on practice with just a "pinch" of mathematics.


<spoiler title="Список статей серии">
1. [Первичный анализ данных с Pandas](https://habrahabr.ru/company/ods/blog/322626/)
2. [Визуальный анализ данных c Python](https://habrahabr.ru/company/ods/blog/323210/)
3. [Классификация, деревья решений и метод ближайших соседей](https://habrahabr.ru/company/ods/blog/322534/)
4. [Линейные модели классификации и регрессии](https://habrahabr.ru/company/ods/blog/323890/)
5. [Композиции: бэггинг, случайный лес](https://habrahabr.ru/company/ods/blog/324402/)
6. [Построение и отбор признаков. Приложения в задачах обработки текста, изображений и геоданных](https://habrahabr.ru/company/ods/blog/325422/)
7. Обучение без учителя: PCA, кластеризация, поиск аномалий
</spoiler>


<habracut/>

#### Plan of this article

1. [Bagging](https://habrahabr.ru/company/ods/blog/324402/#1-begging)
   - [Ensembles](https://habrahabr.ru/company/ods/blog/324402/#ansambli)
   - [Bootstrap](https://habrahabr.ru/company/ods/blog/324402/#butstrep)
   - [Bagging](https://habrahabr.ru/company/ods/blog/324402/#begging)
   - [Out-of-bag error](https://habrahabr.ru/company/ods/blog/324402/#out-of-bag-error)
2. [Random forest](https://habrahabr.ru/company/ods/blog/324402/#2-sluchaynyy-les)
    - [Algorithm](https://habrahabr.ru/company/ods/blog/324402/#algoritm)
    - [Сравнение с деревом решений и бэггингом](https://habrahabr.ru/company/ods/blog/324402/#sravnenie-s-derevom-resheniy-i-beggingom)
    - [Параметры](https://habrahabr.ru/company/ods/blog/324402/#parametry)
    - [Вариация и декорреляционный эффект](https://habrahabr.ru/company/ods/blog/324402/#variaciya-i-dekorrelyacionnyy-effekt)
    - [Смещение](https://habrahabr.ru/company/ods/blog/324402/#smeschenie)
    - [Сверхслучайные деревья](https://habrahabr.ru/company/ods/blog/324402/#sverhsluchaynye-derevya)
    - [Схожесть с алгоритмом k-ближайших соседей](https://habrahabr.ru/company/ods/blog/324402/#shozhest-sluchaynogo-lesa-s-algoritmom-k-blizhayshih-sosedey)
    - [Преобразование признаков в многомерное пространство](https://habrahabr.ru/company/ods/blog/324402/#preobrazovanie-priznakov-v-mnogomernoe-prostranstvo)
3. [Оценка важности признаков](https://habrahabr.ru/company/ods/blog/324402/#3-ocenka-vazhnosti-priznakov)
4. [Плюсы и минусы случайного леса](https://habrahabr.ru/company/ods/blog/324402/#4-plyusy-i-minusy-sluchaynogo-lesa)
5. [Домашнее задание №5](https://habrahabr.ru/company/ods/blog/324402/#5-domashnee-zadanie)
6. [Полезные источники](https://habrahabr.ru/company/ods/blog/324402/#6-poleznye-istochniki)


##1. Bagging
From the last lectures you learned about different classification algorithms, as well as how to validate and evaluate the quality of the model. But what if you've already found the best model and can no longer improve the accuracy of the model? In this case, you need to apply more advanced machine learning techniques, which can be jointly called "ensembles". Ensemble is a certain set, parts of which form a whole. From everyday life you know music ensembles, where several musical instruments are combined, or architectural ensembles with various buildings, etc. 

### Ensembles

A good example of ensembles is represented by Condorcet's jury theorem (1784). If each member of the jury has an independent opinion and the probability of correct decision of each juror is more than 0.5, then the probability of correct decision of the whole jury increases with the number of jurors and tends to one. At the same time, if the probability of being right is less than 0.5 for each juror, then the probability of correct decision by the jury decreases monotonically and tends to zero with an increasing number of jurors. 
$inline$\large N $inline$ is the number of jurors
$inline$\large p $inline$ is the probability of correct decision of a juror
$inline$\large \mu $inline$ is the probability of correct decisions of the whole jury
$$display$$ \large \mu = \sum_{i=m}^{N}C_N^ip^i(1-p)^{N-i} $$display$$
Если $inline$\large p > 0.5 $inline$, то $inline$\large \mu > p $inline$
Если $inline$\large N \rightarrow \infty $inline$, то $inline$\large \mu \rightarrow 1 $inline$

Let's look at another example of ensembles - "Wisdom of Crowds". Francis Galton in 1906 visited the market where a certain lottery was held for farmers. <img src="https://habrastorage.org/getpro/habr/post_images/d58/66e/f0b/d5866ef0bfe6416952f8ebc14f07ed2b.png" align="right" width=15% height=15%> 
There were about 800 people, and they tried to guess the weight of the bull that stood in front of them. Bull weighed 1,198 pounds. None of the farmers guessed the exact weight of the bull, but if we computed the average of their predictions, we'd get 1197 pounds.
This idea of error reduction was also used in machine learning.


### Bootstrap

Bagging (from Bootstrap aggregation) is one of the first and most basic kinds of ensembles. It was coined by [Leo Breiman](https://ru.wikipedia.org/wiki/Брейман,_Лео) in 1994.Bagging is based on a statistical method of bootstrap that allows to evaluate many parameters of complex distributions.
 
Bootstrap method consists of the following. Suppose we have a sample $inline$\large X$inline$ of size $inline$\large N$inline$. Let's evenly draw a sample of $inline$\large N$inline$ objects with replacement. This means that we will select a random object of the sample $inline$\large N$inline$ times (we assume that each object is taken with equal probability $inline$\large \frac{1}{N}$inline$)), and each time we choose from all of the original $inline$\large N$inline$ objects. One can imagine a bag from which we pull beads: on each step, selected bead is returned to the bag, and the next selection is made with equal probability from the same number of beads. Note that due to replacement there will be duplicates. Let's label the new sample $inline$\large X_1$inline$. By repeating this procedure $inline$\large M$inline$ times we'll generate $inline$\large M$inline$ sub-samples $inline$\large X_1, \dots, X_M$inline$. Now we have a sufficiently large number of samples, and can evaluate various statistics of the initial distribution.

![image](https://github.com/Yorko/mlcourse_open/blob/master/img/bootstrap.jpg?raw=true)

Let's take `telecom_churn` dataset that is already known to you from past lessons of our course. Recall that it is the task of binary classification of customer churn. One of the most important features in the dataset that is the number of calls to the service center that were made by the client. Let's try to vizualize the data and look at the distribution of this feature.

<spoiler title="Code for downloading the data and plotting">
```python
import pandas as pd
from matplotlib import pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 10, 6
import seaborn as sns
%matplotlib inline

telecom_data = pd.read_csv('data/telecom_churn.csv')

fig = sns.kdeplot(telecom_data[telecom_data['Churn'] == False]['Customer service calls'], label = 'Loyal')
fig = sns.kdeplot(telecom_data[telecom_data['Churn'] == True]['Customer service calls'], label = 'Churn')        
fig.set(xlabel=Number of calls, ylabel=Density)    
plt.show()
```
</spoiler>

![image](https://github.com/Yorko/mlcourse_open/blob/master/img/bootstrap_new.png?raw=true)

As you can see, loyal customers make fewer calls to the service center than our former clients. Now it'd be a good idea to estimate the average number of calls done by each group. Since we have little data in the dataset, computing the mean is not quite right, it is better to apply our new knowledge of bootstrap. Let's generate 1000 new subsamples of our general population and do interval estimation of the mean.

<spoiler title="Code for constructing a confidence interval using the bootstrap">
```python
import numpy as np
def get_bootstrap_samples(data, n_samples):
    # Function to generate the sub-samples using bootstrap
    indices = np.random.randint(0, len(data), (n_samples, len(data)))
    samples = data[indices]
    return samples
def stat_intervals(stat, alpha):
    # Function for interval estimation
    boundaries = np.percentile(stat, [100 * alpha / 2., 100 * (1 - alpha / 2.)])
    return boundaries

# Saving datasets on loyal and former clients to separate numpy arrays
loyal_calls = telecom_data[telecom_data['Churn'] == False]['Customer service calls'].values
churn_calls= telecom_data[telecom_data['Churn'] == True]['Customer service calls'].values

# Fix the seed for reproducibility
np.random.seed(0)

# Generate samples using bootstrap and immediately compute mean for each of them
loyal_mean_scores = [np.mean(sample) 
                       for sample in get_bootstrap_samples(loyal_calls, 1000)]
churn_mean_scores = [np.mean(sample) 
                       for sample in get_bootstrap_samples(churn_calls, 1000)]

#  Print interval estimate of the mean
print("Service calls from loyal:  mean interval",  stat_intervals(loyal_mean_scores, 0.05))
print("Service calls from churn:  mean interval",  stat_intervals(churn_mean_scores, 0.05))
```
</spoiler>


As a result, we have found that with 95% probability the average number of calls from loyal customers will lie between 1.40 and 1.50, while our former clients called on average 2.06-2.40 times. It is also worth noting that the interval for loyal customers is narrower, which is quite logical, as they call rarely (mostly 0, 1 or 2 times), and dissatisfied customers will call more often, but over time their patience is over, and they will change the operator.

### Bagging


Now that you have an idea of ​​bootstrap we can proceed to bagging intself. Suppose we have a training set $inline$\large X$inline$. Let's generate from samples $inline$\large X_1, \dots, X_M$inline$ using bootstrap. Now, on each sample let's train its own classifier $inline$\large a_i(x)$inline$. The final classifier will average responses of all these algorithms (in the case of classification, this corresponds to a vote): $inline$\large a(x) = \frac{1}{M}\sum_{i = 1}^M a_i(x)$inline$. This scheme can be represented by the picture below.

![image](https://github.com/Yorko/mlcourse_open/blob/master/img/bagging.png?raw=true)

Let's consider regression problem with basic algorithms $inline$\large b_1(x), \dots , b_n(x)$inline$. Assume that there is a true response function for all objects $inline$\large y(x)$inline$, and the distribution is set on objects $inline$\large p(x)$inline$. In this case, we can write the error of each regression function $$display$$ \large \varepsilon_i(x) = b_i(x) − y(x),  i = 1, \dots, n$$display$$ and record the expectation of mean square error $$display$$ \large E_x(b_i(x) − y(x))^{2} = E_x \varepsilon^2 (x). $$display$$

The average error of the constructed regression functions has the form $$display$$ \large E_1 = \frac{1}{n}E_x\varepsilon_i^{2}(x) $$display$$

Suppose that the errors are uncorrelated and unbiased: 

$$display$$ \large \begin{array}{rcl} E_x\varepsilon_i(x) &=& 0, \\
E_x\varepsilon_i(x)\varepsilon_j(x) &=& 0, i \neq j. \end{array}$$display$$

Let's now construct a new regression function that will average responses from constructed functions:
$$display$$ \large a(x) = \frac{1}{n}\sum_{i=1}^{n}b_i(x) $$display$$

Let's find its mean square error:

$$display$$ \large \begin{array}{rcl}E_n &=& E_x\Big(\frac{1}{n}\sum_{i=1}^{n}b_i(x)-y(x)\Big)^2 \\
&=& E_x\Big(\frac{1}{n}\sum_{i=1}^{n}\varepsilon_i\Big)^2 \\
&=& \frac{1}{n^2}E_x\Big(\sum_{i=1}^{n}\varepsilon_i^2(x) + \sum_{i \neq j}\varepsilon_i(x)\varepsilon_j(x)\Big) \\
&=& \frac{1}{n}E_1\end{array}$$display$$

Thus, averaging the responses reduced the average square error by n times!

A reminder from our [previous]() lesson about how the error is factorized:
$$display$$\large \begin{array}{rcl} 
\text{Err}\left(\vec{x}\right) &=& \mathbb{E}\left[\left(y - \hat{f}\left(\vec{x}\right)\right)^2\right] \\
&=& \sigma^2 + f^2 + \text{Var}\left(\hat{f}\right) + \mathbb{E}\left[\hat{f}\right]^2 - 2f\mathbb{E}\left[\hat{f}\right] \\
&=& \left(f - \mathbb{E}\left[\hat{f}\right]\right)^2 + \text{Var}\left(\hat{f}\right) + \sigma^2 \\
&=& \text{Bias}\left(\hat{f}\right)^2 + \text{Var}\left(\hat{f}\right) + \sigma^2
\end{array}$$display$$

Bagging reduces the variance of the classifier by reducing the amount by which the error will differ if the model is trained on different sets of data, or in other words, prevents overfitting. Bagging efficiency is achieved due to the fact that the basic algorithms that are trained on different subsamples appear to be quite different, and their errors cancel each other out in voting. Also outliers will not get into some of the training sets.
 
`scikit-learn` library has an implementation of `BaggingRegressor` and `BaggingClassifier` which allow you to use most of the other algorithms "inside". Let's look how bagging works in practice and compare it with the decision tree using the example from [documentation](http://scikit-learn.org/stable/auto_examples/ensemble/plot_bias_variance.html#sphx-glr-auto-examples-ensemble-plot-bias-variance-py).

![image](https://github.com/Yorko/mlcourse_open/blob/master/img/tree_vs_bagging.png?raw=true)

Error of decision tree
$$display$$ \large 0.0255 (Err) = 0.0003 (Bias^2)  + 0.0152 (Var) + 0.0098 (\sigma^2) $$display$$
Error of bagging
$$display$$ \large 0.0196 (Err) = 0.0004 (Bias^2)  + 0.0092 (Var) + 0.0098 (\sigma^2) $$display$$

The plot and the results above show that the error of the variance is much less with bagging, as we have shown theoretically above. 

Bagging is effective on small samples when exclusion of even a small part of training objects leads to the construction of significantly different base classifiers. In the case of large samples, it's common to generate significantly smaller subsamples.
 
It should be noted that the above example cannot not put into practice, because we made the assumption that errors are uncorrelated, and that is rarely fulfilled. If this assumption is wrong, the error reduction is not so significant. In the next parts we will look at more complex methods of combining algorithms in compositions, which allow to achieve high quality in real problems.

### Out-of-bag error

Looking ahead, we'll note that when using random forests there's no need in cross-validation or in a separate test set to get an unbiased evaluation of the test sets error. Let's see how the "internal" evaluation is obtained during the model training.
 
Each tree is constructed using different bootstrap samples from the original data. Approximately 37% of the examples remain outside the bootstrap sample and are not used in the construction of the k-th tree.
 
It is easy to prove: let the sample have $inline$\large \ell$inline$ objects. At each step, all objects fall into the sub-sample with replacement equiprobably, ie each object with probability $inline$\large\frac{1}{\ell}.$inline$. The probability that the object will not appear in the subsample (ie, it wasn't drawn $inline$\large \ell$inline$ times): $inline$\large (1 - \frac{1}{\ell})^\ell$inline$. When $inline$\large \ell \rightarrow +\infty$inline$ we get one of the "remarkable" limits $inline$\large \frac{1}{e}$inline$. Then the probability for a specific object to be drawn to a subsample is $inline$\large \approx  1 - \frac{1}{e} \approx 63\%$inline$.
 
Let's see how this works in practice: 

![image](https://github.com/Yorko/mlcourse_open/blob/master/img/oob.png?raw=true)
The figure shows the estimation of OOB-error. The top image is our initial sample, we divide it into training (left) and test (right) sets. On the right figure, we have a grid of squares that perfectly divides our sample. Now we need to estimate the proportion of correct answers on our test sample. The figure shows that our classifier was mistaken in 4 cases which were not used in training. So, the proportion of correct answers of our classifier is $inline$\large \frac{11}{15}*100\% = 73.33\%$inline$
 
It turns out, that each basic algorithm is trained on ~ 63% of the original objects. So it can be checked right away on the remaining ~ 37%. Out-of-Bag estimate is the average score of the basic algorithms on these ~ 37% of the data on which they were not trained.

## 2. Random forest

Leo Breiman found application for bootstrap not only in statistics, but also in machine learning. He, along with Adel Cutler improved random forest algorithm, proposed by [Ho](http://ect.bell-labs.com/who/tkh/publications/papers/odt.pdf), by adding to the original version construction of uncorrelated trees based on CART, in combination with the method and random subspaces and bagging.
 
Decision trees are good for the family of base classifiers bagging because they are quite complicated and can achieve zero error for any sample. Random subspace method can reduce the correlation between the trees and avoid overfitting. Base algorithms are trained on different subsets of feature vectors which are also allocated randomly.
The ensemble of models using the method of random subspace can be constructed using the following algorithm:
1. Let the number of objects for training be equal to $inline$\large N$inline$, and the number of features $inline$\large D$inline$.
2. Select $inline$\large L$inline$ as a number of individual models in the ensemble.
3. For each model $inline$\large l$inline$, select $inline$\large dl (dl < D) $inline$ as the number of features for $inline$\large l$inline$. Usually, for all models only one value $inline$\large dl$inline$. is used.
4. For each model $inline$\large l$inline$ create a training set by choosing $inline$\large dl$inline$ features from $inline$\large D$inline$ and train a model.
5. Now, to apply the ensemble model to a new object, combine the results of individual $inline$\large L$inline$ models by majority voting, or by combining the posterior probabilities.


### Algorithm

An algorithm for constructing a random forest consisting of $inline$\large N$inline$ trees is as follows:
* For each $inline$\large n = 1, \dots, N$inline$:
     * Generate sample $inline$\large X_n$inline$ via bootstrap;
     * Build a decision tree $inline$\large b_n$inline$ on sample $inline$\large X_n$inline$:
         — we choose the best feature according to a defined criteria, do the partition in a tree and do so until the exhaustion of the sample
         — the tree is constructed until there's no more than $inline$\large n_\text{min}$inline$ objects in each leaf, or until you reach a certain height of the tree
         — for each partition, $inline$\large m$inline$ random features are selected first from $inline$\large n$inline$ s initial features, 
         and the optimal division of the sample is searched only among them.
         
The final classifier is $inline$\large a(x) = \frac{1}{N}\sum_{i = 1}^N b_i(x)$inline$, in simple words, for classification tasks we choose the decision by the majority vote, and in the regression problem - by average.

It is recommended to take $inline$\large m = \sqrt{n}$inline$ for classification problems, and $inline$\large m = \frac{n}{3}$inline$ for regression problems, where $inline$\large n$inline$ is the number of features. For classification, it is also recommended to build each tree until each leaf has one object, and for regression problems - until each leaf has five objects.
 
Thus, random forest is bagging on such decision trees, where for each partition features are selected from a random subset of features.

### Comparison With the Decision Tree and Bagging

<spoiler title="Code for comparing the decision tree, random forest and bagging for the regression problem">
```python
from __future__ import division, print_function
# Disable any warnings from Anaconda
import warnings
warnings.filterwarnings('ignore')
%pylab inline
np.random.seed(42)
figsize(8, 6)
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
     
n_train = 150        
n_test = 1000       
noise = 0.1

# Generate data
def f(x):
    x = x.ravel()
    return np.exp(-x ** 2) + 1.5 * np.exp(-(x - 2) ** 2)

def generate(n_samples, noise):
    X = np.random.rand(n_samples) * 10 - 5
    X = np.sort(X).ravel()
    y = np.exp(-X ** 2) + 1.5 * np.exp(-(X - 2) ** 2)\
        + np.random.normal(0.0, noise, n_samples)
    X = X.reshape((n_samples, 1))

    return X, y

X_train, y_train = generate(n_samples=n_train, noise=noise)
X_test, y_test = generate(n_samples=n_test, noise=noise)

# One decision tree regressor
dtree = DecisionTreeRegressor().fit(X_train, y_train)
d_predict = dtree.predict(X_test)

plt.figure(figsize=(10, 6))
plt.plot(X_test, f(X_test), "b")
plt.scatter(X_train, y_train, c="b", s=20)
plt.plot(X_test, d_predict, "g", lw=2)
plt.xlim([-5, 5])
plt.title("Решающее дерево, MSE = %.2f" 
          % np.sum((y_test - d_predict) ** 2))

# Bagging decision tree regressor
bdt = BaggingRegressor(DecisionTreeRegressor()).fit(X_train, y_train)
bdt_predict = bdt.predict(X_test)

plt.figure(figsize=(10, 6))
plt.plot(X_test, f(X_test), "b")
plt.scatter(X_train, y_train, c="b", s=20)
plt.plot(X_test, bdt_predict, "y", lw=2)
plt.xlim([-5, 5])
plt.title("Bagging of decision trees, MSE = %.2f" % np.sum((y_test - bdt_predict) ** 2));

# Random Forest
rf = RandomForestRegressor(n_estimators=10).fit(X_train, y_train)
rf_predict = rf.predict(X_test)

plt.figure(figsize=(10, 6))
plt.plot(X_test, f(X_test), "b")
plt.scatter(X_train, y_train, c="b", s=20)
plt.plot(X_test, rf_predict, "r", lw=2)
plt.xlim([-5, 5])
plt.title("Random forest, MSE = %.2f" % np.sum((y_test - rf_predict) ** 2));
```
</spoiler>
![image](https://github.com/Yorko/mlcourse_open/blob/master/img/tree_mse.png?raw=true)

![image](https://github.com/Yorko/mlcourse_open/blob/master/img/bagging_mse.png?raw=true)

![image](https://github.com/Yorko/mlcourse_open/blob/master/img/rf_mse.png?raw=true)

As we can see from the graphs and MSE error values, a random forest of 10 trees gives better results than a single tree or bagging on 10 decision trees. The main difference between random forest and bagging on decision trees is that random forest randomly selects a subset of features, and the best feature for the node partition is selected from this subset, unlike bagging where all functions are considered for the partition of the node.
 
You can also see the benefit of random forest and bagging in classification problems.

<spoiler title="Code for comparing the decision tree, bagging and random forest for classification task">
```python
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_circles
from sklearn.cross_validation import train_test_split
import numpy as np
from matplotlib import pyplot as plt
plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = 10, 6
%matplotlib inline

np.random.seed(42)
X, y = make_circles(n_samples=500, factor=0.1, noise=0.35, random_state=42)
X_train_circles, X_test_circles, y_train_circles, y_test_circles = train_test_split(X, y, test_size=0.2)

dtree = DecisionTreeClassifier(random_state=42)
dtree.fit(X_train_circles, y_train_circles)

x_range = np.linspace(X.min(), X.max(), 100)
xx1, xx2 = np.meshgrid(x_range, x_range)
y_hat = dtree.predict(np.c_[xx1.ravel(), xx2.ravel()])
y_hat = y_hat.reshape(xx1.shape)
plt.contourf(xx1, xx2, y_hat, alpha=0.2)
plt.scatter(X[:,0], X[:,1], c=y, cmap='autumn')
plt.title("Дерево решений")
plt.show()

b_dtree = BaggingClassifier(DecisionTreeClassifier(),n_estimators=300, random_state=42)
b_dtree.fit(X_train_circles, y_train_circles)

x_range = np.linspace(X.min(), X.max(), 100)
xx1, xx2 = np.meshgrid(x_range, x_range)
y_hat = b_dtree.predict(np.c_[xx1.ravel(), xx2.ravel()])
y_hat = y_hat.reshape(xx1.shape)
plt.contourf(xx1, xx2, y_hat, alpha=0.2)
plt.scatter(X[:,0], X[:,1], c=y, cmap='autumn')
plt.title("Бэггинг(дерево решений)")
plt.show()

rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train_circles, y_train_circles)

x_range = np.linspace(X.min(), X.max(), 100)
xx1, xx2 = np.meshgrid(x_range, x_range)
y_hat = rf.predict(np.c_[xx1.ravel(), xx2.ravel()])
y_hat = y_hat.reshape(xx1.shape)
plt.contourf(xx1, xx2, y_hat, alpha=0.2)
plt.scatter(X[:,0], X[:,1], c=y, cmap='autumn')
plt.title("Random forest")
plt.show()
```
</spoiler>

![image](https://github.com/Yorko/mlcourse_open/blob/master/img/tree_class.png?raw=true)

![image](https://github.com/Yorko/mlcourse_open/blob/master/img/bagg_class.png?raw=true)

![image](https://github.com/Yorko/mlcourse_open/blob/master/img/rf_class.png?raw=true)

The figure above shows that the dividing boundary of the decision tree is very "jagged" and with a lot of sharp angles, which indicates overfitting and poor generalization capability. At the same time, bagging and random forest have quite smooth boundaries with little indication of overfitting.
 
Let us now try to understand the parameters by tuning which we can increase the proportion of correct answers.

### Parameters


Random forest method is implemented in machine learning library [scikit-learn](http://scikit-learn.org/stable/) with two classes: RandomForestClassifier and RandomForestRegressor.

Full list of parameters for random forest regression problem:

```python
class sklearn.ensemble.RandomForestRegressor(
    n_estimators — the number of trees in the "forest" (by default - 10)
    criterion — a function that measures the quality of split of a tree branch (by default - "mse", also "mae" can be selected)
    max_features — the number of features on which a partition is sought. You can specify a certain number or percentage of features, or choose from the available values: "auto" (all features), "sqrt", "log2". Default value is "auto".
    max_depth — the maximum depth of the tree (by default depth is not limited)
    min_samples_split — minimum number of objects required for the split of the internal node. You can specify the number or percentage of the total number of objects (by default it's 2)
    min_samples_leaf — the minimum number of objects in a leaf. Can be defined by the number or percentage of the total number of objects (by default it's 1)
    min_weight_fraction_leaf — minimum weighted proportion of the total sum of weights (of all input objects) that should be in a leaf (by default, all have the same weight)
    max_leaf_nodes — the maximum number of leaves (by default there is no limit)
    min_impurity_split — threshold to stop building the tree (by default 1е-7)
    bootstrap — whether to apply bootstrap to build a tree (by default True)
    oob_score — whether to use the out-of-bag objects for evaluation of R^2 (by default False) 
    n_jobs — the number of cores to build the model and predictions (by default 1, if you put -1, then all cores will be used)
    random_state — initial value for random number generation (by default there's none, if you want reproducible results, it is necessary to specify any number of int type)
    verbose — log output of tree construction (by default 0)
    warm_start — uses a pretrained model and adds the trees to the ensemble (by default False)
)
```

For classification problem it's almost all the same, we'll present only those parameters that are different from RandomForestRegressor 
```python
class sklearn.ensemble.RandomForestClassifier(
    criterion — as we now have classification problem, by default "gini" is selected (also, "entropy" can be selected)
    class_weight — weight of each class (by default all weights equal to 1, but you can pass a dictionary with weights, or explicitly specify "balanced" to make weight classes equal to their proportions in the general population; you can also specify "balanced_subsample", and then the weights on each subsample will vary depending on the distribution of classes on this subsample.
)
```

Next, let's consider several parameters to look at in the first place when building a model:
- n_estimators — the number of trees in the "forest"
- criterion — the criterion for splitting the sample at the node
- max_features — the number of features on which a partition is sought
- min_samples_leaf — the minimum number of objects in a leaf
- max_depth — the maximum depth of the tree

**Let's consider the use of a random forest in a real problem**

To do this, we will use the example of the problem of customer churn. This is a classification problem, so we will use the accuracy metric to assess the quality of the model. To start with, let's build the simplest classifier which will be our baseline. Let's take only numerical features for simplicity.

<spoiler title="Code to build baseline for random forest">
```python
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score

# Loading data
df = pd.read_csv("../../data/telecom_churn.csv")

# Selecting only columns with numeric data type
cols = []
for i in df.columns:
    if (df[i].dtype == "float64") or (df[i].dtype == 'int64'):
        cols.append(i)
        
# Dividing into features and objects
X, y = df[cols].copy(), np.asarray(df["Churn"],dtype='int8')

# Initializing stratified breakdown of our dataset for validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize our classifier with the default settings
rfc = RandomForestClassifier(random_state=42, n_jobs=-1, oob_score=True)

# Train the model on a training set
results = cross_val_score(rfc, X, y, cv=skf)

# Evaluate the proportion of correct answers on the test set
print("CV accuracy score: {:.2f}%".format(results.mean()*100))
```
</spoiler>
We've obtained a share of 91.21% correct answers, now we will try to improve this result and look at the behavior of the validation curves when changing the basic settings.

Let's start with the number of trees:
<spoiler title="Code for constructing the validation curves on selecting the number of trees">
```python
# Initialize validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create lists to save accuracy on training and test sets
train_acc = []
test_acc = []
temp_train_acc = []
temp_test_acc = []
trees_grid = [5, 10, 15, 20, 30, 50, 75, 100]

# Train on the training set
for ntrees in trees_grid:
    rfc = RandomForestClassifier(n_estimators=ntrees, random_state=42, n_jobs=-1, oob_score=True)
    temp_train_acc = []
    temp_test_acc = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        rfc.fit(X_train, y_train)
        temp_train_acc.append(rfc.score(X_train, y_train))
        temp_test_acc.append(rfc.score(X_test, y_test))
    train_acc.append(temp_train_acc)
    test_acc.append(temp_test_acc)
    
train_acc, test_acc = np.asarray(train_acc), np.asarray(test_acc)
print("Best accuracy on CV is {:.2f}% with {} trees".format(max(test_acc.mean(axis=1))*100, 
                                                        trees_grid[np.argmax(test_acc.mean(axis=1))]))
```
</spoiler>

<spoiler title="Code for plotting validation curves">
```python
import matplotlib.pyplot as plt
plt.style.use('ggplot')
%matplotlib inline

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(trees_grid, train_acc.mean(axis=1), alpha=0.5, color='blue', label='train')
ax.plot(trees_grid, test_acc.mean(axis=1), alpha=0.5, color='red', label='cv')
ax.fill_between(trees_grid, test_acc.mean(axis=1) - test_acc.std(axis=1), test_acc.mean(axis=1) + test_acc.std(axis=1), color='#888888', alpha=0.4)
ax.fill_between(trees_grid, test_acc.mean(axis=1) - 2*test_acc.std(axis=1), test_acc.mean(axis=1) + 2*test_acc.std(axis=1), color='#888888', alpha=0.2)
ax.legend(loc='best')
ax.set_ylim([0.88,1.02])
ax.set_ylabel("Accuracy")
ax.set_xlabel("N_estimators")
```
</spoiler>

![image](https://github.com/Yorko/mlcourse_open/blob/master/img/rf_n_est.png?raw=true)

As you can seen, when a certain number of trees is reached, our share of correct answers on the test goes to the asymptote, and you can decide by yourselves how many trees is optimal for your problem.
The figure also shows that on the training set we were able to achieve 100% accuracy, it indicates overfitting. To avoid overfitting we need to add the regularization parameters to the model.
 
Let's start with the maximum depth parameter max_depth. (let's fix the number of trees at 100)
<spoiler title="Code for building learning curves on the selection of the maximum depth of the tree">
```python
# Create lists to save accuracy on training and test sets
train_acc = []
test_acc = []
temp_train_acc = []
temp_test_acc = []
max_depth_grid = [3, 5, 7, 9, 11, 13, 15, 17, 20, 22, 24]

# Train on the training set
for max_depth in max_depth_grid:
    rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, oob_score=True, max_depth=max_depth)
    temp_train_acc = []
    temp_test_acc = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        rfc.fit(X_train, y_train)
        temp_train_acc.append(rfc.score(X_train, y_train))
        temp_test_acc.append(rfc.score(X_test, y_test))
    train_acc.append(temp_train_acc)
    test_acc.append(temp_test_acc)
    
train_acc, test_acc = np.asarray(train_acc), np.asarray(test_acc)
print("Best accuracy on CV is {:.2f}% with {} max_depth".format(max(test_acc.mean(axis=1))*100, 
                                                        max_depth_grid[np.argmax(test_acc.mean(axis=1))]))
```
</spoiler>

<spoiler title="Code to plot learning curves">
```python
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(max_depth_grid, train_acc.mean(axis=1), alpha=0.5, color='blue', label='train')
ax.plot(max_depth_grid, test_acc.mean(axis=1), alpha=0.5, color='red', label='cv')
ax.fill_between(max_depth_grid, test_acc.mean(axis=1) - test_acc.std(axis=1), test_acc.mean(axis=1) + test_acc.std(axis=1), color='#888888', alpha=0.4)
ax.fill_between(max_depth_grid, test_acc.mean(axis=1) - 2*test_acc.std(axis=1), test_acc.mean(axis=1) + 2*test_acc.std(axis=1), color='#888888', alpha=0.2)
ax.legend(loc='best')
ax.set_ylim([0.88,1.02])
ax.set_ylabel("Accuracy")
ax.set_xlabel("Max_depth")
```
</spoiler>

![image](https://github.com/Yorko/mlcourse_open/blob/master/img/rf_max_depth.png?raw=true)

`max_depth` parameter copes well with the regularization of the model, and we do not overfit so much this time.Share of correct answers of our model slightly increased.
 
Another important parameter is `min_samples_leaf`, it also performs the function of regularizor.

<spoiler title="Code for constructing the validation curves for selection of the minimum number of objects in a single leaf of the tree">
```python
# Create lists to save accuracy on training and test sets
train_acc = []
test_acc = []
temp_train_acc = []
temp_test_acc = []
min_samples_leaf_grid = [1, 3, 5, 7, 9, 11, 13, 15, 17, 20, 22, 24]

# Train on the training set
for min_samples_leaf in min_samples_leaf_grid:
    rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, 
                                 oob_score=True, min_samples_leaf=min_samples_leaf)
    temp_train_acc = []
    temp_test_acc = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        rfc.fit(X_train, y_train)
        temp_train_acc.append(rfc.score(X_train, y_train))
        temp_test_acc.append(rfc.score(X_test, y_test))
    train_acc.append(temp_train_acc)
    test_acc.append(temp_test_acc)
    
train_acc, test_acc = np.asarray(train_acc), np.asarray(test_acc)
print("Best accuracy on CV is {:.2f}% with {} min_samples_leaf".format(max(test_acc.mean(axis=1))*100, 
                                                        min_samples_leaf_grid[np.argmax(test_acc.mean(axis=1))]))
```
</spoiler>

<spoiler title="Code for plotting validation curves">
```python
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(min_samples_leaf_grid, train_acc.mean(axis=1), alpha=0.5, color='blue', label='train')
ax.plot(min_samples_leaf_grid, test_acc.mean(axis=1), alpha=0.5, color='red', label='cv')
ax.fill_between(min_samples_leaf_grid, test_acc.mean(axis=1) - test_acc.std(axis=1), test_acc.mean(axis=1) + test_acc.std(axis=1), color='#888888', alpha=0.4)
ax.fill_between(min_samples_leaf_grid, test_acc.mean(axis=1) - 2*test_acc.std(axis=1), test_acc.mean(axis=1) + 2*test_acc.std(axis=1), color='#888888', alpha=0.2)
ax.legend(loc='best')
ax.set_ylim([0.88,1.02])
ax.set_ylabel("Accuracy")
ax.set_xlabel("Min_samples_leaf")
```
</spoiler>
![image](https://github.com/Yorko/mlcourse_open/blob/master/img/rf_leaf.png?raw=true)

In this case, we do not win in accuracy on validation, but we can greatly reduce overfitting up to 2%, while maintaining the accuracy of about 92%.
 
Let us consider a parameter called `max_features`. For classification purposes the default value $inline$\large \sqrt{n}$inline$ is used, where n is the number of features. Let's see whether in our case it's optimal to use 4 features or not.

<spoiler title="Code for constructing validation curves for selection of the maximum number of features for a single tree">
```python
# Create lists to save accuracy on training and test sets
train_acc = []
test_acc = []
temp_train_acc = []
temp_test_acc = []
max_features_grid = [2, 4, 6, 8, 10, 12, 14, 16]

# Train on the training set
for max_features in max_features_grid:
    rfc = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, 
                                 oob_score=True, max_features=max_features)
    temp_train_acc = []
    temp_test_acc = []
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        rfc.fit(X_train, y_train)
        temp_train_acc.append(rfc.score(X_train, y_train))
        temp_test_acc.append(rfc.score(X_test, y_test))
    train_acc.append(temp_train_acc)
    test_acc.append(temp_test_acc)
    
train_acc, test_acc = np.asarray(train_acc), np.asarray(test_acc)
print("Best accuracy on CV is {:.2f}% with {} max_features".format(max(test_acc.mean(axis=1))*100, 
                                                        max_features_grid[np.argmax(test_acc.mean(axis=1))]))
```
</spoiler>
<spoiler title="Code for plotting validation curves">
```python
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(max_features_grid, train_acc.mean(axis=1), alpha=0.5, color='blue', label='train')
ax.plot(max_features_grid, test_acc.mean(axis=1), alpha=0.5, color='red', label='cv')
ax.fill_between(max_features_grid, test_acc.mean(axis=1) - test_acc.std(axis=1), test_acc.mean(axis=1) + test_acc.std(axis=1), color='#888888', alpha=0.4)
ax.fill_between(max_features_grid, test_acc.mean(axis=1) - 2*test_acc.std(axis=1), test_acc.mean(axis=1) + 2*test_acc.std(axis=1), color='#888888', alpha=0.2)
ax.legend(loc='best')
ax.set_ylim([0.88,1.02])
ax.set_ylabel("Accuracy")
ax.set_xlabel("Max_features")
```
</spoiler>
![image](https://github.com/Yorko/mlcourse_open/blob/master/img/rf_max_features.png?raw=true)

In our case the optimal number of features is 10, this is the value where the best result is achieved.
 
We have examined the behavior of the validation curves depending on changes in the basic parameters. Let's now use `GridSearchCV` to find optimal parameters for our example.

<spoiler title="Code for selecting the optimal model parameters">
```python
# Make the initialization of parameters on which we want to do an exhaustive search
parameters = {'max_features': [4, 7, 10, 13], 'min_samples_leaf': [1, 3, 5, 7], 'max_depth': [5,10,15,20]}
rfc = RandomForestClassifier(n_estimators=100, random_state=42, 
                             n_jobs=-1, oob_score=True)
gcv = GridSearchCV(rfc, parameters, n_jobs=-1, cv=skf, verbose=1)
gcv.fit(X, y)
```
</spoiler>
The best proportion of correct answers that we were able to achieve by grid search of parameters is 92.83% with `'max_depth': 15, 'max_features': 7, 'min_samples_leaf': 3`. 

### Variation and Decorrelation Effect

Let's write the variance of a random forest as $$display$$ \large Varf(x) = \rho(x)\sigma^2(x) $$display$$
Here  
- $inline$ \large \rho(x)$inline$ – is sample correlation between any two trees used in the averaging
$$display$$ \large \rho(x) = corr[T(x;\Theta_1(Z)),T(x_2,\Theta_2(Z))], $$display$$ where $inline$ \large \Theta_1(Z) $inline$ and $inline$ \large  \Theta_2(Z) $inline$ are randomly selected pair of trees on a randomly selected objects of sample $inline$ \large Z$inline$
- $inline$ \large \sigma^2(x)$inline$ is the sample variance of any randomly selected tree: $$display$$ \large \sigma^2(x) = VarT(x;\Theta(X)$$display$$

It's easy to confuse $inline$ \large \rho(X) $inline$ with an average correlation between the trained trees in a random forest, while considering trees as N-vectors and calculating the average pair correlation between them. This is not the case. This conditional correlation is not directly related to the averaging process, and the dependence on $inline$ \large x$inline$ in $inline$ \large \rho(x)$inline$ warns us of this distinction. Rather $inline$ \large \rho(x)$inline$ is the theoretical correlation between a pair of random trees evaluated in the object $inline$ \large x$inline$, which was caused by repeated sampling of the training set from the general population $inline$ \large Z $inline$, and thereafter this pair of random trees was selected. On the statistical jargon, this is the correlation caused by the sampling distribution of $inline$ \large Z $inline$ and $inline$ \large \Theta $inline$.

In fact, the conditional covariance of pairs of trees is equal to 0 because bootstrap and feature selection are independent and identically distributed.

If we consider the variance of one tree, it practically does not change depending on the variables for the split ($inline$ \large m $inline$), but it is crucial for ensemble, and the variance of the tree is much higher than for the ensemble.
The book *The Elements of Statistical Learning* (Trevor Hastie , Robert Tibshirani and Jerome Friedman) has an excellent example that demonstrates this.
![image](https://github.com/Yorko/mlcourse_open/blob/master/img/variance_rf.png?raw=true)

### Bias

As in bagging, bias in a random forest is the same as bias in a single tree $inline$ \large T(x,\Theta(Z))$inline$:
$$display$$ \large \begin{array}{rcl} Bias &=& \mu(x) - E_Zf_{rf}(x) \\
&=& \mu(x) - E_ZE_{\Theta | Z}T(x,\Theta(Z))\end{array}$$display$$
It is also usually bigger (in absolute value) than the bias of the unprunned tree because randomization and reduction of the sample space impose restrictions. Therefore, improvement in the prediction obtained by bagging or random forests is solely the result of decreasing the dispersion.

###  Extremely Randomized Trees

Extremely Randomized Trees are more random in way that they compute partitions in the nodes. As in random forests, a random subset of possible features is used, but instead of searching the optimal threshold, the threshold values ​​are selected at random for each possible feature, and the best of these randomly generated thresholds is selected as the best rule to split the node. This usually allows to slightly reduce dispersion of the model due to a slightly larger increase in bias.

In scikit-learn library there are implementations [ExtraTreesClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn.ensemble.ExtraTreesClassifier) and [ExtraTreesRegressor](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html#sklearn.ensemble.ExtraTreesRegressor).This method should be used when you strongly overfit with random forest or gradient boosting. 

### The Similarity of the Random Forest and k-Nearest Neighbors Algorithm

Random forest method is similar to the method of nearest neighbors. Random forests, in fact, carry out predictions for objects based on labels of similar objects from the training set. And the more often these objects appear in the same leaf, the greater their similarity. Let's show this formally.
 
Let's consider regression problem with a quadratic loss function. Let $inline$ \large T_n(x) $inline$ be the number of leaf of $inline$ \large n(x)$inline$-th tree of a random forest, into which the object $inline$ \large x $inline$ falls. Response for object $inline$ \large x $inline$ is the average response for all objects in the training sample that fall into this leaf $inline$ \large T_n(x) $inline$. It can be written as
$$display$$ \large b_n(x) = \sum_{i=1}^{l}w_n(x,x_i)y_i,$$display$$ 
where $$display$$ \large w_n(x, x_i) = \frac{[T_n(x) = T_n(x_i)]}{\sum_{j=1}^{l}[T_n(x) = T_n(x_j)]}$$display$$
Then the response of the composition is $$display$$ \large \begin{array}{rcl} a_n(x) &=& \frac{1}{N}\sum_{n=1}^{N}\sum_{i=1}^{l}w_n(x,x_i)y_i \\
&=& \sum_{i=1}^{l}\Big(\frac{1}{N}\sum_{j=1}^{N}w_n(x,x_j)\Big)y_i \end{array}$$display$$
It's seen that the response of a random forest is the sum of the responses of all the training objects with some weights. Note that the leaf number $inline$ \large T_n(x)$inline$ where the object fell into is in itself a valuable feature. An approach where a composition of a small number of trees is trained using random forest or gradient boosting, and then categorical features $inline$ \large T_1(x), \dots, T_n(x) $inline$ are added thereto, works quite well. New features are the result of non-linear decomposition of space and carry information about the similarity of objects.
 
The same book The Elements of Statistical Learning has a good illustrative example of similarity of random forest and k-nearest neighbors.

![image](https://github.com/Yorko/mlcourse_open/blob/master/img/knn_vs_rf.png?raw=true)

### Converting Features into a Multidimensional Space

Everyone got used to applying random forest for supervised learning problems, but there's also a way to train the model without a teacher. Using the method [RandomTreesEmbedding](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomTreesEmbedding.html#sklearn.ensemble.RandomTreesEmbedding) we can transform our dataset into a its sparse multi-dimensional representation.Its essence is that we are building a completely random trees, and an index of the leaf where an observation fell into is considered a new feature. If an object fell into the first leaf, we put 1, if not we put 0. The so-called binary encoding. We can control the number of variables and the degree of sparseness of our new representation of the dataset by increasing/decreasing the number of trees and their depth. Since the adjacent data points are likely to lie in one leaf, the transformation does an implicit nonparametric density estimate.

#3. Evaluation of the Feature Importance

Very often you want to understand your algorithm, why it responded this way and not the other. Or if not completely understand it, then at least find out which of all the variables have a greater impact on the result. One can quite easily obtain this information from random forest.

### Essence of the Method

According to this picture, it is intuitively clear that the importance of the "Age" feature is greater than the importance of the "income" feature in the problem of credit scoring. It is formalized by means of the information gain concept..
<img src="https://habrastorage.org/getpro/habr/post_images/32d/465/71f/32d46571feaa7835080ac980bcf685ae.gif" align='center'>

If you build a lot of decision trees (random forest), then the higher in average the feature is in the decision tree, the more important it is in this problem of classification/regression. With each partition for each tree an improvement in the split criterion (in our case Gini impurity) is a measure of the importance that is associated with a partition feature, and accumulates in all the trees of the forest separately for each variable.
 
Let's get into the details. The average decrease in accuracy caused by the feature is determined during the calculation of out-of-bag error. The greater the accuracy of predictions decreases because of an exclusion (or rearrangement) of one feature, the more important this feature is, and therefore features with greater average reduction in accuracy are more important for classification of data. Mean reduction of Gini uncertainty (or mse in regression) is a measure of how each variable contributes to uniformity of leaves and nodes in the final random forest model. Each time when a single feature is used to split the node, Gini uncertainty is calculated for the child nodes and compared to the coefficient of the initial node. Gini uncertainty is a measure of uniformity from 0 (uniform) to 1 (heterogeneous). Changes in the value of the separation criteria are summed for each feature and normalized at the end of the computation. Features that lead to the nodes with a higher purity have a higher reduction in the Gini coefficient.

Now let's represent all of the above in the form of formulas.
$$display$$ \large VI^{T} = \frac{\sum_{i \in \mathfrak{B}^T}I \Big(y_i=\hat{y}_i^{T}\Big)}{\Big |\mathfrak{B}^T\Big |} - \frac{\sum_{i \in \mathfrak{B}^T}I \Big(y_i=\hat{y}_{i,\pi_j}^{T}\Big)}{\Big |\mathfrak{B}^T\Big |} $$display$$

$inline$ \large \hat{y}_i^{(T)} = f^{T}(x_i)  $inline$ is prediction of class before rearrangement/deletion of feature
$inline$ \large \hat{y}_{i,\pi_j}^{(T)} = f^{T}(x_{i,\pi_j})   $inline$ is prediction of class after rearrangement/deletion of feature
$inline$ \large x_{i,\pi_j} = (x_{i,1}, \dots , x_{i,j-1}, \quad x_{\pi_j(i),j}, \quad x_{i,j+1}, \dots , x_{i,p})$inline$
Note that $inline$ \large VI^{(T)}(x_j) = 0 $inline$ if $inline$ \large X_j $inline$ is not in the tree $inline$ \large T $inline$ 

Calculation of the feature importance in the ensemble:
— unnormalized 
$$display$$ \large VI(x_j) = \frac{\sum_{T=1}^{N}VI^{T}(x_j)}{N} $$display$$

— normalized 
$$display$$ \large z_j = \frac{VI(x_j)}{\frac{\hat{\sigma}}{\sqrt{N}}} $$display$$

**Example**

Let's consider the results of the survey of visitors of hostels from Booking.com and TripAdvisor.com. Features are the average grades on various factors (listed below), like staff, state of the rooms, etc. Target variable is hostel rank on the site.

<spoiler title="Code to evaluate the feature importance">
```python
from __future__ import division, print_function
# Disable any warnings from Anaconda
import warnings
warnings.filterwarnings('ignore')
%pylab inline
import seaborn as sns
# russian headres
from matplotlib import rc
font = {'family': 'Verdana',
        'weight': 'normal'}
rc('font', **font)
import pandas as pd
import numpy as np
from sklearn.ensemble.forest import RandomForestRegressor

hostel_data = pd.read_csv("../../data/hostel_factors.csv")
features = {"f1":u"Staff",
"f2":u"Reservation of the Hostel",
"f3":u"Check-in and check-out of the hostel",
"f4":u"Room Condition",
"f5":u"Condition of the shared kitchen",
"f6":u"Condition of the common space",
"f7":u"Extras",
"f8":u"General terms and convenience",
"f9":u"price/quality",
"f10":u"SCC"}

forest = RandomForestRegressor(n_estimators=1000, max_features=10,
                                random_state=0)

forest.fit(hostel_data.drop(['hostel', 'rating'], axis=1), 
           hostel_data['rating'])
importances = forest.feature_importances_

indices = np.argsort(importances)[::-1]
# Plot the feature importances of the forest
num_to_plot = 10
feature_indices = [ind+1 for ind in indices[:num_to_plot]]

# Print the feature ranking
print("Feature ranking:")
  
for f in range(num_to_plot):
    print("%d. %s %f " % (f + 1, 
            features["f"+str(feature_indices[f])], 
            importances[indices[f]]))
plt.figure(figsize=(15,5))
plt.title(u"The importance of the constructs")
bars = plt.bar(range(num_to_plot), 
               importances[indices[:num_to_plot]],
       color=([str(i/float(num_to_plot+1)) 
               for i in range(num_to_plot)]),
               align="center")
ticks = plt.xticks(range(num_to_plot), 
                   feature_indices)
plt.xlim([-1, num_to_plot])
plt.legend(bars, [u''.join(features["f"+str(i)]) 
                  for i in feature_indices]);
```
</spoiler>
![image](https://github.com/Yorko/mlcourse_open/blob/master/img/rf_features_imp.png?raw=true)
The figure above shows that people are more likely to pay attention to personnel and the quality/price ratio, and write their reviews based on these things. But the difference between these features and less important features is not very significant, and discarding of some feature will reduce the accuracy of our model. But even on the basis of our analysis, we can make recommendations to hotels to better train personnel and/or to improve the quality of the declared price in the first place. 

#4. Pros and Cons of Random Forest
**Pros**:
— it has a high prediction accuracy, and for most tasks it will perform better than linear algorithms; accuracy is comparable to the accuracy of boosting
— almost insensitive to outliers in the data due to random sampling
— insensitive to scaling (and in general to any monotonic transformations) of features, it is ​​associated with the choice of the random subspaces
— does not require careful parameter tuning, works well "out of the box." By "tuning" parameters one can achieve a gain of 0.5 to 3% in accuracy, depending on the task and data
— it is able to efficiently process data with a large number of features and classes
— equally well processes both continuous and discrete variables
— is rarely overfitted; in practice, the addition of trees almost always only improves the composition, but in validation, after reaching a certain number of trees, the learning curve goes to the asymptote
— for random forest, there exist methods for evaluating the significance of individual features in the model
— works well with missing data; it maintains good accuracy even if most of the data is missing
— implies the possibility to balance the weight of each class either on the whole sample, or on the subsample for each tree
— calculates the proximity between pairs of objects that can be used for clustering, anomaly detection or (by scaling) provide an interesting representation of the data
— features described above may be extended to the untagged data, which leads to the possibility of making data visualizations and clustering to detect anomalies
— high parallelization and scalability.
 
**Cons**:
— as opposed to a single tree, the results of random forest are difficult to interpret
— no formal conclusions (p-values) are available to assess the importance of the variables
— algorithm is worse than many linear methods when the sample has many sparse features (texts, Bag of words)
— Random Forest can not extrapolate, in contrast to the linear regression (but this can be considered a plus, as in case of an outlier there won't be extreme values)
— the algorithm tends to overfit in some problems, especially with noisy data
— for data including categorical variables with different number of levels, random forests are biased in favor of features with many levels: when a feature has a lot of levels, the tree will try harder to adapt to this feature, because it can give a higher value of the optimized functional (like information gain)
— if the data contains groups of correlated features of similar importance to the labels, preference is given to small groups before large
— larger size of the resulting models. It takes $inline$\large O(NK) $inline$ memory for storing a model where $inline$\large K $inline$ is the number of trees.

#5. Homework
In this home [assignment](https://github.com/Yorko/mlcourse_open/blob/master/jupyter_notebooks/topic5_bagging_rf/hw5_logit_rf_credit_scoring.ipynb), we will solve the problem of credit scoring. We’ll see in practice, how and where to apply bootstrap, what are the advantages and disadvantages of logistic regression in comparison with random forest. We see also see in which cases bagging works better, and in which - worse. You can find answers to many questions if you read this article carefully.

Answers should be filled in in the [web-form](https://goo.gl/forms/wNLR2QJj0ZqQ7B9q1). You can get maximum 12 points for this assignment. Дедлайн – 3 апреля 23:59 UTC+3.

#6. Useful resources
 – section 15 of the book “[Elements of Statistical Learning](https://statweb.stanford.edu/~tibs/ElemStatLearn/)” Jerome H. Friedman, Robert Tibshirani, and Trevor Hastie
– [Блог](https://alexanderdyakonov.wordpress.com/2016/11/14/случайный-лес-random-forest/) Александра Дьяконова
– more on practical applications of random forest and other composition algorithms in official documentation of [scikit-learn](http://scikit-learn.org/stable/modules/ensemble.html)
– [Курс](https://github.com/esokolov/ml-course-hse) Евгения Соколова по машинному обучению (материалы на GitHub). Есть дополнительные практические задания для углубления ваших знаний
– Обзорная [статья](https://www.researchgate.net/publication/278019662_Istoria_razvitia_ansamblevyh_metodov_klassifikacii_v_masinnom_obucenii) "История развития ансамблевых методов классификации в машинном обучении" (Ю. Кашницкий)

*Статья написана в соавторстве с @yorko (Юрием Кашницким). Автор домашнего задания – @vradchenko (Виталий Радченко). Благодарю @bauchgefuehl (Анастасию Манохину) за редактирование.*

