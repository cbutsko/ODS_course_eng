<img src="https://habrastorage.org/files/62b/42b/b53/62b42bb533f44cbaa8d6306332512555.png" align="right" width="320"/>

Greetings to everyone who started the course! Welcome to the new members/newcomers! The second part is about data visualization in Python. First, we'll look at the basic methods of the Seaborn and Plotly libraries, then we’ll analyze the dataset on the telecom operator customers churn rate that we had in the [first article](https://habrahabr.ru/company/ods/blog/322626/) and peek into the n-dimensional space using the t-SNE algorithm.

Напомним, что к курсу еще можно подключиться, дедлайн по 1 домашнему заданию – 6 марта 23:59.

This article will be much longer. Ready? Let’s go!
 <cut/>

<spoiler title="Список статей серии">
1. [Первичный анализ данных с Pandas](https://habrahabr.ru/company/ods/blog/322626/)
2. [Визуальный анализ данных c Python](https://habrahabr.ru/company/ods/blog/323210/)
3. [Классификация, деревья решений и метод ближайших соседей](https://habrahabr.ru/company/ods/blog/322534/)
4. [Линейные модели классификации и регрессии](https://habrahabr.ru/company/ods/blog/323890/)
5. [Композиции: бэггинг, случайный лес](https://habrahabr.ru/company/ods/blog/324402/)
6. [Построение и отбор признаков. Приложения в задачах обработки текста, изображений и геоданных](https://habrahabr.ru/company/ods/blog/325422/)
7. Обучение без учителя: PCA, кластеризация, поиск аномалий
</spoiler>

## Plan of this article

- [Demonstration of Seaborn and Plotly main methods](https://habrahabr.ru/company/ods/blog/323210/#demonstraciya-osnovnyh-metodov-seaborn-i-plotly)
- [Example of visual analysis of data](https://habrahabr.ru/company/ods/blog/323210/#primer-vizualnogo-analiza-dannyh)
- [Peeking into n-dimensional space with t-SNE](https://habrahabr.ru/company/ods/blog/323210/#podglyadyvanie-v-n-mernoe-prostranstvo-s-t-sne)
- [Homework #2](https://habrahabr.ru/company/ods/blog/323210/#domashnee-zadanie--2)
- [Useful links and resources](https://habrahabr.ru/company/ods/blog/323210/#obzor-poleznyh-resursov)


# Demonstration of Seaborn and Plotly main methods
For starters, let’s set up the environment: import all the necessary libraries and adjust the default display of pictures a little.

```python
# Python 2 and 3 compatibility
# pip install future
from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
# turn off Anaconda warmnings
import warnings
warnings.simplefilter('ignore')

# we’ll display plots right in jupyter
%pylab inline

# increase default plot size
from pylab import rcParams
rcParams['figure.figsize'] = 8, 5
import pandas as pd
import seaborn as sns
```

After that let’s load the data we’ll be working with into a data frame. As an example I chose data on sales and ratings of video games from [Kaggle Datasets](https://www.kaggle.com/rush4ratio/video-game-sales-with-ratings).

```python
df = pd.read_csv('../../data/video_games_sales.csv')
df.info()
```
```python
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 16719 entries, 0 to 16718
Data columns (total 16 columns):
Name               16717 non-null object
Platform           16719 non-null object
Year_of_Release    16450 non-null float64
Genre              16717 non-null object
Publisher          16665 non-null object
NA_Sales           16719 non-null float64
EU_Sales           16719 non-null float64
JP_Sales           16719 non-null float64
Other_Sales        16719 non-null float64
Global_Sales       16719 non-null float64
Critic_Score       8137 non-null float64
Critic_Count       8137 non-null float64
User_Score         10015 non-null object
User_Count         7590 non-null float64
Developer          10096 non-null object
Rating             9950 non-null object
dtypes: float64(9), object(7)
memory usage: 2.0+ MB
```
Not all the movies have ratings, so let’s leave only those observations that don’t have missing values using `dropna` method.

```python
df = df.dropna()
print(df.shape)
```
```
(6825, 16)
```
The table contains 6825 observations and 16 variables/features in total. Let’s look at couple of first entries with `head` method to make sure everything is parsed correctly. For convenience I left only those variables that we’ll use later.

```python
useful_cols = ['Name', 'Platform', 'Year_of_Release', 'Genre', 
               'Global_Sales', 'Critic_Score', 'Critic_Count',
               'User_Score', 'User_Count', 'Rating'
              ]
df[useful_cols].head()
```
![img](https://habrastorage.org/files/594/274/ef6/594274ef67c74d748bb961c9820f8aae.png)

Before we turn to the `seaborn` and `plotly` methods, let’s discuss the simplest and often the most convenient way to visualize data from the `pandas dataframe` - by using plot function.
As an example we will construct a plot of sales of video games in different countries depending on the year. To begin with, let’s filter only the columns we need, then calculate the total sales by years, and call the `plot` function on the resultant dataframe without any parameters.

```python
sales_df = df[[x for x in df.columns if 'Sales' in x] + ['Year_of_Release']]
sales_df.groupby('Year_of_Release').sum().plot()
```
`plot` implementation in `pandas` is based on `matplotlib` library. 

![img](https://habrastorage.org/files/ebb/da5/6a6/ebbda56a661b44a29dd60c795e98ee52.png)

Using the `kind` parameter, you can change the type of the graph/plot to, say, a __`bar chart`__. `Matplotlib` allows quite flexible customization of plots. You can change almost anything on the chart, but you need to dig into the documentation and find the required parameters. For example, the parameter `rot` is responsible for the angle of inclination of the signatures to the `x` axis.

```python
sales_df.groupby('Year_of_Release').sum().plot(kind='bar', rot=45)
```

![img](https://habrastorage.org/files/0a5/24d/4d7/0a524d4d7a2b454ba4a85e9a13e92fd9.png)

## Seaborn

Now let's move on to the `seaborn` library. `Seaborn` is essentially a higher-level API based on the `matplotlib` library. `Seaborn` contains more adequate/reasonable default settings for plotting. If you just add `import seaborn` to the code, the pictures/plots will become much nicer. Also the library contains quite complex types of visualization that would require a large amount of code in `matplotlib`.

Let's get acquainted with the first such "complex" type of plots __`pair plot`__ (`scatter plot matrix`). This visualization will help us look at the relationship of different variables on a single picture. 

```python
cols = ['Global_Sales', 'Critic_Score', 'Critic_Count', 'User_Score', 'User_Count']
sns_plot = sns.pairplot(df[cols])
sns_plot.savefig('pairplot.png')
```

As you can see, histograms of variables distributions are located on the diagonal of the plot(s) matrix. The remaining plots are the usual/regular `scatter plots` for the corresponding feature/variable pairs.

To save plots to files, use `savefig` method.

![img](https://habrastorage.org/files/1ea/ebc/9ec/1eaebc9ecf6d430f825836c3dcc67a8c.png)

It is also possible to plot distribution with `seaborn` __`dist plot`__. For example, let's look at the distribution of critics' ratings `Critic_Score`. By default the graph/chart/plot displays a histogram and a [kernel density estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation).

```python
sns.distplot(df.Critic_Score)
```

![img](https://habrastorage.org/files/b09/02c/fe7/b0902cfe7aaa43c2bdce22840c39d6fe.png)

In order to look more closely at the relationship between two numerical variables, you can also use a __`joint plot`__ - a hybrid of `scatter plot` and `histogram`. Let's see how the `Critic_Score` and the `User_Score` are related.

![img](https://habrastorage.org/files/951/9b1/e29/9519b1e29fd143b3ab26f7c14a5af215.png)

Another useful type of plots is the __`box plot`__. Let's compare critics ratings for games of/in the top 5 biggest gaming platforms.

```python
top_platforms = df.Platform.value_counts().sort_values(ascending = False).head(5).index.values
sns.boxplot(y="Platform", x="Critic_Score", data=df[df.Platform.isin(top_platforms)], orient="h")
```
![img](https://habrastorage.org/files/081/d0d/809/081d0d8095554636befc11b6e0dc879b.png)

I think it's worth discussing a bit more how to understand `box plot`. `Box plot` consists of a box (that's why it is called a `box plot`), whiskers and dots. The box shows the interquartile spread of the distribution, that is, respectively 25% (`Q1`) and 75% (`Q3`) percentiles. The bar/line inside the box indicates the median of the distribution.
Now that we’ve sorted out the box/With the box figured out, let's move on to the whiskers. The whiskers represent the entire scatter of the points except the outliers, that is, the minimum and maximum values that fall within the interval `(Q1 - 1.5*IQR, Q3 + 1.5*IQR)`, where `IQR = Q3 - Q1` is interquartile range. Points on the graph/plot indicate outliers - those values that do not fit into the range of values bounded by the whiskers.

It’s better to see once, so here’s a picture from [wikipedia](https://en.wikipedia.org/wiki/Box_plot):
![img](https://habrastorage.org/files/d0f/f53/8d5/d0ff538d59154901b18a98469de07fde.png)

One more type of plots (the last one we'll consider in this article) is the __`heat map`__. `Heat map` allows to look at the distribution of some numerical variable over two categorical ones. Let’s visualize the total sales of games by genre and gaming platforms.

```python
platform_genre_sales = df.pivot_table(
                        index='Platform', 
                        columns='Genre', 
                        values='Global_Sales', 
                        aggfunc=sum).fillna(0).applymap(float)
sns.heatmap(platform_genre_sales, annot=True, fmt=".1f", linewidths=.5)
```
![img](https://habrastorage.org/files/90c/47e/235/90c47e23522f4c7d98c0cd08f80da54f.png)

## Plotly
We examined the visualizations based on the `matplotlib` library. However, this is not the only option for plotting in `python`. Let’s also get acquainted with the __`plotly`__ library. `Plotly` is an open-source library that allows you to build interactive graphics/plots in jupyter.notebook without having to bury into javascript.

The beauty of interactive graphics/plots is that you can see the exact numerical value by hovering the mouse, hide uninteresting/unnecessary rows/items in the visualization, zoom in a certain part of the graph/plot, etc.

Before we start, let’s import all the necessary modules and initialize `plotly` with the `init_notebook_mode` command.

```python
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly
import plotly.graph_objs as go

init_notebook_mode(connected=True)
```

First, let’s build a __`line plot`__ with the dynamics of the number of games released and their sales by years.

```python
# compute the number of games released and their sales by years
years_df = df.groupby('Year_of_Release')[['Global_Sales']].sum().join(
    df.groupby('Year_of_Release')[['Name']].count()
)
years_df.columns = ['Global_Sales', 'Number_of_Games']

# create the line for the copies sold
trace0 = go.Scatter(
    x=years_df.index,
    y=years_df.Global_Sales,
    name='Global Sales'
)

# create the line for the number of games released 
trace1 = go.Scatter(
    x=years_df.index,
    y=years_df.Number_of_Games,
    name='Number of games released'
)

# define array of data and set the title of the graph in layout
data = [trace0, trace1]
layout = {'title': 'Statistics of video games'}

# create Figure object and visualize it
fig = go.Figure(data=data, layout=layout)
iplot(fig, show_link=False)
```
`plotly` makes visualization of the `Figure` object, which consists of data (an array of lines that are called `traces` in the library) and the design, or style, which the `layout` object is responsible for. In simple cases you can just call the `iplot` function from the `traces` array.

The parameter `show_link` is responsible for links to the online platform plot.ly on the graphs/plot. Since usually this functionality is not needed, I prefer to hide it to prevent accidental clicks.

![img](https://habrastorage.org/files/8b9/3fb/be5/8b93fbbe58b44ed79e5b7f40d83f8d1d.png)

You can save the plot as an html file.

```python
plotly.offline.plot(fig, filename='years_stats.html', show_link=False)
```

Let's also look at the market share of gaming platforms calculated by the number of games released and by total revenue. To do this let’s construct a __`bar chart`__.

```python
# compute the number of games sold and released by platforms
platforms_df = df.groupby('Platform')[['Global_Sales']].sum().join(
    df.groupby('Platform')[['Name']].count()
)
platforms_df.columns = ['Global_Sales', 'Number_of_Games']
platforms_df.sort_values('Global_Sales', ascending=False, inplace=True)

# create traces for visualisation
trace0 = go.Bar(
    x=platforms_df.index,
    y=platforms_df.Global_Sales,
    name='Global Sales'
)

trace1 = go.Bar(
    x=platforms_df.index,
    y=platforms_df.Number_of_Games,
    name='Number of games released'
)

# create an array of data and set titles for chart/plot and x axis in layout
data = [trace0, trace1]
layout = {'title': 'Share of platforms', 'xaxis': {'title': 'platform'}}

# create Figure object and visualise it
fig = go.Figure(data=data, layout=layout)
iplot(fig, show_link=False)
```

![img](https://habrastorage.org/files/496/3f9/2eb/4963f92ebf8a4a93a8d7abb4a6205c96.png)

You can also build __`box plot`__ in `plotly`. Let’s consider the distribution of critics ratings depending on the genre of the game.

```python

# create Box trace for each genre in our data
data = []
for genre in df.Genre.unique():
    data.append(
        go.Box(y=df[df.Genre==genre].Critic_Score, name=genre)
    )

# visualize data
iplot(data, show_link = False)
```
![img](https://habrastorage.org/files/6a4/e4e/a10/6a4e4ea10da049d2b56c9323dea77c2e.png)

You can build various types of visualizations with `plotly`. The plots are quite nice with the default settings. However, the library allows you to flexibly configure various visualization options: colors, fonts, labels, annotations and much more.

# Example of Visual Data Analysis
Let’s read the already familiar data on the churn of customers of the telecom operator to a `DataFrame`. 



```python
df = pd.read_csv('../../data/telecom_churn.csv')
```

Let’s check whether everything is parsed correctly and look at the first 5 entries (`head` method).


```python
df.head()
```
<img src="https://habrastorage.org/files/978/022/6a8/9780226a800b4a1da342daaa966b4a0e.png" align='center'/>
<br>

Number of rows (customers) and columns (variables):


```python
df.shape
```




    (3333, 20)



Let’s have a look at the variables and make sure that there are no missing values and each column contains 3333 entries.


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 3333 entries, 0 to 3332
    Data columns (total 20 columns):
    State                     3333 non-null object
    Account length            3333 non-null int64
    Area code                 3333 non-null int64
    International plan        3333 non-null object
    Voice mail plan           3333 non-null object
    Number vmail messages     3333 non-null int64
    Total day minutes         3333 non-null float64
    Total day calls           3333 non-null int64
    Total day charge          3333 non-null float64
    Total eve minutes         3333 non-null float64
    Total eve calls           3333 non-null int64
    Total eve charge          3333 non-null float64
    Total night minutes       3333 non-null float64
    Total night calls         3333 non-null int64
    Total night charge        3333 non-null float64
    Total intl minutes        3333 non-null float64
    Total intl calls          3333 non-null int64
    Total intl charge         3333 non-null float64
    Customer service calls    3333 non-null int64
    Churn                     3333 non-null bool
    dtypes: bool(1), float64(8), int64(8), object(3)
    memory usage: 498.1+ KB


<spoiler title="Variables description">



|  Name  | Description | Type |
|---         |--:       |     |
| **State** | Letter state code| nominal |
| **Account length** | How long the client has been with the company | numerical |
| **Area code** | Phone number prefix | numerical  |
| **International plan** | International Roaming (on/off) | binary |
| **Voice mail plan** | Voicemail (on/off) | binary |
| **Number vmail messages** | Number of voicemail messages | numerical |
| **Total day minutes** |  Total duration of calls during day | numerical |
| **Total day calls** | Total number of calls during day | numerical |
| **Total day charge** | Total charge for services during day | numerical |
| **Total eve minutes** | Total duration of calls during evening | numerical |
| **Total eve calls** | Total number of calls during evening | numerical |
| **Total eve charge** | Total charge for services during evening | numerical |
| **Total night minutes** | Total duration of calls during night | numerical |
| **Total night calls** | Total number of calls during night | numerical |
| **Total night charge** | Total charge for services during night | numerical |
| **Total intl minutes** | Total duration of international calls | numerical |
| **Total intl calls** | Total number of international calls | numerical |
| **Total intl charge** | Total charge for international calls| numerical |
| **Customer service calls** | Number of calls to service center | numerical |

Target variable is **Churn** – feature of outflow, it’s binary(1 means the loss of client, i.e. churn). Later we’ll build models that predict this variable based on others, that’s why we called it target.
</spoiler>

Let’s look at the distribution of target variable - churn rate.


```python
df['Churn'].value_counts()
```




    False    2850
    True      483
    Name: Churn, dtype: int64




```python
df['Churn'].value_counts().plot(kind='bar', label='Churn')
plt.legend()
plt.title('Distribution of churn rate');
```
<img src="https://habrastorage.org/files/363/837/4ca/3638374ca01a4777b9c7280e30669b7d.png" align='center'/><br>


Let’s divide all variables except *Churn* into following groups:
 - binary: *International plan*, *Voice mail plan*
 - categorical: *State*
 - ordinal: *Customer service calls*
 - numerical: all others


Let's look at the correlation of numerical variables. We can seen on the colored correlation matrix that variables as *Total day charge* are computed from the minutes spent on phone (*Total day minutes*). That is, 4 variables can be easily thrown out, they do not carry useful information.


```python
corr_matrix = df.drop(['State', 'International plan', 'Voice mail plan',
                      'Area code'], axis=1).corr()
```


```python
sns.heatmap(corr_matrix);
```

<img src="https://habrastorage.org/files/148/611/03f/14861103fbf14dff9c56802f709f0749.png" align='center' /><br>



Now let's look at the distributions of all numerical variables left. We’ll look at binary/categorical/ordinal variables separately.


```python
features = list(set(df.columns) - set(['State', 'International plan', 'Voice mail plan',  'Area code',
                                      'Total day charge',   'Total eve charge',   'Total night charge',
                                        'Total intl charge', 'Churn']))

df[features].hist(figsize=(20,12));
```
<img src="https://habrastorage.org/files/de3/01b/18f/de301b18f8e5459b9ad8158314a80769.png" align='center' /><br>

We see that most of the variables are normally distributed. Exceptions are the *Customer Service calls* (here the Poisson distribution is more appropriate) and the number of voice messages (*Number vmail messages*, it has peak at zero, i.e. those who haven’t enabled voicemail). Also, the distribution of the number of international calls (*Total intl calls*) is shifted.

It is also useful to build such pictures where the distributions of the features/variables are drawn on the main diagonal, and outside it lie scatter plots for feature/variable pairs. Sometimes it can lead to some conclusions, but in this case everything is pretty clear, without surprises.


```python
sns.pairplot(df[features + ['Churn']], hue='Churn');
```


<img src="https://habrastorage.org/files/aef/703/336/aef703336cbf42feabe289b0c849dc2b.png" align='center' /><br>


**Now let’s see how variables are related to the target - Churn.**



Let’s construct boxplots to see distribution statistics of numerical variables in two groups: loyal customers and those who left. 


```python
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(16, 10))

for idx, feat in  enumerate(features):
    sns.boxplot(x='Churn', y=feat, data=df, ax=axes[idx / 4, idx % 4])
    axes[idx / 4, idx % 4].legend()
    axes[idx / 4, idx % 4].set_xlabel('Churn')
    axes[idx / 4, idx % 4].set_ylabel(feat);
```

<img src="https://habrastorage.org/files/ca8/96c/e02/ca896ce024c5491fa8f21672efa8ab6e.png" align='center' /><br>


By eye we can see the greatest differences for the *Total day minutes*, *Customer service calls* and *Number vmail messages* variables. Later we’ll learn to determine the feature importance in classification problems using random forest (or gradient boosting), and it turns out that the first two are really very important features for churn prediction.

Let's have a more detailed look at the distribution of the number of minutes spoken during the day among the loyal/disloyal customers. To the left we see the boxplots we know, to the right are the smoothed histograms of the distribution of the numerical variable in two groups (more a nice picture, everything is clear from the boxplot).



An interesting **observation**: customers that leave use communication services more on average. Perhaps they are unhappy with the tariffs, and one of the measures to combat/prevent the churn will be a reduction in tariff rates (mobile communication costs). The company will need to conduct additional economic analysis to find out whether such measures will be justified.





```python
_, axes = plt.subplots(1, 2, sharey=True, figsize=(16,6))

sns.boxplot(x='Churn', y='Total day minutes', data=df, ax=axes[0]);
sns.violinplot(x='Churn', y='Total day minutes', data=df, ax=axes[1]);
```

<img src="https://habrastorage.org/files/f19/c94/fac/f19c94facf30431cb37804b449bac34b.png" align='center'/><br>



Let’s now plot the distribution of the number of calls to the service center (we constructed this picture in the first article). This variable doesn’t have many unique values (it can be considered either as numerical or ordinal), and it is easier to depict the distribution using `countplot`. **Observation**: the churn rate increases significantly from 4 calls to the service center.


```python
sns.countplot(x='Customer service calls', hue='Churn', data=df);
```

<img src="https://habrastorage.org/files/43c/30f/8b7/43c30f8b7053410c90e575d5b26d3ae1.png" align='center'/><br>


Now let's look at the relationship between the binary features *International plan* and the *Voice mail plan* with churn. **Observation**: when roaming is on/enabled, churn rate is much higher, i.e. the presence of international roaming is a strong feature. We can’t say the same about voice mail.


```python
_, axes = plt.subplots(1, 2, sharey=True, figsize=(16,6))

sns.countplot(x='International plan', hue='Churn', data=df, ax=axes[0]);
sns.countplot(x='Voice mail plan', hue='Churn', data=df, ax=axes[1]);
```
<img src="https://habrastorage.org/files/60f/6d0/691/60f6d06912784245acb19571361e6627.png" align='center' /><br>

Finally, let's see how the categorical variable *State* is related to churn. This one is not so pleasant to work with, since the number of unique states is quite high - 51. We can first build a summary table or calculate churn rate for each state. But there’s not much data for each state individually (only 3 to 17 customers in each state left). So perhaps we better leave the *State* variable out of the classification models because of the *overfitting* risk (but we will check this for *cross-validation*, stay tuned!).


Churn rate by state:


```python
df.groupby(['State'])['Churn'].agg([np.mean]).sort_values(by='mean', ascending=False).T
```

<img src="https://habrastorage.org/files/e08/460/918/e0846091885244e9b124207a834e908f.png" align='center'/>
<img src="https://habrastorage.org/files/3c4/d55/e55/3c4d55e55f9044f0836fc513598de442.png" align='center' /><br>


We can see that churn rate in New Jersey and California is above 25%, and it’s less than 5% in Hawaii and Alaska. But these conclusions are based on too small statistics/dataset and perhaps it's just the feature of the data available (it is possible here to test Matthews and Cramer correlation hypotheses, but this is beyond the scope of this article).

# Peeking at the n-Dimensional Space With t-SNE 

Let’s construct a t-SNE representation of the same churn data. The name of the method is complex - t-distributed Stohastic Neighbor Embedding, math is also cool/impressive/tough (and we’ll not delve into it, but for those who want it [here](http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf) is the original paper by Geoffrey Hinton and his graduate student in JMLR). But the basic idea is as easy as pie: finding such a representation of a multidimensional feature space to a plane (or 3D, but almost always 2D is chosen) so that points that were far apart in the initial space appeared also far apart on the plane, and those that were close also remained close to each other. That is, neighbor embedding is a kind of search for a new data representation that preserves neighborhood.

A few details: let’s throw away the State and Churn variables and convert the binary Yes/No variables to numbers (`pd.factorize`). We also need to scale the data - subtract mean from each variable and divide it into its standard deviation, this is done by StandardScaler.


```python
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
```


```python
# convert all variable to numerical and throw away State
X = df.drop(['Churn', 'State'], axis=1)
X['International plan'] = pd.factorize(X['International plan'])[0]
X['Voice mail plan'] = pd.factorize(X['Voice mail plan'])[0]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```


```python
%%time
tsne = TSNE(random_state=17)
tsne_representation = tsne.fit_transform(X_scaled)
```

    CPU times: user 20 s, sys: 2.41 s, total: 22.4 s
    Wall time: 21.9 s



```python
plt.scatter(tsne_representation[:, 0], tsne_representation[:, 1]);
```

<img src="https://habrastorage.org/files/5ad/048/43c/5ad04843caac44f7890fd61a0d9bb9b1.png" align='center'/><br>


Let’s color the t-SNE representation according to churn (green for loyal, red for those who left).


```python
plt.scatter(tsne_representation[:, 0], tsne_representation[:, 1], 
            c=df['Churn'].map({0: 'green', 1: 'red'}));
```
<img src="https://habrastorage.org/files/ca2/192/1c1/ca21921c11a44aa787dcedc74b7abe19.png" align='center'/><br>


We can see that the leaving customers are mostly "huddled" in certain areas of the feature space. 


To better understand the picture, we can also color it according the remaining binary features - roaming and voice mail. Green areas correspond to objects that have this binary feature.




```python
_, axes = plt.subplots(1, 2, sharey=True, figsize=(16,6))

axes[0].scatter(tsne_representation[:, 0], tsne_representation[:, 1], 
            c=df['International plan'].map({'Yes': 'green', 'No': 'red'}));
axes[1].scatter(tsne_representation[:, 0], tsne_representation[:, 1], 
            c=df['Voice mail plan'].map({'Yes': 'green', 'No': 'red'}));
axes[0].set_title('International plan');
axes[1].set_title('Voice mail plan');
```

<img src="https://habrastorage.org/files/29e/faf/56a/29efaf56ad824fba914123fe51871d92.png" align='center'/><br>


Now it is clear that, for example, many leaving customers are crowded together in the left cluster of people with roaming enabled, but without voice mail.

Finally, we’ll note the disadvantages of t-SNE (yes, it’s better to write a separate article on it):
 - Great computational complexity. [This](http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) `sklearn` implementation will most likely not help you in a real task; on large samples, it’s better to look at [Multicore-TSNE](https://github.com/DmitryUlyanov/Multicore-TSNE);
 - The picture/plot can change greatly when changing a random seed, and this complicates interpretation. [This one](http://distill.pub/2016/misread-tsne/) is a good tutorial on t-SNE. But in general you shouldn’t make any far-reaching conclusions based on such pictures, it’s better not to read tea leaves. Sometimes some findings strike the eye and are confirmed later in the study, but it does not happen very often.
 
Couple more pictures. With t-SNE you can get a really good idea of the data (as in the case of handwritten digits, [here's](https://colah.github.io/posts/2014-10-Visualizing-MNIST/) a good paper), or you can just draw a Christmas tree toy.

<img src="https://habrastorage.org/files/583/9c1/f2f/5839c1f2f402489aaa928e593e9b1153.png" align='center'/>

<img src="https://habrastorage.org/files/ad9/089/1a1/ad90891a141c448f90de89c34a79cc9f.jpg" align='center'/>

## Homework #2

The second homework assignment is devoted to the analysis of the dataset on the popularity of articles on Habrahabr.

We suggest that you complete this task and then answer a few questions. Here’s the [link](https://docs.google.com/forms/d/e/1FAIpQLSf3b5OG8zX_nLQBQ-t20c6M5Auz-VUL-yxj8Fm9_o_XWDBTrg/viewform?c=0&w=1) to the form for answers (it is also in the [notebook](https://github.com/Yorko/mlcourse_open/blob/master/jupyter_notebooks/topic2_visual_analysis/hw2_habr_visual_analysis.ipynb)). You can change the answers in the form after sending, but not after the deadline. 

Дедлайн: 13 марта 23:59 (жесткий).

## Useful Links and Resources
-  Прежде всего, [официальная документация](http://seaborn.pydata.org/api.html) и [галерея](http://seaborn.pydata.org/examples/index.html) примеров различных графиков для `seaborn`
- При работе с `plotly` также поможет официальный сайт: [полная документация](https://plot.ly/python/reference/), [большое количество разобранных примеров](https://plot.ly/python/) 
- Кроме того, примеры анализа данных и визуализаций на `plotly` можно посмотреть в моей статье на Хабрахабре [Немного про кино или как делать интерактивные визуализации в python](https://habrahabr.ru/post/308162/). Из не расcмотренного здесь, но иногда полезного, в статье можно найти пример графика с drop-down menu.

*Статья написана в соавторстве с @yorko (Юрием Кашницким). Автор домашнего задания –  @cotique (Екатерина Демидова).*

