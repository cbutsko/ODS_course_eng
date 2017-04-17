Open Data Science community welcomes the participants of the course!
 
In the course we have already met several key algorithms of machine learning. However, before moving on to the more fancy algorithms and approaches, we'd like to take a step to the side and talk about data preparation for the model training. A well-known principle of "garbage in - garbage out" is 100% applicable to any task in learning machine; any experienced analyst can think of examples from practice when a simple model trained on high-quality data has proved to be better than a highbrow ensemble built on data that wasn't clean enough. 

<img src="https://habrastorage.org/files/cd7/2d8/d16/cd72d8d16d8f409898546ba5d397240f.jpg" align='center'><br>
<habracut/>

<spoiler title="Список статей серии">
1. [Первичный анализ данных с Pandas](https://habrahabr.ru/company/ods/blog/322626/)
2. [Визуальный анализ данных c Python](https://habrahabr.ru/company/ods/blog/323210/)
3. [Классификация, деревья решений и метод ближайших соседей](https://habrahabr.ru/company/ods/blog/322534/)
4. [Линейные модели классификации и регрессии](https://habrahabr.ru/company/ods/blog/323890/)
5. [Композиции: бэггинг, случайный лес](https://habrahabr.ru/company/ods/blog/324402/)
6. [Построение и отбор признаков. Приложения в задачах обработки текста, изображений и геоданных](https://habrahabr.ru/company/ods/blog/325422/)
7. Обучение без учителя: PCA, кластеризация, поиск аномалий
</spoiler>

As part of today's article I want to give a review of three similar but different tasks:
- feature extraction and feature engineering: conversion of data specific to the domain area to a model-comprehensible vectors;
- feature transformation: transformation of data to improve the accuracy of the algorithm;
- feature selection: removing unnecessary features.

I want to separately notice that this article will contain almost no formulas, but there will be relatively large amount of code. 

Some examples will use the dataset from [Renthop](https://www.renthop.com/) company, used in [Two Sigma Connect: Rental Listing Inquires](https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries) Kaggle competition. In this task, you need to predict the popularity of the rental listing, ie solve the classification problem with three classes `[ 'low', 'medium' , 'high']`. To evaluate the solutions log loss metric is used (the less the better). Those who still don't have a Kaggle account, will have to register; also you'll need to accept the rules of the competition to download the data.

```
# Before you start don't forget to download the file train.json.zip from Kaggle and unzip it
import json
import pandas as pd

# Let's load the dataset from Renthop right away
with open('train.json', 'r') as raw_data:
    data = json.load(raw_data)
    df = pd.DataFrame(data)
```

<img src='https://habrastorage.org/files/58e/152/f83/58e152f8398743d6abca8f287a4c715f.jpg' align='center'><br>

* [Feature extraction](https://habrahabr.ru/company/ods/blog/325422/#izvlechenie-priznakov-feature-extraction)
	* [Texts](https://habrahabr.ru/company/ods/blog/325422/#teksty)
	* [Images](https://habrahabr.ru/company/ods/blog/325422/#izobrazheniya)
	* [Geospatial data](https://habrahabr.ru/company/ods/blog/325422/#geodannye)
	* [Date and time](https://habrahabr.ru/company/ods/blog/325422/#data-i-vremya)
	* [Time series, web, etc.](https://habrahabr.ru/company/ods/blog/325422/#vremennye-ryady-veb-i-prochee)
* [Feature transformations](https://habrahabr.ru/company/ods/blog/325422/#preobrazovaniya-priznakov-feature-transformations)
	* [Normalization and changing distribution](https://habrahabr.ru/company/ods/blog/325422/#normalizaciya-i-izmenenie-raspredeleniya)
	* [Interactions](https://habrahabr.ru/company/ods/blog/325422/#vzaimodeystviya-interactions)
	* [Filling in the missing values](https://habrahabr.ru/company/ods/blog/325422/#zapolnenie-propuskov)
* [Feature selection](https://habrahabr.ru/company/ods/blog/325422/#vybor-priznakov-feature-selection)
	* [Statistical approaches](https://habrahabr.ru/company/ods/blog/325422/#statisticheskie-podhody)
	* [Selection by modelling](https://habrahabr.ru/company/ods/blog/325422/#otbor-s-ispolzovaniem-modeley)
	* [Grid search](https://habrahabr.ru/company/ods/blog/325422/#perebor)
* [Homework](https://habrahabr.ru/company/ods/blog/325422/#domashnee-zadanie)

## Feature Extraction
In real life data rarely come in the form of ready-to-use matrices, that's why every task begins with the feature extraction. Sometimes, of course, it is enough to read the csv file and convert it into `numpy.array`, but this is a happy exception.Let's look at some of the popular types of data from which features can be extracted.

###  Texts

Text is the most obvious example of data in free format; there are enough methods of text processing so that they won't fit into a single article. Nevertheless, let's review the most popular ones.
 
Before working with text, one must tokenize it. Tokenization implies splitting the text into tokens, in the simplest case, these are just words. But making it with a too simple regularka ("head-on"), we may lose some of the meaning: "Santa Barbara" is one token, not two. But the call to "steal, kill!" can be needlessly split into two tokens. There are ready tokenayzers that take into account peculiarities of the language, but they can be wrong too, especially if you work with specific texts (professional vocabulary, slang, misspellings).
 
After tokenization, in most cases you need to think about the reduction to normal form. It is about stemming and/or lemmatization; these are similar processes used to process word forms. One can read about the difference between them [here](http://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html).
 
So we turned the document into a sequence of words, now we can start turning them into a vector. The easiest approach is called Bag of Words: we create a vector with length of the dictionary, compute the number of occurrences of each word in the text, and place this number to the appropriate position in the vector. In the code it looks even simpler than in words: 

<spoiler title="Bag of Words without extra libraries">
```
from functools import reduce
import numpy as np

texts = [['i', 'have', 'a', 'cat'],
         ['he', 'have', 'a', 'dog'],
         ['he', 'and', 'i', 'have', 'a', 'cat', 'and', 'a', 'dog']]

dictionary = list(enumerate(set(list(reduce(lambda x, y: x + y, texts)))))

def vectorize(text):
    vector = np.zeros(len(dictionary))
    for i, word in dictionary:
        num = 0
        for w in text:
            if w == word:
                num += 1
        if num:
            vector[i] = num
    return vector

for t in texts:
    print(vectorize(t))
```
</spoiler>

Also, the idea is well illustrated by a picture:

<img src='https://habrastorage.org/files/549/810/b75/549810b757f94e4784b6780d84a1112a.png' align='center'>

This is an extremely naive implementation. In real life, you need to take care of stop words, the maximum size of the dictionary, efficient data structure (usually text data is converted to a sparse vector) ...
 
When using algorithms like Bag of Words, we lose the order of the words in the text, which means that the texts "i have no cows" and "no, i have cows" will be identical after vectorization, although semantically they are opposite. To avoid this problem, we can step back and change the approach to tokenization, for example, by using N-grams (the combination of N consecutive terms).

<spoiler title="Let's try it out in practice">
```
In : from sklearn.feature_extraction.text import CountVectorizer

In : vect = CountVectorizer(ngram_range=(1,1)) 

In : vect.fit_transform(['no i have cows', 'i have no cows']).toarray()
Out: 
array([[1, 1, 1],
       [1, 1, 1]], dtype=int64)

In : vect.vocabulary_
Out: {'cows': 0, 'have': 1, 'no': 2}

In : vect = CountVectorizer(ngram_range=(1,2)) 

In : vect.fit_transform(['no i have cows', 'i have no cows']).toarray()
Out: 
array([[1, 1, 1, 0, 1, 0, 1],
       [1, 1, 0, 1, 1, 1, 0]], dtype=int64)

In : vect.vocabulary_
Out: 
{'cows': 0,
 'have': 1,
 'have cows': 2,
 'have no': 3,
 'no': 4,
 'no cows': 5,
 'no have': 6}
```
</spoiler>
Also note that it is optional to operate on words: in some cases it is possible to generate N-grams of characters (e.g., such an algorithm will take into account the similarity of related words or typos).
```
In : from scipy.spatial.distance import euclidean

In : vect = CountVectorizer(ngram_range=(3,3), analyzer='char_wb') 

In : n1, n2, n3, n4 = vect.fit_transform(['иванов', 'петров', 'петренко', 'смит']).toarray()

In : euclidean(n1, n2)
Out: 3.1622776601683795

In : euclidean(n2, n3)
Out: 2.8284271247461903

In : euclidean(n3, n4)
Out: 3.4641016151377544
```

Development of the Bag of Words idea: words that are rarely found in the body (in all the documents of this set of data), but are present in this particular document might be more important. Then it makes sense to increase the weight of more domain specific words to separate them from general words. This approach is called TF-IDF, and this one can not be written in ten lines, so everyone interested can get acquainted with the details in the external sources such as [wiki](https://en.wikipedia.org/wiki/Tf%E2%80%93idf).The default option is as follows:

$$display$$ idf(t, D) = \log \frac {\mid D \mid} {df(d,t) + 1}
$$display$$

$$display$$ tfidf(t, d, D) = tf(t, d) \times idf(t, D)
$$display$$

Analogs of Bag of words can also be found outside of word problems: for example, bag of sites in the [competition](https://inclass.kaggle.com/c/catch-me-if-you-can-intruder-detection-through-webpage-session-tracking) that we hold - Catch Me If You Can. You can look for other examples - [bag of apps](https://www.kaggle.com/xiaoml/talkingdata-mobile-user-demographics/bag-of-app-id-python-2-27392), [bag of events](http://www.interdigital.com/download/58540a46e3b9659c9f000372).

<img src="https://habrastorage.org/files/ec1/273/bc7/ec1273bc740145ec92e25991415b1644.jpg" align='center'><br>

Using these algorithms, it is possible to obtain a working solution to a simple problem, a kind of baseline. However, for those who don't like the classics, there are more new approaches. The best-selling method of the new wave is Word2Vec, but there are alternatives (Glove, Fasttext ...).
 
Word2Vec is a special case of the Word Embedding algorithms. Using Word2Vec and similar models, we can not only vectorize words in the space of high dimensionality (typically a few hundred), but also compare their semantic proximity. A classic example of the operations on vectorized concepts: king - man + woman = queen.

<img src='https://habrastorage.org/getpro/habr/post_images/158/230/d1a/158230d1ad839c517d1855ea005bd590.gif' align='center'><br>

It is worth to understand that this model does not, of course, comprehend the meaning of the words, but simply tries to place the vectors in such a way that the words used in common context are placed close to each other. If this is not taken into account, it is possible to come up with a lot of fun things: for example, to find an opposite of Hitler by multiplying the appropriate vector by -1.
 
Such models need to be trained on very large data sets in order for the coordinates of the vectors to really reflect the semantics of words. A pretrained model for your own tasks can be downloaded, for example, [here](https://github.com/3Top/word2vec-api#where-to-get-a-pretrained-models).

Similar methods are, incidentally, also applied in other areas (for example, in bioinformatics). Quite an unexpected application is [food2vec](https://jaan.io/food2vec-augmented-cooking-machine-intelligence/).

### Images

Working with images is easier and harder at the same time. Easier because it is often possible not to think at all and just use one of the popular pretrained networks; harder because if it is still necessary to go into details, this rabbit hole will be damn deep. But let's start from the beginning.
 
At a time when GPUs were weaker, and the "renaissance of neural networks" has not yet happened, feature generation from the images was a separate complex area. To work with pictures one had to work at a low level, determining, for example, corners, borders of regions and so on. Experienced specialists in computer vision could have drawn a lot of parallels between older approaches and neural network hipsterism: in particular, convolutional layers in today's networks are very similar to [Haar cascades](https://habrahabr.ru/post/208092/). Not being experienced in this matter, I will not even try to transfer knowledge from public sources, I'll just leave a couple of links to libraries [skimage](http://scikit-image.org/docs/stable/api/skimage.feature.html) and [SimpleCV](http://simplecv.readthedocs.io/en/latest/SimpleCV.Features.html) and move directly to our days.
 
Often for problems associated with images a convolution network is used. You can not come up with the architecture and not train a network from scratch, but take a pretrained state-of-the-art network, the weights of which can be downloaded from public sources. To adapt it to their needs, data scientists practice of so-called fine tuning: "detach" the last fully connected layers of the network, add new layers chosen for a specific task instead of them, and network is trained on new data. But if you just want to vectorize the image for some your needs (for example, to use some non-network classifier), just tear off the last layers and use the output of the previous layers:

```
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from scipy.misc import face
import numpy as np

resnet_settings = {'include_top': False, 'weights': 'imagenet'}
resnet = ResNet50(**resnet_settings)

img = image.array_to_img(face())
# What a cute raccoon!
img = img.resize((224, 224))
# In real life, you may need to pay more attention to resizing
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
# Need an extra dimension because model is designed to work with an array of images

features = resnet.predict(x)
```
<img src='https://habrastorage.org/getpro/habr/post_images/200/12a/d64/20012ad648ebf0f8519b6465d9e9bda7.png' align='center'><br>
_Here's a classifier trained on one dataset and adapted for a different one via "tearing off" of the last layer and adding a new one instead_

Nevertheless, we should not over-focus on neural network techniques. Some features generated by hand may be useful in our day: for example, predicting the popularity of rental listing, we can assume that bright apartments attract more attention and make the feature of "the average value of the pixel." One can get inspired by examples in the documentation of [relevant libraries](http://pillow.readthedocs.io/en/3.1.x/reference/ImageStat.html).
 
If you expect text on the image, you can also read it without unwinding a complicated neural network: for example, using [pytesseract](https://github.com/madmaze/pytesseract).
```
In : import pytesseract

In : from PIL import Image

In : import requests

In : from io import BytesIO

In : img = 'http://ohscurrent.org/wp-content/uploads/2015/09/domus-01-google.jpg'
# Just a random picture from search 

In : img = requests.get(img)
     ...: img = Image.open(BytesIO(img.content))
     ...: text = pytesseract.image_to_string(img)
     ...: 

In : text
Out: 'Google'
```

One must understand that pytesseract is not a panacea:
```
# This time we take a picture from Renthop
In : img = requests.get('https://photos.renthop.com/2/8393298_6acaf11f030217d05f3a5604b9a2f70f.jpg')
     ...: img = Image.open(BytesIO(img.content))
     ...: pytesseract.image_to_string(img)
     ...: 

Out: 'Cunveztible to 4}»'
```

Another case where neural networks won't help is the extraction of features from meta information. And EXIF ​​can store many useful things: manufacturer and camera model, resolution, use of the flash, geographic coordinates of shooting, software used to process image and more.

### Geospatial data

Geographic data is not so often found in the problems, but it's still useful to master the basic techniques for working with it, especially since there are quite a number of ready-made solutions in this area.
 
Geospatial data is often presented in the form of addresses or pairs "Latitude + Longitude", i.e. points. Depending on the task, you may need two mutually inverse operations: geocoding (recovery of a point from address) and a reverse geocoding (vice versa). Both are put into practice via external APIs like Google Maps or OpenStreetMap. Different geocoders have their own characteristics, the quality varies from region to region. Fortunately, there are universal libraries like [geopy](https://github.com/geopy/geopy) that act as wrappers on a variety of external services.
 
If you have a lot of data, it is easy to hit the limits of external API. Besides, it's not always the best solution in terms of speed to receive information via HTTP. Therefore it is necessary to bear in mind the possibility of using a local version of OpenStreetMap.
 
If you have rather little data, enough time, and no desire to extract fancy features, you can not bother with OpenStreetMap and use `reverse_geocoder`:

```
In : import reverse_geocoder as revgc

In : revgc.search((df.latitude, df.longitude))
Loading formatted geocoded file...
Out: 
[OrderedDict([('lat', '40.74482'),
              ('lon', '-73.94875'),
              ('name', 'Long Island City'),
              ('admin1', 'New York'),
              ('admin2', 'Queens County'),
              ('cc', 'US')])]
```

Working with geoсoding, we must not forget that the address may contain typos, thus it is worth taking the time to clean data. Coordinates usually contain fewer misprints, but all is also not that rosy with them: GPS can be noisy by nature, and in some places (tunnels, skyscrapers neighborhoods ...) - quite a bit. If the data source is a mobile device, it is worth considering that in some cases the geolocation is determined not by GPS, but by WiFi networks in the area, which leads to holes in space and teleportation: among a set of points that describe the journey in Manhattan there can suddenly appear one from Chicago. 

<spoiler title="Hypotheses about teleportation">
WiFi location tracking is based on the combination of SSID and MAC-addresses, which may correspond at quite different points (e.g., federal provider standardizes the firmware of routers up to MAC-address, and places them in different cities). There are also more trivial reasons such as a company moving to another office with its routers.
</spoiler>

The point is usually located not in the middle of nowhere, but among infrastructure. Here you can unleash the imagination and begin to invent features applying your life experience and knowledge of the domain area. The proximity of the point to the subway, number of storeys of the building, the distance to the nearest store, the number of ATMs around - in one task, you can come up with dozens of features and extract them from various external sources. For problems outside the urban infrastructure, features from more specific sources may be useful: for example, the height above sea level.
 
If two or more points are interconnected, it may be worthwhile to extract features from the route between them. Here distances (it's worth looking at both great circle distance, and "honest" distance calculated by the routing graph), number of turns with the ratio of left and right ones, number of traffic lights, junctions, bridges may be useful. For example, in one of my tasks a feature that I called "the complexity of the road" - the graph-calculated distance divided by the GCD - worked well.

### Date and time 

It would seem that working with date and time should be standardized because of the prevalence of these features, but pitfalls remain.
 
Let's start with days of the week - they are easy to turn into 7 dummy variables using one-hot encoding. In addition, it is useful to allocate a separate feature for the weekend.
```
df['dow'] = df['created'].apply(lambda x: x.date().weekday())
df['is_weekend'] = df['created'].apply(lambda x: 1 if x.date().weekday() in (5, 6) else 0)
```
Some tasks may require additional calendar features: for example, cash withdrawals can be linked to the day of salary, and the purchase of the metro card - to the beginning of the month. Truth be told, when working with time series data, it is a good practice to have on hand a calendar with public holidays, abnormal weather conditions, and other important events. 

<spoiler title="Professional unfunny humor">
- What do Chinese New Year, the New York marathon, gay parade and Trump inauguration have in common?
- They all need to be put to the calendar of potential anomalies.
</spoiler>

Situation with hour (minute, day of the month ...) is not so rosy. If you use the hour as a real variable, we contradict a little the nature of data: 0 <23, although 0:00:00 02.01> 01.01 23:00:00. For some problems, it can be critical. If, however, you encode them as categorical variables, you'll breed a large numbers of features and lose information about proximity: the difference between 22 and 23 will be the same as the difference between 22 and 7.
 
There also exist some more esoteric approaches to such data. For example, projection on a circle and using the two coordinates.
```
def make_harmonic_features(value, period=24):
    value *= 2 * np.pi / period
    return np.cos(value), np.sin(value)
```
This transformation preserves the distance between points, which is important for some algorithms based on distance (kNN, SVM, k-means ...)
```
In : from scipy.spatial import distance

In : euclidean(make_harmonic_features(23), make_harmonic_features(1))
Out: 0.5176380902050424

In : euclidean(make_harmonic_features(9), make_harmonic_features(11))
Out: 0.5176380902050414

In : euclidean(make_harmonic_features(9), make_harmonic_features(21))
Out: 2.0
```
However, the difference between such coding methods usually can be caught only in the third decimal place of the metric.

### Time series, web, etc. 
I didn't have a chance to work enough with time series, so I leave a link to the [library that automatically generates features for time series](https://github.com/blue-yonder/tsfresh) and will go further.

If you're working with web, then you usually have information about the user's User Agent. It is a mine of information.
Firstly, one needs to extract the operating system from it. Secondly, make a feature `is_mobile`. Third, look at the browser. 

<spoiler title="Example of extracting features from the user agent">
```
In : ua = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/56.0.2924.76 Chrome/
     ...: 56.0.2924.76 Safari/537.36'

In : import user_agents

In : ua = user_agents.parse(ua) 

In : ua.is_bot
Out: False

In : ua.is_mobile
Out: False

In : ua.is_pc
Out: True

In : ua.os.family
Out: 'Ubuntu'

In : ua.os.version
Out: ()

In : ua.browser.family
Out: 'Chromium'

In : ua.os.version
Out: ()

In : ua.browser.version
Out: (56, 0, 2924)
```
</spoiler>
As in other domain areas, you can come up with your own features based on guesses about the nature of the data. At the time of this writing, Chromium 56 was new, but after a while, only users who haven't rebooted their browser for a long time will have this version of it. In this case, why not to introduce a feature "lag behind the latest version of the browser"?
 
In addition to the operating system and browser, you can look at the referrer (not always available), [http_accept_language](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Accept-Language) and other meta information.

The next most useful information is IP-address from which you can extract at least the country, and even the city, provider, connection type (mobile/stationary). You need to understand that there is a variety of proxy and outdated databases, so this feature can contain noise. Network administration gurus may try to extract even much more fancy features like suggestions about [using the VPN](https://habrahabr.ru/post/216295/). By the way, the data from the IP-address is well combined with http_accept_language: if the user is sitting at the Chilean proxies and browser locale is ru_RU, something is unclean and worth a 1 in the corresponding column in the table (`is_traveler_or_proxy_user`).

In general, any given area has so much specifics that it can barely fit one head. Therefore, I invite dear readers to share their experience and tell in the comments about feature extraction and generation in their work. 


## Feature transformations

### Normalization and changing distribution

Monotonic transformation of features is critical for some algorithms and has no effect on others. By the way, this is one of the reasons for the popularity of decision trees and all derivatives algorithms (random forest, gradient boosting) - not everyone can/want to tinker with transformations, and these algorithms are robust to unusual distributions.
 
There are also purely engineering reasons: np.log as a way to deal with too large numbers that do not fit in np.float64.But this is an exception rather than the rule; often it's driven by the desire to adapt the dataset to the requirements of the algorithm. Parametric methods usually require a minimum of symmetrical and unimodal distribution of data, which is not always provided by the real world. There may be more stringent requirements (here it's appropriate to recall a lesson about the [linear models](https://habrahabr.ru/company/ods/blog/323890/#1-lineynaya-regressiya)).

However, data requirements are imposed not only by parametric methods: the same [nearest neighbors method](https://habrahabr.ru/company/ods/blog/322534/#metod-blizhayshih-sosedey) will predict the complete nonsense if features are not normalized: one distribution is located in the vicinity of zero and does not go beyond (-1, 1) and the other's range is hundreds and thousands . 

A simple example: suppose that the task is to predict the cost of an apartment by two variables - the distance from the center and the number of rooms. Number of rooms rarely exceeds 5, and the distance from the center in big cities can easily be measured in tens of thousands of meters.

The simplest transformation is Standart Scaling (aka Z-score normalization). 
$$display$$\large z = \frac{x – \mu}{\sigma}$$display$$

StandartScaling doesn't make the distribution normal in the strict sense ...

```
In : from sklearn.preprocessing import StandardScaler  

In : from scipy.stats import beta

In : from scipy.stats import shapiro

In : data = beta(1, 10).rvs(1000).reshape(-1, 1)

In : shapiro(data)
Out: (0.8783774375915527, 3.0409122263582326e-27)
# Value of the statistic, p-value 

In : shapiro(StandardScaler().fit_transform(data))
Out: (0.8783774375915527, 3.0409122263582326e-27)
# With such p-value we'd have to reject the null hypothesis of normality of the data
```
... but protects from outliers in a way
```
In : data = np.array([1, 1, 0, -1, 2, 1, 2, 3, -2, 4, 100]).reshape(-1, 1).astype(np.float64)

In : StandardScaler().fit_transform(data)
Out: 
array([[-0.31922662],
       [-0.31922662],
       [-0.35434155],
       [-0.38945648],
       [-0.28411169],
       [-0.31922662],
       [-0.28411169],
       [-0.24899676],
       [-0.42457141],
       [-0.21388184],
       [ 3.15715128]])

In : (data – data.mean()) / data.std()
Out: 
array([[-0.31922662],
       [-0.31922662],
       [-0.35434155],
       [-0.38945648],
       [-0.28411169],
       [-0.31922662],
       [-0.28411169],
       [-0.24899676],
       [-0.42457141],
       [-0.21388184],
       [ 3.15715128]])
```

Another fairly popular option is MinMax Scaling, which carries all the points to the predetermined interval (usually (0, 1)).
$$display$$\large X_{norm} = \frac{X – X_{min}}{X_{max}-X_{min}}$$display$$

```
In : from sklearn.preprocessing import MinMaxScaler

In : MinMaxScaler().fit_transform(data)
Out: 
array([[ 0.02941176],
       [ 0.02941176],
       [ 0.01960784],
       [ 0.00980392],
       [ 0.03921569],
       [ 0.02941176],
       [ 0.03921569],
       [ 0.04901961],
       [ 0.        ],
       [ 0.05882353],
       [ 1.        ]])

In : (data – data.min()) / (data.max() – data.min())
Out: 
array([[ 0.02941176],
       [ 0.02941176],
       [ 0.01960784],
       [ 0.00980392],
       [ 0.03921569],
       [ 0.02941176],
       [ 0.03921569],
       [ 0.04901961],
       [ 0.        ],
       [ 0.05882353],
       [ 1.        ]])
```
StandartScaling and MinMax Scaling have similar range of applicability and often are more or less interchangeable. However, if the algorithm involves the calculation of distances between points or vectors, the default choice is StandartScaling. But MinMax Scaling is useful for visualization, to bring features to the interval (0, 255).
 
If we assume that some data is not normally distributed, but is described by the [og-normal distribution](https://ru.wikipedia.org/wiki/%D0%9B%D0%BE%D0%B3%D0%BD%D0%BE%D1%80%D0%BC%D0%B0%D0%BB%D1%8C%D0%BD%D0%BE%D0%B5_%D1%80%D0%B0%D1%81%D0%BF%D1%80%D0%B5%D0%B4%D0%B5%D0%BB%D0%B5%D0%BD%D0%B8%D0%B5), it can easily be brought to an honest normal distribution:

```
In : from scipy.stats import lognorm

In : data = lognorm(s=1).rvs(1000)

In : shapiro(data)
Out: (0.05714237689971924, 0.0)

In : shapiro(np.log(data))
Out: (0.9980740547180176, 0.3150389492511749)
```

The lognormal distribution is suitable to describe salaries, price of securities, urban population, number of comments to articles on the internet, etc. However, to apply this procedure, distribution does not necessarily have to be lognormal, you can try to subject any distribution with a heavy right tail to this transformation. Furthermore, one can try to use other similar transformations, being guided by own hypotheses about how to approximate the available distribution to normal. Examples of such transformations are [Box-Cox transformation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html) (logarithm is a special case of the Box-Cox transformation) or [Yeo-Johnson transformation](https://gist.github.com/mesgarpour/f24769cd186e2db853957b10ff6b7a95) that extends the range of applicability to negative numbers; in addition, you can try to simply add a constant to the feature - `np.log (x + const)`.
 
In the examples above, we have worked with synthetic data and strictly tested normality using the Shapiro-Wilk test. Let's try to look at the actual data, and to test for normality using a less formal method - [Q-Q plot](https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot). For normal distribution, it will look like a smooth diagonal line, and visual anomalies are intuitively understandable.

<img src="https://habrastorage.org/files/ad1/3bb/a14/ad13bba14dd541feac9e211ba94c9223.png" align='center'><br>
_Q-Q plot for lognormal distribution_

<img src="https://habrastorage.org/files/f25/215/046/f25215046b8d4f67bea16b7b0faf5884.png" align='center'><br>
_Q-Q plot for the same distribution after taking the logarithm_

<spoiler title="Let's draw plots!">
```
In : import statsmodels.api as sm

# Let's take the price feature from Renthop dataset and filter by hands the most extreme values ​​for clarity
In : price = df.price[(df.price <= 20000) & (df.price > 500)]

In : price_log = np.log(price)

In : price_mm = MinMaxScaler().fit_transform(price.values.reshape(-1, 1).astype(np.float64)).flatten()
# A lot of gestures so that sklearn didn't shower us with warnings

In : price_z = StandardScaler().fit_transform(price.values.reshape(-1, 1).astype(np.float64)).flatten()

In : sm.qqplot(price_log, loc=price_log.mean(), scale=price_log.std()).savefig('qq_price_log.png')

In : sm.qqplot(price_mm, loc=price_mm.mean(), scale=price_mm.std()).savefig('qq_price_mm.png')

In : sm.qqplot(price_z, loc=price_z.mean(), scale=price_z.std()).savefig('qq_price_z.png')
```
</spoiler>

<img src='https://habrastorage.org/files/9ce/9d3/1f6/9ce9d31f6d344e5a9036778cf18bfefb.png' align='center'>

_Q-Q plot of the initial feature_

<img src='https://habrastorage.org/files/a28/bbf/93d/a28bbf93da474fb2b1417f837f460440.png' align='center'>

_Q-Q plot after StandartScaler. Shape doesn't change_

<img src='https://habrastorage.org/files/77b/b6e/fb6/77bb6efb62ba41d19d31f2402a2c4a5c.png' align='center'>

_Q-Q plot after MinMaxScaler. Shape doesn't change_

<img src='https://habrastorage.org/files/946/a83/18c/946a8318cbc9446f95074de39c37030f.png' align='center'>

_Q-Q plot after taking the logarithm. Things are getting better!_

Let's see whether transformations can somehow help the real model. I made a [small script](https://github.com/Yorko/mlcourse_open/blob/master/jupyter_notebooks/topic6_features/demo.py) that reads data from Renthop competition, selects some features (the others are dictatorially thrown for simplicity), and returns us a more or less ready for demonstration data.

<spoiler title="Quite a lot of code">
```
In : from demo import get_data

In : x_data, y_data = get_data()

In : x_data.head(5)
Out: 
        bathrooms  bedrooms     price  dishwasher  doorman  pets  \
10            1.5         3  8.006368           0        0     0   
10000         1.0         2  8.606119           0        1     1   
100004        1.0         1  7.955074           1        0     1   
100007        1.0         1  8.094073           0        0     0   
100013        1.0         4  8.116716           0        0     0   

        air_conditioning  parking  balcony  bike       ...        stainless  \
10                     0        0        0     0       ...                0   
10000                  0        0        0     0       ...                0   
100004                 0        0        0     0       ...                0   
100007                 0        0        0     0       ...                0   
100013                 0        0        0     0       ...                0   

        simplex  public  num_photos  num_features  listing_age  room_dif  \
10            0       0           5             0          278       1.5   
10000         0       0          11            57          290       1.0   
100004        0       0           8            72          346       0.0   
100007        0       0           3            22          345       0.0   
100013        0       0           3             7          335       3.0   

        room_sum  price_per_room  bedrooms_share  
10           4.5      666.666667        0.666667  
10000        3.0     1821.666667        0.666667  
100004       2.0     1425.000000        0.500000  
100007       2.0     1637.500000        0.500000  
100013       5.0      670.000000        0.800000  

[5 rows x 46 columns]

In : x_data = x_data.values

In : from sklearn.linear_model import LogisticRegression

In : from sklearn.ensemble import RandomForestClassifier

In : from sklearn.model_selection import cross_val_score

In : from sklearn.feature_selection import SelectFromModel

In : cross_val_score(LogisticRegression(), x_data, y_data, scoring='neg_log_loss').mean()
/home/arseny/.pyenv/versions/3.6.0/lib/python3.6/site-packages/sklearn/linear_model/base.py:352: RuntimeWarning: overflow encountered in exp
  np.exp(prob, prob)
# It seems something went wrong! In fact, it helps to understand what the problem is

Out: -0.68715971821885724

In : from sklearn.preprocessing import StandardScaler

In : cross_val_score(LogisticRegression(), StandardScaler().fit_transform(x_data), y_data, scoring='neg_log_loss').mean()
/home/arseny/.pyenv/versions/3.6.0/lib/python3.6/site-packages/sklearn/linear_model/base.py:352: RuntimeWarning: overflow encountered in exp
  np.exp(prob, prob)
Out: -0.66985167834479187
# Wow! It really helps!

In : from sklearn.preprocessing import MinMaxScaler

In : cross_val_score(LogisticRegression(), MinMaxScaler().fit_transform(x_data), y_data, scoring='neg_log_loss').mean()
    ...: 
Out: -0.68522489913898188
# And this time - no :(
```
</spoiler>

### Interactions
If previous transformations were rather math-driven, this part is justified more with the nature of the data; it can be attributed to both transformations and creation of new features.
 
Let's come back again to the Two Sigma Connect: Rental Listing Inquires problem. Among the features in this problem are the number of rooms and the price. Worldly wisdom suggests that the cost per single room is more indicative than the total cost - so we can try to make such feature.  

```
rooms = df["bedrooms"].apply(lambda x: max(x, .5))
# Avoid division by zero; .5 is chosen more or less arbitrarily
df["price_per_bedroom"] = df["price"] / rooms
```

You don't have to be guided by the logic of life. If there are not too many features, it is possible to generate all the possible interactions and then weed out the unnecessary ones, using one of the techniques described in the next section. In addition, not all interactions between features must have any physical meaning, for example, (often used in linear models)[https://habrahabr.ru/company/ods/blog/322076/] polynomial features (see [`sklearn.preprocessing.PolynomialFeatures`](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html)) are almost impossible to interpret. 

### Filling in the missing values

Not many algorithms can work with missing values ​​"out of the box", and the real world often provides data with gaps. Fortunately, this is one of the tasks for which one doesn't need any creativity. Both key python libraries for data analysis provide as easy as pie solutions: [`pandas.DataFrame.fillna`](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.fillna.html) and [`sklearn.preprocessing.Imputer`](http://scikit-learn.org/stable/modules/preprocessing.html#imputation).

Ready-made library solutions do not hide any magic behind the scenes. Approaches to handling missing values ​​arise at the level of common sense:
- encode with a separate blank value like `"n/a"` (for categorical variables);
- use the most probable value of the feature (mean or median for the numerical variables, the most common value for categorical);
- on the contrary, encode with some incredible value (good for models based on decision trees, as it allows to make a partition between the missing and non-missing values);
- for ordered data (e.g., time series) it's possible to take the adjacent value - next or previous.

<img src='https://habrastorage.org/files/4b3/f3d/229/4b3f3d229a8447f6aa2ea433d85c57e9.png' align='center'>

Ease of use of library solutions sometimes suggests to stick something like `df = df.fillna(0)` and not sweat the gaps. But this is not the most clever solution: data preparation usually takes most of the time, not building models; thoughtless implicit gaps filling may hide a bug in processing and damage the model.

## Feature selection

Why it may even be necessary to select features? To some, this idea may seem counterintuitive, but in fact there are at least two important reasons to get rid of unimportant features. The first is clear to every engineer: the more data, the higher computational complexity. As long as we frolic with toy datasets, the size of the data is not a problem, but for real loaded productions hundreds of extra features may be quite tangible. Another reason is that some algorithms take noise (non-informative features) for a signal and ovefit. 

### Statistical approaches

The most obvious candidate for cull is a feature which value is unchanged, ie, it contains no information at all. If we take a small step away from this degenerate case, it is reasonable to assume that features with low variance are rather worse than with high variance. So one can come to the idea to cut features with variance below a certain threshold. 

```
In : from sklearn.feature_selection import VarianceThreshold

In : from sklearn.datasets import make_classification

In : x_data_generated, y_data_generated = make_classification()

In : x_data_generated.shape
Out: (100, 20)

In : VarianceThreshold(.7).fit_transform(x_data_generated).shape
Out: (100, 19)

In : VarianceThreshold(.8).fit_transform(x_data_generated).shape
Out: (100, 18)

In : VarianceThreshold(.9).fit_transform(x_data_generated).shape
Out: (100, 15)
```

There are other ways, that are also [based on classical statistics](http://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection).
```
In : from sklearn.feature_selection import SelectKBest, f_classif

In : x_data_kbest = SelectKBest(f_classif, k=5).fit_transform(x_data_generated, y_data_generated)

In : x_data_varth = VarianceThreshold(.9).fit_transform(x_data_generated)

In : from sklearn.linear_model import LogisticRegression

In : from sklearn.model_selection import cross_val_score

In : cross_val_score(LogisticRegression(), x_data_generated, y_data_generated, scoring='neg_log_loss').mean()
Out: -0.45367136377981693

In : cross_val_score(LogisticRegression(), x_data_kbest, y_data_generated, scoring='neg_log_loss').mean()
Out: -0.35775228616521798

In : cross_val_score(LogisticRegression(), x_data_varth, y_data_generated, scoring='neg_log_loss').mean()
Out: -0.44033042718359772
```
It can be seen that the selected features have improved the quality of the classifier. Of course this example is _purely_ artificial, however, the trick is worth checking in real problems.

### Selection by modelling

Another approach: use some baseline model for feature evaluation, and the model should clearly show the importance of the features. Two types of models are usually used: some "wooden" composition (eg, Random Forest) or linear model with Lasso regularization prone to nullify weights of weak features. The logic is intuitive: if features are clearly useless in a simple model, there's no need to drag them to a more complex one.

<spoiler title="Synthetic example">
```
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

x_data_generated, y_data_generated = make_classification()

pipe = make_pipeline(SelectFromModel(estimator=RandomForestClassifier()),
                     LogisticRegression())

lr = LogisticRegression()
rf = RandomForestClassifier()

print(cross_val_score(lr, x_data_generated, y_data_generated, scoring='neg_log_loss').mean())
print(cross_val_score(rf, x_data_generated, y_data_generated, scoring='neg_log_loss').mean())
print(cross_val_score(pipe, x_data_generated, y_data_generated, scoring='neg_log_loss').mean())

-0.184853179322
-0.235652626736
-0.158372952933
```
</spoiler>

We must not forget that this is not a silver bullet - it can make even worse.
<spoiler title="Let's go back to the Renthop dataset.">
```
x_data, y_data = get_data()
x_data = x_data.values

pipe1 = make_pipeline(StandardScaler(),
                      SelectFromModel(estimator=RandomForestClassifier()),
                      LogisticRegression())

pipe2 = make_pipeline(StandardScaler(),
                      LogisticRegression())

rf = RandomForestClassifier()

print('LR + selection: ', cross_val_score(pipe1, x_data, y_data, scoring='neg_log_loss').mean())
print('LR: ', cross_val_score(pipe2, x_data, y_data, scoring='neg_log_loss').mean())
print('RF: ', cross_val_score(rf, x_data, y_data, scoring='neg_log_loss').mean())

LR + selection:  -0.714208124619
LR:  -0.669572736183
# It got only worse! 
RF:  -2.13486716798

```
</spoiler>

### Grid search

Finally, the most reliable, but also the most computationally complex method is based on the trivial grid search: train model on a subset of features, store results, repeat for different subsets, compare the quality of models. This approach is called [Exhaustive Feature Selection](http://rasbt.github.io/mlxtend/user_guide/feature_selection/ExhaustiveFeatureSelector/).

Searching all combinations is usually too long, so you can try to reduce the searching space. We fix a small number N, iterate through all the combinations of N features, choose the best combination, and then iterate through the combination of N + 1 features so that the previous best combination of features is fixed and only new feature is searched. Thus it is possible to sort until we hit a maximum number of characteristics or until the quality of the model ceases to increase significantly. This algorithm is called [Sequential Feature Selection](http://rasbt.github.io/mlxtend/user_guide/feature_selection/SequentialFeatureSelector/).

This algorithm can be unfolded: start with the complete feature space and throw features one by one, until it does not impair the quality of the model, or until the desired number of features is reached.

<spoiler title="Time for leisurely grid search!">
```
In : selector = SequentialFeatureSelector(LogisticRegression(), scoring='neg_log_loss', verbose=2, k_features=3, forward=False, n_jobs=-1)

In : selector.fit(x_data_scaled, y_data)

In : selector.fit(x_data_scaled, y_data)

[2017-03-30 01:42:24] Features: 45/3 -- score: -0.682830838803
[2017-03-30 01:44:40] Features: 44/3 -- score: -0.682779463265
[2017-03-30 01:46:47] Features: 43/3 -- score: -0.682727480522
[2017-03-30 01:48:54] Features: 42/3 -- score: -0.682680521828
[2017-03-30 01:50:52] Features: 41/3 -- score: -0.68264297879
[2017-03-30 01:52:46] Features: 40/3 -- score: -0.682607753617
[2017-03-30 01:54:37] Features: 39/3 -- score: -0.682570678346
[2017-03-30 01:56:21] Features: 38/3 -- score: -0.682536314625
[2017-03-30 01:58:02] Features: 37/3 -- score: -0.682520258804
[2017-03-30 01:59:39] Features: 36/3 -- score: -0.68250862986
[2017-03-30 02:01:17] Features: 35/3 -- score: -0.682498213174
# "It was getting dark. And the old ladies kept falling..."
...
[2017-03-30 02:21:09] Features: 10/3 -- score: -0.68657335969
[2017-03-30 02:21:18] Features: 9/3 -- score: -0.688405548594
[2017-03-30 02:21:26] Features: 8/3 -- score: -0.690213724719
[2017-03-30 02:21:32] Features: 7/3 -- score: -0.692383588303
[2017-03-30 02:21:36] Features: 6/3 -- score: -0.695321584506
[2017-03-30 02:21:40] Features: 5/3 -- score: -0.698519960477
[2017-03-30 02:21:42] Features: 4/3 -- score: -0.704095390444
[2017-03-30 02:21:44] Features: 3/3 -- score: -0.713788301404
# But improvement couldn’t last forever
```
</spoiler>

## Homework

As part of the individual work we invite you to answer a few simple questions: [Jupyter-template](https://github.com/Yorko/mlcourse_open/tree/master/jupyter_notebooks/topic6_features/hw6_features.ipynb), [web-form for answers](https://goo.gl/forms/2LyfudBnL21GZ6a13). As usual, you have one week to complete the task. т.е. ответы принимаются до 10 апреля 23:59 UTC+3. В случае возникновения каких-то сложностей с ответами на вопросы, пишите в Slack-чат Open Data Science (канал #mlcourse_open, для оперативного ответа может быть полезно обратиться к @arsenyinfo). 

Open Data Science желает удачи в выполнении домашней работы, а также чистых данных и полезных признаков в реальной работе!

