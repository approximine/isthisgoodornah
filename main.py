# https://kavita-ganesan.com/how-to-use-countvectorizer/
# https://www.analyticsvidhya.com/blog/2021/06/nlp-sentiment-analysis/
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import regex as re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

df_train = pd.read_csv("archive/train.txt", delimiter=';', names =['text', 'label'])
df_val = pd.read_csv("archive/val.txt", delimiter=';', names=['text', 'label'])

#Concats training data and validation data into one total set of data
# Do this because we can use cross validation on entire set, don't need to divide
df = pd.concat([df_train, df_val])
df.reset_index(inplace=True, drop=True)

#Shows 5 examples of the resulting "df"
print("Shape of dataframe", df.shape)
print(df.sample(5))

#The data is already split into "labels" and "texts"
#This line basically creates a graph to count the amt of "labels"
# sns.countplot(x=df.label)
# plt.show()
#There's 6 labels! For this, they end up squashing into positive and
#   negative sentiment (See below)
def custom_encoder(df):
    # df['label'] = df['label'].str.replace(["sadness"], 1)
    df['label'] = df['label'].replace(["joy"], 1)
    df['label'] = df['label'].replace(['surprise'], 1)
    df['label'] = df['label'].replace(['love'], 1)
    df['label'] = df['label'].replace(['fear'], 0)
    df['label'] = df['label'].replace(['anger'], 0)
    df['label'] = df['label'].replace(['sadness'], 0)

custom_encoder(df)
print(df)
sns.countplot(x=df.label)
# plt.show()

# #Data Pre-processing
# # 1) Iterate through each record, and using a regex, get rid of characters that don't belong to the alphabet
# # 2) convert input strings to lowercase
# #       - This is because "Good" and "good" are two different inputs
# # 3) Check for stopwords and get rid of them
# #       - Stopwords are words that are commonly used in a sentence that don't add value to the sentence
# #       - Used structurally
# # 4) Lemmatization (lemma = base form of word) of words (turning them all into the same base word)
# # 5) return corpus of processed data


# #object of WordNetLemmatizer
lm = WordNetLemmatizer()

def text_transformation(df_col):
    corpus = []
    for item in df_col:
        new_item = re.sub('[^a-zA-Z0-9\\s] ','',str(item)) #Replace anything not in a-z and A-Z with empty space
        new_item = new_item.lower() #convert new_item into lowercases
        new_item = new_item.split() #turns provided string into list
        new_item = [lm.lemmatize(word) for word in new_item if word not in set(stopwords.words('english'))]
        corpus.append(' '.join(str(x) for x in new_item))
    return corpus
# Only want the corpus of the actual data
# Label, index doesn't need to be processed
# for x, y in enumerate(df["text"]):
#     print(str(x) + " " + y)
corpus = text_transformation(df['text'])
# print(corpus)
word_cloud = " ".join(corpus)
string_corpus = str(word_cloud)
# print(corpus)
for row in string_corpus:
    for word in row:
        word_cloud+=" ".join(word)
print("Finished creating string for word cloud")
#Not sure if this is lagging or just... really slow. Check tmrw
wordcloud = WordCloud(width = 500, height = 250, background_color = "white")
wordcloud.generate(string_corpus)
# wordcloud.to_file('wordcloud_output.png')
plt.imshow(wordcloud, interpolation="bilinear")
plt.figure(figsize=[8,10])
plt.axis("off")

print("Starting Count Vectorizing now")
#Scikit-Learn

# After count vectorizing it should... print? cv?? Whatever it is. Is it
# just setting cv to that lmfao
# cv is just a CountVectorizer object built with the idea of ngram_range
cv = CountVectorizer(ngram_range=(1,2))

# print(cv)
# print("Done!")

# we now ask cv to run fit_transform with the vectorizer and the passed in corpus
# output is set to traindata
traindata = cv.fit_transform(corpus)
# print(traindata.toarray())

x = traindata
y = df.label

parameters = {'n_estimators': [500, 1000, 1500],
             'max_depth': [5, 10, None],
             'min_samples_split': [5, 10, 15],
             'min_samples_leaf': [1, 2, 5, 10],
             'bootstrap': [True, False]}

grid_search = GridSearchCV(RandomForestClassifier(),parameters,cv=5,return_train_score=True,n_jobs=-1)
grid_search.fit(x,y)
# print(grid_search.best_params_)

rfc = RandomForestClassifier(max_features=grid_search.best_params_['max_features'],
                                      max_depth=grid_search.best_params_['max_depth'],
                                      n_estimators=grid_search.best_params_['n_estimators'],
                                      min_samples_split=grid_search.best_params_['min_samples_split'],
                                      min_samples_leaf=grid_search.best_params_['min_samples_leaf'],
                                      bootstrap=grid_search.best_params_['bootstrap'])

rfc.fit(x, y)