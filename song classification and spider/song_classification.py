from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
import sys
import pandas as pd
import json
import numpy as np
from scrapy.crawler import CrawlerRunner
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.decomposition import NMF, LatentDirichletAllocation


######finishing the spidering process ####

my_search = ["emimen", "taylor swift", "kings of leon", "imagine dragons"]

data = pd.read_json("output_metro.json")

data = data[data["lyrics"] != "1"]

data.groupby("artist")["song_name"].count()

# vectorize lyrics
nlp = spacy.load('en_core_web_md')

####my functions ###


def clean_text(review, model):
    """preprocess a string (tokens, stopwords, lowercase, lemma & stemming) returns the cleaned result
        params: review - a string
                model - a spacy model

        returns: list of cleaned tokens
    """

    new_doc = ''
    doc = model(review)
    for word in doc:
        if not word.is_stop and word.is_alpha:
            new_doc = f'{new_doc} {word.lemma_.lower()}'

    return new_doc


def transform_data(df, my_search=my_search):
    my_search = my_search
    upper_case_search = []
    for s in my_search:
        upper_case_search.append(s.upper())
    df["upper_case_artist"] = [x.upper() for x in df["artist"]]

    df = df[data["upper_case_artist"].isin(upper_case_search)]
    df.drop("upper_case_artist", axis=1, inplace=True)

    def f(x): return x["song_name"].split("- ")[1].split("Lyrics")[0]

    df["song_name"] = df.apply(f, axis=1)

    df["lyrics_new"] = ["0"] * len(df)
    df.reset_index(inplace=True)

    df.drop("index", axis=1, inplace=True)

    for i in range(df.shape[0]):
        df["lyrics_new"][i] = clean_text(df["lyrics"].iloc[i], nlp)

    return df


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()


def model_transform_fit(X_train, y_train, resample=False, sample_size1=400, sample_size2=400, sample_size3=400):
    '''Transforms the data for model input.
    Uses Oversampler and tfidf vectorizers for creation of bag of words.

    #####Parameters:#####
        X_train Pandas Data DataFrame
        y_train Pandas DataFrame
        resample boolean (if you want to oversample you data)
        sample sizes  int, what shoudl be the new sample sizes

    ####Result#####
        X_train and y_train transformed
        '''

    if resample:
        rand_over_samp = RandomOverSampler(sampling_strategy={"Kings of Leon": sample_size1,
                                                              "Imagine Dragons": sample_size2,
                                                              "Taylor Swift": sample_size3})
        X_resampled, y_resampled = rand_over_samp.fit_resample(
            np.array(X_train).reshape(-1, 1), y_train)

        X_train = pd.DataFrame(X_resampled)
        X_train.reset_index(inplace=True)
        X_train.drop("index", axis=1, inplace=True)
        X_train = X_train.stack()
        y_train = y_resampled

    vectorizer = TfidfVectorizer(ngram_range=(
        1, 1), stop_words="english", max_df=0.6)
    vectorizer.fit(X_train)
    X_train = vectorizer.transform(X_train)
    X_train = pd.DataFrame(
        X_train.todense(), columns=vectorizer.get_feature_names())

    y_train, unique_codes = pd.factorize(y_train)

    return X_train, y_train, unique_codes, vectorizer


data = transform_data(data)


data.groupby("artist")["song_name"].count()

total = data["song_name"].count()

data.groupby("artist")["song_name"].nunique() / total

# vectorizing data for data vizualization
viz_vec = CountVectorizer()
viz_vec.fit(data["lyrics_new"])
X_trans_viz = viz_vec.transform(data)
X_trans_viz = pd.DataFrame(
    X_trans_viz.todense(), columns=viz_vec.get_feature_names())

data_viz = pd.concat([X_trans_viz, data["artist"]], axis=1)

lda = LatentDirichletAllocation(10, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)

lda.fit(X_trans_viz)
print("\nTopics in LDA model:")
tf_feature_names = viz_vec.get_feature_names()
print_top_words(lda, tf_feature_names, 20)

# Splitting data in X and Y
X = data["lyrics_new"]
y = data.artist


X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42)

X_train.shape, y_train.shape


# investigate imbalance in the data

X_all_train = pd.concat([X_train, y_train], axis=1)
total = X_all_train.count()
X_all_train.groupby("artist").count()/total
X_all_train.groupby("artist").count()


# transforming the data, resampling if necessary and vectorizing the data

# data with resampling
X_train_re, y_train_re, unique_codes, vectorizer_rebalanced = model_transform_fit(
    X_train, y_train, resample=True)


# implement model_nlp Random Forest###
params = {"max_depth": [10, 20, 100],
          "n_estimators": [100, 200, 500]}

rf = RandomForestClassifier(n_estimators=100, max_depth=20)
grid = GridSearchCV(rf, params, cv=5)
grid.fit(X_train, y_train)


grid.score(X_train, y_train)
pred_rf = grid.predict(X_train)
print(classification_report(y_train, pred_rf, target_names=unique_codes))
# end grid fit


# fit best params model
rf = RandomForestClassifier(n_estimators=500, max_depth=200)
rf.fit(X_train_re, y_train_re)

pred_rf = rf.predict(X_train_re)
print(classification_report(y_train_re, pred_rf, target_names=unique_codes))

# plot fetaure importance
pd.DataFrame(rf.feature_importances_[
             rf.feature_importances_ > 0.005], index=X_train_re.columns[rf.feature_importances_ > 0.005]).plot.barh()

# implementing linear regression
logreg = LogisticRegression(
    class_weight="balanced", C=0.9, multi_class="multinomial", solver="saga",
    random_state=42)
logreg.fit(X_train_re, y_train_re)
logreg.score(X_train_re, y_train_re)
pred_logreg = logreg.predict(X_train_re)

print(classification_report(y_train_re, pred_logreg, target_names=unique_codes))

# implement a naive bayes approach
nB = MultinomialNB()
nB.fit(X_train_re, y_train_re)
pred_nB = nB.predict(X_train_re)
print(classification_report(y_train_re, pred_nB, target_names=unique_codes))


###no rebalancing ###

X_train_, y_train_, unique_codes, vectorizer = model_transform_fit(
    X_train, y_train)

# fit random forest
rf_2 = RandomForestClassifier(n_estimators=500, max_depth=200)
rf_2.fit(X_train_, y_train_)
pred_rf_no_resmp = rf_2.predict(X_train_)

print(classification_report(y_train_,
                            pred_rf_no_resmp, target_names=unique_codes))

# fit a logistic model
logreg_2 = LogisticRegression(
    class_weight="balanced", C=0.9, multi_class="multinomial", solver="saga",
    random_state=42)
logreg_2.fit(X_train_, y_train_)
pred_logreg_no_resmapling = logreg_2.predict(X_train_)
print(classification_report(y_train_,
                            pred_logreg_no_resmapling, target_names=unique_codes))

# implement naive bayes
# dosn't perform at all good with no resampling
nB_2 = MultinomialNB()
nB_2.fit(X_train_, y_train_)
pred_nB_new = nB_2.predict(X_train_)
print(classification_report(y_train_, pred_nB_new, target_names=unique_codes))


# transform X_test

X_test_trans = vectorizer.transform(X_test)
X_test_trans = pd.DataFrame(
    X_test_trans.todense(), columns=vectorizer.get_feature_names())

X_test_trans.head()

# transform y
y_test_trans, unique_codes_test = pd.factorize(y_test)


pred_logreg_test = logreg_2.predict(X_test_trans)
rf_pred_test = rf_2.predict(X_test_trans)
nB_ped_test = nB_2.predict(X_test_trans)

# results
print(classification_report(y_test_trans,
                            pred_logreg_test, target_names=unique_codes))
print(classification_report(y_test_trans,
                            rf_pred_test, target_names=unique_codes))
print(classification_report(y_test_trans,
                            nB_ped_test, target_names=unique_codes))
