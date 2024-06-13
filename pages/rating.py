# import module
import streamlit as st
import pandas as pd
import numpy as np

# rating to-do list
# svc classification model
# sentient analysis
# filter out positive comments

st.header("Community Rating")
df = pd.read_csv("./dataset/comments.csv")

st.subheader("Dataset")
st.write(df.head())
st.bar_chart(df.groupby(["category"]).size())

# for testing
# just use LLM to generate fake comments and ratings to train
# "give me another 20 short website review comments about x accessibility problems, i want this to be csv format with just one column and don't put semicolons"

st.text("")
st.text("")
st.text("")
st.subheader("Give Comment")
comment = st.text_input("Website improvements to reduce reliance on screen reader summariser?", value="there is a thing flying around the screen")



# sentiment analysis
st.text("")
st.text("")
st.text("")
st.subheader("Sentiment Analysis")
# results

# filter mood of comment
st.subheader("Evaluation")
st.write("Mood of comment: Positive / Negative")

# filter invalid feedback



# svc classification model
st.text("")
st.text("")
st.text("")
st.subheader("SVC Topic Classification")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.comment).toarray()
labels = df.category
st.write("Total unigrams/bigrams: "+str(features.shape))

# learn the terms associated with each category
from sklearn.feature_selection import chi2

categories = ["colour", "size", "animation"]

for category in categories:
    N = 10
    features_chi2 = chi2(features, labels == category)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names_out())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    with st.expander("Unigrams/Bigrams for "+category+" issues"):
        st.write("Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
        st.write("Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

X_train, X_test, y_train, y_test = train_test_split(df['comment'], df['category'], random_state = 0)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = SVC(kernel='linear').fit(X_train_tfidf, y_train)

# results
st.subheader("Evaluation")
st.write("Issue category: "+str(clf.predict(count_vect.transform([comment]))))
