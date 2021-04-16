# Gregory Happ's NLP & ML Project: Predicting the time an article was written using the article's title

# load data
import dask.dataframe as dd
df = dd.read_csv(r"Eluvio_DS_Challenge.csv")

# look at data
print(df.head(), df.tail(), df.describe().compute(), df.isnull().compute().any(), df.isna().compute().any())

# get variables of interest
X = df["title"].compute()
y = df["time_created"].compute()

# shuffle
from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state=42)

# train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# the model
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDRegressor
est = Pipeline([('nlp', CountVectorizer(ngram_range=(1, 1), binary=True)), ('sgd', SGDRegressor(max_iter=9999))])

# train model
est.fit(X_train, y_train)
print("training score:", est.score(X_train, y_train))

# test model
print("testing score:", est.score(X_test, y_test))

# final model to be exported
est = Pipeline([('nlp', CountVectorizer(ngram_range=(1, 1), binary=True)), ('sgd', SGDRegressor(max_iter=9999))])
est.fit(X, y)
print("training (entire dataset) score:", est.score(X, y))

# save model
import dill
dill.dump(est, open('nlp_regressor', mode='wb'))
