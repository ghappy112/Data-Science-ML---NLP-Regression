# Data-Science-ML---NLP-Regression
Option 1 - Data Science/ML
Gregory Happ

My project was predicting the time an article was written using the article's title as input for a machine learning model. Dask was used to load the dataset, then scikit-learn was used for both the natural language processing and the machine learning. The chosen model was a stochastic gradient descent regressor. Finally, I saved the model using dill, so it can be conveniently deployed.

train_and_save_model.py: loads the dataset with dask, then trains the machine learning model. Finally, it saves the model as a dill file.

nlp_regressor: the model itself. Should be opened with dill.

load_model: a demo that loads and uses the model
