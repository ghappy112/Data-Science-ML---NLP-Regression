# load and use the model the to predict the time an article was written using the article's title as input
import dill
model = dill.load(open(r'nlp_regressor', 'rb'))
print("Please enter the title of the article:")
title = input()
prediction = model.predict([title])[0]
print("Predicted time that the article was written (as a type of julian date):", prediction)
