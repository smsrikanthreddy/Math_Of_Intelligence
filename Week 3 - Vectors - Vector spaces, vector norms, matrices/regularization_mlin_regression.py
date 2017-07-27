import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load the IMDB data
movie_data = pd.read_csv("movie_metadata.csv")
#print(movie_data.head())
#print('shape of movie data is:-', np.shape(movie_data))

#Data Cleaning and Summary statistics
movie_data = movie_data.dropna(subset = ['gross'])
#print('shape of movie data after drop is:-', np.shape(movie_data))
movie_data = movie_data[movie_data['country'] == "USA"] #related to US country movies
movie_data[['gross']] = (1.025**(2017-movie_data['title_year']))*movie_data['gross']
np.random.seed(2017)
movie_data["uniform"] = list(np.random.uniform(0, 1, len(movie_data.index)))
movie_data = movie_data[movie_data["uniform"] < 0.1]
movie_data = movie_data[['gross', 'imdb_score']]
#print('movie dataset shape is;-', np.shape(movie_data))
#print('desciption of new data:-',movie_data.describe())

# Visualize data
plt.scatter(movie_data['imdb_score'], movie_data['gross'])
plt.title('IMDB Rating and Gross Sales')
plt.ylabel('Gross sales revenue ($ millions)')
plt.xlabel('IMDB Rating (0 - 10)')
#plt.show()

#building the model
Y = np.array(movie_data['gross'])
X = np.array(movie_data['imdb_score'])
bias0 = np.ones(len(Y))
theta = np.random.choice([0, 1], size=len(Y))
print('shape of weights:-',np.shape(theta), 'shape of bias:-', np.shape(bias0), 'shape of input and output',
      np.shape(X), np.shape(Y),'np.transpose(X):-', np.shape(np.transpose(X)))
#print(w1)

N = float(len(X))
learning_rate = 0.0000001
iterations = 5
lambda_value = 500

def model_building(X, Y, bias0, w1):
    Y_mul = (1/(2 * N)) * (np.sum(np.square(np.transpose(X).dot(w1) - Y)) + lambda_value * np.sum(w1))
    #print(Y_mul)
    loss = Y - Y_mul
    print(sum(loss))
    w1 = gradient_building(X, Y, w1, bias0, lambda_value)
    return w1

def gradient_building(X, Y, w1, bias0, lambda_value):
    w1_gradient = w1 - learning_rate(1/N * np.sum((np.transpose(X).dot(w1) - Y) * np.transpose(X)) +
                                     ((lambda_value/N) * (w1)))
    return w1_gradient

for i in range(iterations):
    w1 = theta
    theta = model_building(X, Y, bias0, w1)
