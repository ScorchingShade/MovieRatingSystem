# coding: utf-8

#Ankush sharma ka code hai, dhyan naal dekho... I dont understand many parts, but thats how they are coded
from numpy import *
import time

#total movies we want

num_movies = 10

# total users bored enough to watch movie
num_users = 5

# we are initialising movie ratings randomly... since movies are 10 and users 5 its a 10x5 matrix
#random comes from numpy
ratings = random.randint(11, size=(num_movies, num_users))#isse sab banega , ye function 1 se 10 random rating banega

###ab yha pe random jo bani ratings unko print kra do, dekho phle kya ratings bani har user ki
print("Creating random ratings for 10 movies and 5 users\n")
print(ratings)
print("\n")


# we need to check if the rating was made or not, so we create a matrix for that... simply not equal 0 will check this
did_rate = (ratings != 0) * 1
print("\nChecking if ratings were made")
print(did_rate)
print("\n")

# ye us bande ki bakchodi hai, iska koi matlab ni velle mei bas shekhi jhad rrha hai
print("\nChecking for null return")
print(ratings != 0)


###iske errors ignore kro, agar isko sahi value ni mili to ye none fenkega bas, move on
#print(ratings != 0) * 1


# dimensions store kra lo taki use kar paao

ratings.shape

##shape shayad numpy ka hai, dimensions ke liye hai
did_rate.shape


# Apni ratings ko ek 10x1  matrix mein rkhte hai, essentially we are a user rating movies now

Damini_ratings = zeros((num_movies, 1))
print("\nNow we are rating our movies in 10x1 matrix")
print(Damini_ratings)


# Python data structures 0 se bante hai, us bande ne ye likha hai but bda bc hai fir 10 pe chala gya khud, out  of bounds kar gya nalla kahi ka
print("\nPrinting Damini wali ratings")
print(Damini_ratings[9])


#let us rate three movies ourselves here...always provide ratings like this
Damini_ratings[0] = 8
Damini_ratings[4] = 7
Damini_ratings[7] = 3
print("\nPrinting new ratings after we have ourselves rated three movies")
print(Damini_ratings)


# Update ratings and did_rate
ratings = append(Damini_ratings, ratings, axis=1)
did_rate = append(((Damini_ratings != 0) * 1), did_rate, axis=1)

print("\nprint updated damini ratings...ugh this is long")
print(ratings)


#fir shape se dimensions lo, aur sab repeat mardo
ratings.shape


#ye bhi
did_rate

#ye bhi
print("\nprinting rate again..")
print(did_rate)


#firse
did_rate.shape


# dekho dataset normalisation hui hai yha, vo kya hota hai padhlo kaam aega....

a = [10, 20, 30]
aSum = sum(a)

#ye no
print(aSum)
print("\n")
#we are creating a mean of the three normalised dataset definitions, phle sum, fir dataset
aMean = aSum / 3
print("Sab dikhao, mean bhi")
print(aMean)
print("\n")

aMean = mean(a)
print(aMean)
print("\n")

a = [10 - aMean, 20 - aMean, 30 - aMean]
print("\nPrinting a values")
print(a)

#now print ratings
print("\nRatings now")
print(ratings)


# a function that normalizes a dataset, aur to kya hi bolen , yahi hai bas
#kya hota hai normal dataset vo padho phle, maths hai
def normalize_ratings(ratings, did_rate):
    num_movies = ratings.shape[0]

    ratings_mean = zeros(shape=(num_movies, 1))
    ratings_norm = zeros(shape=ratings.shape)

    for i in range(num_movies):
        # Get all the indexes where there is a 1
        idx = where(did_rate[i] == 1)[0]
        #  Calculate mean rating of ith movie only from user's that gave a rating
        ratings_mean[i] = mean(ratings[i, idx])
        ratings_norm[i, idx] = ratings[i, idx] - ratings_mean[i]

    return ratings_norm, ratings_mean


# Normalize ratings ab, this is important to proceed to rating prediction

ratings, ratings_mean = normalize_ratings(ratings, did_rate)

# ab yha users update kro, because engine at this time might get confused and needs updating

num_users = ratings.shape[1]
num_features = 3

# linear regression pta hai na? bas uso vectorize kr rhe hai...matlab array use krke vector bnana in py

X = array([[1, 2], [1, 5], [1, 9]])
Theta = array([[0.23], [0.34]])

print("\nPrint X Vector")
print(X)

print("\nPrint Theta vector")
print(Theta)

#relationship between x and theta is defined by this
Y = X.dot(Theta)
print("\nPrint Y")
print(Y)



# Initialize Parameters theta (user_prefs), X (movie_features)

movie_features = random.randn(num_movies, num_features)
user_prefs = random.randn(num_users, num_features)
initial_X_and_theta = r_[movie_features.T.flatten(), user_prefs.T.flatten()]

print("\nPrinting movie features")
print(movie_features)

# In[52]:
print("\n User preferances")
print(user_prefs)


print("\nInitial x and theta values")
print(initial_X_and_theta)

#ab iske dimensions chahiye
initial_X_and_theta.shape

#idk what flatten does honestly...probably gets the dimenions after converting non matricised values
movie_features.T.flatten().shape

#ye bhi vese hi
user_prefs.T.flatten().shape


initial_X_and_theta

###This function takes above defined variables  and transposes the matrix values to make appropriate matrix packaging that we require
def unroll_params(X_and_theta, num_users, num_movies, num_features):
    # Retrieve the X and theta matrixes from X_and_theta, based on their dimensions (num_features, num_movies, num_movies)
    # Get the first 30 (10 * 3) rows in the 48 X 1 column vector
    first_30 = X_and_theta[:num_movies * num_features]
    # Reshape this column vector into a 10 X 3 matrix
    X = first_30.reshape((num_features, num_movies)).transpose()
    # Get the rest of the 18 the numbers, after the first 30
    last_18 = X_and_theta[num_movies * num_features:]
    # Reshape this column vector into a 6 X 3 matrix
    theta = last_18.reshape(num_features, num_users).transpose()
    return X, theta

###gradient hi print hoga...its like differentiation, small changes predict krna, like for every given feature how much choice changes
def calculate_gradient(X_and_theta, ratings, did_rate, num_users, num_movies, num_features, reg_param):
    X, theta = unroll_params(X_and_theta, num_users, num_movies, num_features)

    # we multiply by did_rate because we only want to consider observations for which a rating was given
    difference = X.dot(theta.T) * did_rate - ratings
    X_grad = difference.dot(theta) + reg_param * X
    theta_grad = difference.T.dot(X) + reg_param * theta

    # wrap the gradients back into a column vector
    return r_[X_grad.T.flatten(), theta_grad.T.flatten()]


##zyada cost hogi to matlab vo choice bekar hai, it calculates that, cant explain on text
def calculate_cost(X_and_theta, ratings, did_rate, num_users, num_movies, num_features, reg_param):
    X, theta = unroll_params(X_and_theta, num_users, num_movies, num_features)

    # we multiply (element-wise) by did_rate because we only want to consider observations for which a rating was given
    cost = sum((X.dot(theta.T) * did_rate - ratings) ** 2) / 2
    # '**' means an element-wise power
    regularization = (reg_param / 2) * (sum(theta ** 2) + sum(X ** 2))
    return cost + regularization


#ab iske baad bhayankar cheezen hai, bas stupidly copy this, ye mere bhi bas ke bahar hai bht kuch, full maths hai
# import these for advanced optimizations (like gradient descent)

from scipy import optimize


# regularization paramater

reg_param = 30


# perform gradient descent, find the minimum cost (sum of squared errors) and optimal values of X (movie_features) and Theta (user_prefs)

minimized_cost_and_optimal_params = optimize.fmin_cg(calculate_cost, fprime=calculate_gradient, x0=initial_X_and_theta,
                                                     args=(
                                                         ratings, did_rate, num_users, num_movies, num_features,
                                                         reg_param),
                                                     maxiter=100, disp=True, full_output=True)


###yha cost nikali and optimal movie features bhi...
cost, optimal_movie_features_and_user_prefs = minimized_cost_and_optimal_params[1], minimized_cost_and_optimal_params[0]

# unroll once again
#taki sabki packaging sahi rhe
movie_features, user_prefs = unroll_params(optimal_movie_features_and_user_prefs, num_users, num_movies, num_features)

print("\nPrinting new update to movie features:")
print(movie_features)

print("\nPrinting new update to user_prefs")
print(user_prefs)



# Make some predictions (movie recommendations). Dot product
#yha hua na jadu, dot product uses all the params
all_predictions = movie_features.dot(user_prefs.T)

print("\nPrinting all predictions")
print(all_predictions)



# adding back the ratings_mean column vector to baba ke predictions
print("\n Daminis predictions are being calculated")
time.sleep(4)
predictions_for_Damini = all_predictions[:, 0:1] + ratings_mean

print("\nPrinting Damini's predictions, you should only watch these highest rated movies bro!")
print(predictions_for_Damini)


print("\nPrinting Damini's initial ratings")
print(Damini_ratings)


