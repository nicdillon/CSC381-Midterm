'''
Recommender System
Source info: PCI, TS, 2007, 978...

Author/Collaborator: Carlos Seminario

Researcher: Nic Dillon

'''

import os
import statistics
import numpy as np
import matplotlib.pyplot as plt
import pickle
from math import sqrt

def from_file_to_dict(path, datafile, itemfile):
    ''' Load user-item matrix from specified file

        Parameters:
        -- path: directory path to datafile and itemfile
        -- datafile: delimited file containing userid, itemid, rating
        -- itemfile: delimited file that maps itemid to item name

        Returns:
        -- prefs: a nested dictionary containing item ratings for each user

    '''

    # Get movie titles, place into movies dictionary indexed by itemID
    movies={}
    try:
        with open (path + '/' + itemfile) as myfile:
            # this encoding is required for some datasets: encoding='iso8859'
            for line in myfile:
                (id,title)=line.split('|')[0:2]
                movies[id]=title.strip()

    # Error processing
    except UnicodeDecodeError as ex:
        print (ex)
        print (len(movies), line, id, title)
        return {}
    except Exception as ex:
        print (ex)
        print (len(movies))
        return {}

    # Load data into a nested dictionary
    prefs={}
    for line in open(path+'/'+ datafile):
        #print(line, line.split('\t')) #debug
        (user,movieid,rating,ts)=line.split('\t')
        user = user.strip() # remove spaces
        movieid = movieid.strip() # remove spaces
        prefs.setdefault(user,{}) # make it a nested dicitonary
        prefs[user][movies[movieid]]=float(rating)

    #return a dictionary of preferences
    return prefs

def data_stats(dictionaryName, ratingsFilename):

    numUsers = 0
    numItems = 0
    numRatings = 0
    averageRating = 0.0
    averageItemRating = 0.0
    ratingsSparsity = 0.0
    itemNames = []
    itemRatings = {}
    print("Aerage ratings for each user:")
    for user in dictionaryName:
        numUsers += 1
        averageUserRating = 0.0
        tempItems = 0
        tempRatings = []
        for item in dictionaryName[user]:
            tempItems += 1
            if (item not in itemRatings):
                itemRatings[item] = [dictionaryName[user][item]]
                numItems += 1
            else:
                itemRatings[item].append(dictionaryName[user][item])

            numRatings += 1
            averageRating += float(dictionaryName[user][item])
            tempRatings.append(float(dictionaryName[user][item]))
        print("%s's average rating and deviation: %.2f, %.2f" % (user,
            statistics.mean(tempRatings), statistics.stdev(tempRatings)))

    print("***************")
    movieRatings = []
    print("Average ratings for each item:")
    for item in itemRatings:
        print("Average rating for %s: %.2f, %.2f" % (item,
            float(statistics.mean(itemRatings[item])),
            statistics.stdev(itemRatings[item])))
        for rating in itemRatings[item]:
            movieRatings.append(rating)
    averageRating = averageRating / float(numRatings)

    print("***************")
    print("General statistics:")
    print("Number of users: %d" % numUsers)
    print("Number of items: %d" % numItems)
    print("Number of ratings: %d" % numRatings)
    print("Average rating: %.2f" % averageRating)
    print("Ratings sparsity: %.2f" %
        (1 - float(numRatings)/(numItems * numUsers)))

    a = np.array(movieRatings)

    _ = plt.hist(a, bins=5)  # arguments are passed to np.histogram
    plt.title("Ratings Histogram")
    plt.show()

def popular_items(dictionaryName, ratingsFilename):


    movies = {}
    bestRatedMovies = {}
    highestRatedMovies = {}
    mostRatedMovies = {}
    #make dictionary for movies and their ratings
    for user in dictionaryName:
        for movie in dictionaryName[user]:
            if (movie in movies):
                movies[movie].append(dictionaryName[user][movie])
            else:
                movies[movie] = [dictionaryName[user][movie]]

    moviesCopy = movies

    for i in range(5):
        bestMovie = []
        mostRatedMovie = []
        highestRatedMovie = []
        for movie in moviesCopy:
            if bestMovie == []:
                if(movie not in highestRatedMovies):
                    highestRatedMovie = [movie, statistics.mean(moviesCopy[movie]),
                    len(moviesCopy[movie])]
                if(movie not in bestRatedMovies):
                    bestMovie = [movie, statistics.mean(moviesCopy[movie])]
            if mostRatedMovie == []:
                if(movie not in mostRatedMovies):
                    mostRatedMovie = [movie, statistics.mean(moviesCopy[movie]),
                    len(moviesCopy[movie])]
            if (bestMovie != []):
                if ((statistics.mean(moviesCopy[movie]) >= bestMovie[1]) &
                (movie not in bestRatedMovies)):
                    bestMovie = [movie, statistics.mean(moviesCopy[movie])]
            if (mostRatedMovie != []):
                if ((len(moviesCopy[movie]) > mostRatedMovie[2]) &
                (movie not in mostRatedMovies)):
                    mostRatedMovie = [movie, statistics.mean(moviesCopy[movie]),
                    len(moviesCopy[movie])]
            if (highestRatedMovie != []):
                if ((statistics.mean(moviesCopy[movie]) > highestRatedMovie[1]) &
                (movie not in highestRatedMovies)):
                    highestRatedMovie = [movie, statistics.mean(moviesCopy[movie]),
                    len(moviesCopy[movie])]
        if(highestRatedMovie[2] > 5):
            highestRatedMovies[highestRatedMovie[0]] = highestRatedMovie[1:]

        bestRatedMovies[bestMovie[0]] = bestMovie[1]
        mostRatedMovies[mostRatedMovie[0]] = [mostRatedMovie[2]]

    print("**********")
    print("The best rated movies:")
    for movie in bestRatedMovies:
        print("%s: Rated %0.2f" % (movie, bestRatedMovies[movie]))
    print("**********")
    print("The most rated movies:")
    for movie in mostRatedMovies:
        print("%s: %d ratings" % (movie, mostRatedMovies[movie][0]))
    print("**********")
    print("The best rated movies with 6 or more ratings:")
    for movie in highestRatedMovies:
        print("%s: %d ratings with an average rating of %.2f" %
        (movie, highestRatedMovies[movie][1], highestRatedMovies[movie][0]))
## add tbis function to the other set of functions
# Returns a distance-based similarity score for person1 and person2
def sim_distance(prefs,person1,person2):
    '''
        Calculate Euclidean distance similarity

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person1: string containing name of user 1
        -- person2: string containing name of user 2

        Returns:
        -- Euclidean distance similarity as a float

    '''

    # Get the list of shared_items
    si={}
    for item in prefs[person1]:
        if item in prefs[person2]:
            si[item]=1

    # if they have no ratings in common, return 0
    if len(si)==0:
        return 0

    # Add up the squares of all the differences
    sum_of_squares=sum([pow(prefs[person1][item]-prefs[person2][item],2)
                        for item in prefs[person1] if item in prefs[person2]])

    # sum_of_squares = 0
    # for item in prefs[person1]:
    #     if item in prefs[person2]:
    #         #print(item, prefs[person1][item], prefs[person2][item])
    #         sq = pow(prefs[person1][item]-prefs[person2][item],2)
    #         #print (sq)
    #         sum_of_squares += sq

    return 1/(1+sqrt(sum_of_squares))

def sim_pearson(prefs,p1,p2):
    '''
        Calculate Pearson Correlation similarity

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person1: string containing name of user 1
        -- person2: string containing name of user 2

        Returns:
        -- Pearson Correlation similarity as a float

    '''

    ## place your code here!
    ##
    ## REQUIREMENT! For this function, calculate the pearson correlation
    ## "longhand", i.e, calc both numerator and denominator as indicated in the
    ## formula. You can use sqrt (from math module), and average from numpy.
    ## Look at the sim_distance() function for ideas.
    ##
    p1Movies = []
    p2Movies = []
    numerator = 0.0
    p1Den = 0.0
    p2Den = 0.0
    relationValue = 0.0

    for movie1 in prefs[p1]:
        for movie2 in prefs[p2]:
            if movie1 == movie2:
                p1Movies.append(prefs[p1][movie1])
                p2Movies.append(prefs[p2][movie2])


    p1Average = statistics.mean(p1Movies)
    p2Average = statistics.mean(p2Movies)


    for i in range(len(p1Movies)):
        # if((p1Movies[i] - p1Average) != 0 and (p2Movies[i] - p2Average) != 0):
        numerator += (p1Movies[i] - p1Average)*(p2Movies[i] - p2Average)
        p1Den += (p1Movies[i] - p1Average)**2
        p2Den += (p2Movies[i] - p2Average)**2
    if(sqrt(p1Den) * sqrt(p2Den) > 0 and numerator > 0):
        relationValue = (numerator) / (sqrt(p1Den) * sqrt(p2Den))

    return relationValue


def getRecommendations(prefs,person,similarity):
    '''
        Calculates recommendations for a given user

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person: string containing name of user
        -- similarity: function to calc similarity (sim_pearson is default)

        Returns:
        -- A list of recommended items with 0 or more tuples,
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.

    '''

    totals={}
    simSums={}
    for other in prefs:
      # don't compare me to myself
        if other==person:
            continue
        sim=similarity(prefs,person,other)

        # ignore scores of zero or lower
        if sim<=0: continue
        for item in prefs[other]:

            # only score movies I haven't seen yet
            if item not in prefs[person] or prefs[person][item]==0:
                # Similarity * Score
                totals.setdefault(item,0)
                totals[item]+=prefs[other][item]*sim
                # Sum of similarities
                simSums.setdefault(item,0)
                simSums[item]+=sim

    # Create the normalized list
    rankings=[(total/simSums[item],item) for item,total in totals.items()]

    # Return the sorted list
    rankings.sort()
    rankings.reverse()
    return rankings

def get_all_UU_recs(prefs, sim, num_users=10, top_N=5):
    # print("working")
    for user in prefs:
        recLimit = top_N
        if num_users == 0:
            break
        else:
            num_users -= 1

        recom = getRecommendations(prefs, user, sim)
        print("     %s:" % user)
        for rec in recom:
            if recLimit == 0:
                break
            else:
                recLimit -= 1
            print("       * %s:  %.3f" %(rec[1], rec[0]))

# Compute Leave_One_Out evaluation
def loo_cv(prefs, metric, sim, algo):
    """
    Leave_One_Out Evaluation: evaluates recommender system ACCURACY

     Parameters:
         prefs dataset: critics, ml-100K, etc.
	 metric: MSE, MAE, RMSE, etc.
	 sim: distance, pearson, etc.
	 algo: user-based recommender, item-based recommender, etc.

    Returns:
         error_total: MSE, MAE, RMSE totals for this set of conditions
	 error_list: list of actual-predicted differences

    Create a temp copy of prefs
    For each user in prefs:
       for item in each user's profile:
          delete this item
          get recommendation (aka prediction) list
	  select the recommendation for this item from the list returned
          calc error, save into error list
	  restore this item
    return mean error, error list
    """
    error_list = []
    error = 0.0
    prefsCopy = prefs.copy()
    for user in prefs:
        prefsCopy[user] = prefs[user].copy()
    for user in prefs:
        for movie in prefs[user]:
            rating = prefs[user][movie]
            del prefsCopy[user][movie]
            for item in getRecommendations(prefsCopy, user, sim):
                if(movie in item):
                    #get difference in value of recommendation rating to actual rating
                    if metric != "MAE":
                        error_list.append((rating - item[0])**2)
                    else:
                        error_list.append(abs(rating - item[0]))
            # getRecommendations(prefsCopy, user, sim)
            prefsCopy[user][movie] = rating

    # print(len(error_list))
    error = statistics.mean(error_list)
    if metric == "RMSE":
        error = sqrt(error)
    return error, error_list

def topMatches(prefs,person,similarity=sim_pearson, n=5):
    '''
        Returns the best matches for person from the prefs dictionary

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person: string containing name of user
        -- similarity: function to calc similarity (sim_pearson is default)
        -- n: number of matches to find/return (5 is default)

        Returns:
        -- A list of similar matches with 0 or more tuples,
           each tuple contains (similarity, item name).
           List is sorted, high to low, by similarity.
           An empty list is returned when no matches have been calc'd.

    '''
    scores=[(similarity(prefs,person,other),other)
                    for other in prefs if other!=person]
    scores.sort()
    scores.reverse()
    return scores[0:n]

def transformPrefs(prefs):
    '''
        Transposes U-I matrix (prefs dictionary)

        Parameters:
        -- prefs: dictionary containing user-item matrix

        Returns:
        -- A transposed U-I matrix, i.e., if prefs was a U-I matrix,
           this function returns an I-U matrix

    '''
    result={}
    for person in prefs:
        for item in prefs[person]:
            result.setdefault(item,{})
            # Flip item and person
            result[item][person]=prefs[person][item]
    return result

def calculateSimilarItems(prefs,n=10,similarity=sim_pearson):

    '''
        Creates a dictionary of items showing which other items they are most
        similar to.

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- n: number of similar matches for topMatches() to return
        -- similarity: function to calc similarity (sim_pearson is default)

        Returns:
        -- A dictionary with a similarity matrix

    '''
    result={}
    # Invert the preference matrix to be item-centric
    itemPrefs=transformPrefs(prefs)
    c=0
    for item in itemPrefs:
      # Status updates for larger datasets
        c+=1
        if c%100==0:
            print ("%d / %d") % (c,len(itemPrefs))

        # Find the most similar items to this one
        scores=topMatches(itemPrefs,item,similarity,n=n)
        result[item]=scores
    return result

# Gets recommendations for a person by using a weighted average
# of every other similar item's ratings

def getRecommendedItems(prefs,itemMatch,user):
    '''
        Calculates recommendations for a given user

        Parameters:
        -- prefs: dictionary containing user-item matrix
        -- person: string containing name of user
        -- similarity: function to calc similarity (sim_pearson is default)

        Returns:
        -- A list of recommended items with 0 or more tuples,
           each tuple contains (predicted rating, item name).
           List is sorted, high to low, by predicted rating.
           An empty list is returned when no recommendations have been calc'd.

    '''
    userRatings=prefs[user]
    scores={}
    totalSim={}
    # Loop over items rated by this user
    for (item,rating) in userRatings.items( ):

      # Loop over items similar to this one
        for (similarity,item2) in itemMatch[item]:

            # Ignore if this user has already rated this item
            if item2 in userRatings: continue
            # ignore scores of zero or lower
            if similarity<=0: continue
            # Weighted sum of rating times similarity
            scores.setdefault(item2,0)
            scores[item2]+=similarity*rating
            # Sum of all the similarities
            totalSim.setdefault(item2,0)
            totalSim[item2]+=similarity

    # Divide each total score by total weighting to get an average

    rankings=[(score/totalSim[item],item) for item,score in scores.items( )]

    # Return the rankings from highest to lowest
    rankings.sort( )
    rankings.reverse( )
    return rankings

def get_all_II_recs(prefs, itemsim, sim_method, num_users=10, top_N=5):
    '''
    Print item-based CF recommendations for all users in dataset

    Parameters
    -- prefs: U-I matrix (nested dictionary)
    -- itemsim: item-item similarity matrix (nested dictionary)
    -- sim_method: name of similarity method used to calc sim matrix (string)
    -- num_users: max number of users to print (integer, default = 10)
    -- top_N: max number of recommendations to print per user (integer, default = 5)

    Returns: None

    '''

    for user in prefs:
        recLimit = top_N
        if num_users == 0:
            break
        else:
            num_users -= 1

        recom = getRecommendedItems(prefs, itemsim, user)
        print("     %s, %s:" % (user, sim_method))
        for rec in recom:
            if recLimit == 0:
                break
            else:
                recLimit -= 1
            print("       * %s:  %.3f" %(rec[1], rec[0]))
    ## see comments in main() re coding this function

def loo_cv_sim(prefs, metric, sim, algo, sim_matrix):
    """
    Leave-One_Out Evaluation: evaluates recommender system ACCURACY

     Parameters:
         prefs dataset: critics, etc.
	 metric: MSE, or MAE, or RMSE
	 sim: distance, pearson, etc.
	 algo: user-based recommender, item-based recommender, etc.
         sim_matrix: pre-computed similarity matrix

    Returns:
         error_total: MSE, or MAE, or RMSE totals for this set of conditions
	 error_list: list of actual-predicted differences
    """
    error_list = []
    error = 0.0
    prefsCopy = prefs.copy()
    for user in prefs:
        prefsCopy[user] = prefs[user].copy()
    for user in prefs:
        for movie in prefs[user]:
            rating = prefs[user][movie]
            del prefsCopy[user][movie]
            for item in algo(prefsCopy, sim_matrix, user):
                if(movie in item):
                    #get difference in value of recommendation rating to actual rating
                    if metric != "MAE":
                        error_list.append((rating - item[0])**2)
                    else:
                        error_list.append(abs(rating - item[0]))
            # getRecommendations(prefsCopy, user, sim)
            prefsCopy[user][movie] = rating

    # print(len(error_list))
    error = statistics.mean(error_list)
    if metric == "RMSE":
        error = sqrt(error)
    return error, error_list



def main():
    ''' User interface for Python console '''

    # Load critics dict from file
    path = os.getcwd() # this gets the current working directory
                       # you can customize path for your own computer here
    print('\npath: %s' % path) # debug
    done = False
    prefs = {}
    itemsim = {} ## make sure this in your main()


    while not done:
        print()
        # Start a simple dialog
        file_io = input('R(ead) critics data from file?, '
                        'P(rint) the U-I matrix?, '
                        'V(alidate) the dictionary?, '
                        'S(tats)?, \n'
                        'D(istance) critics data?, '
                        'PC(earson Correlation) critics data?, '
                        'U(ser-based CF Recommendations)?, \n'
                        'LCV(eave one out cross-validation)?, '
                        'Sim(ilarity matrix) calc for Item-based recommender?, '
                        'I(tem-based CF Recommendations)?, '
                        'LCVSIM(eave one out cross-validation)? '
                        )

        if file_io == 'R' or file_io == 'r':
            print()
            file_dir = 'data/'
            datafile = 'critics_ratings.data'
            itemfile = 'critics_movies.item'
            print ('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_dict(path, file_dir+datafile, file_dir+itemfile)
            #data_stats(prefs, whatever)
            print('Number of users: %d\nList of users:' % len(prefs),
                  list(prefs.keys()))
            # data_stats(prefs, itemfile)
            # popular_items(prefs, itemfile)

        elif file_io == 'S' or file_io == 's':
            print()
            if len(prefs) > 0:
                data_stats(prefs, itemfile)
                popular_items(prefs, itemfile)

            else:
                print ('Empty dictionary, R(ead) in some data!')

        elif file_io == 'P' or file_io == 'p':
            # print the u-i matrix
            print()
            if len(prefs) > 0:
                print ('Printing "%s" dictionary from file' % datafile)
                print ('User-item matrix contents: user, item, rating')
                for user in prefs:
                    for item in prefs[user]:
                        print(user, item, prefs[user][item])
            else:
                print ('Empty dictionary, R(ead) in some data!')

        elif file_io == 'V' or file_io == 'v':
            print()
            if len(prefs) > 0:
                # Validate the dictionary contents ..
                print ('Validating "%s" dictionary from file' % datafile)
                print ("critics['Lisa']['Lady in the Water'] =",
                       prefs['Lisa']['Lady in the Water']) # ==> 2.5
                print ("critics['Toby']:", prefs['Toby'])
                # ==> {'Snakes on a Plane': 4.5, 'You, Me and Dupree': 1.0,
                #      'Superman Returns': 4.0}
            else:
                print ('Empty dictionary, R(ead) in some data!')

            # Testing the code ..
        elif file_io == 'D' or file_io == 'd':
            print()
            if len(prefs) > 0:
                print('Examples:')
                print ('Distance sim Lisa & Gene:', sim_distance(prefs, 'Lisa', 'Gene')) # 0.29429805508554946
                num=1
                den=(1+ sqrt( (2.5-3.0)**2 + (3.5-3.5)**2 + (3.0-1.5)**2 + (3.5-5.0)**2 + (3.0-3.0)**2 + (2.5-3.5)**2))
                print('Distance sim Lisa & Gene (check):', num/den)
                print ('Distance sim Lisa & Michael:', sim_distance(prefs, 'Lisa', 'Michael')) # 0.4721359549995794
                print()

                print('User-User distance similarities:')

                ## add some code here to calc User-User distance similarities
                ## for all users or add a new function to do this

                ## TO DO: add trackers for highest and lowest similarities!
                mostSimilar = ["", "", 0.0]
                leastSimilar = ["", "", 0.0]
                for userOne in prefs:
                    for userTwo in prefs:
                        if userOne != userTwo:
                            usersSimilarity = sim_distance(prefs, userOne,
                            userTwo)
                            if usersSimilarity > mostSimilar[2] or mostSimilar[2] == 0.0:
                                mostSimilar = [userOne, userTwo, usersSimilarity]

                            if usersSimilarity < leastSimilar[2] or leastSimilar[2] == 0.0:
                                leastSimilar = [userOne, userTwo, usersSimilarity]
                            print("Distance similarities for %s and %s: %.3f" %
                            (userOne, userTwo, sim_distance(prefs, userOne,
                            userTwo)))

                            print

                print()
            else:
                print ('Empty dictionary, R(ead) in some data!')

        elif file_io == 'PC' or file_io == 'pc':
            print()
            if len(prefs) > 0:
                print ('Example:')
                print ('Pearson sim Lisa & Gene:', sim_pearson(prefs, 'Lisa', 'Gene')) # 0.39605901719066977
                print()

                print('Pearson for all users:')
                for userOne in prefs:
                    for userTwo in prefs:
                        if userOne != userTwo:
                            print("Pearson similarity for %s and %s: %.3f" %
                            (userOne, userTwo, (sim_pearson(prefs, userOne, userTwo))))
                # Calc Pearson for all users

                ## add some code here to calc User-User Pearson Correlation similarities
                ## for all users or add a new function to do this

                print()

            else:
                print ('Empty dictionary, R(ead) in some data!')

        # Testing the code ..
        elif file_io == 'U' or file_io == 'u':
            print()
            if len(prefs) > 0:
                # print ('Example:')
                # user_name = 'Toby'
                # print ('User-based CF recs for %s, sim_pearson: ' % (user_name),
                #        getRecommendations(prefs, user_name, similarity=sim_distance))
                #         # [(3.348, 'The Night Listener'),
                #         #  (2.833, 'Lady in the Water'),
                #         #  (2.531, 'Just My Luck')]
                # print ('User-based CF recs for %s, sim_distance: ' % (user_name),
                #        getRecommendations(prefs, user_name, similarity=sim_distance))
                #         # [(3.457, 'The Night Listener'),
                #         #  (2.779, 'Lady in the Water'),
                #         #  (2.422, 'Just My Luck')]
                # print()

                print('User-based CF Euclidean Distance recommendations for all users:')
                # Calc User-based CF recommendations for all users

                ## add some code here to calc User-based CF recommendations
                ## write a new function to do this ..
                get_all_UU_recs(prefs, sim=sim_distance, num_users=10, top_N=5)
                # print("***************")
                print()
                print('User-based CF Pearson recommendations for all users:')

                get_all_UU_recs(prefs, sim=sim_pearson, num_users=10, top_N=5)
                ##    '''
                ##    Print user-based CF recommendations for all users in dataset
                ##
                ##    Parameters
                ##    -- prefs: nested dictionary containing a U-I matrix
                ##    -- sim: similarity function to use (default = sim_pearson)
                ##    -- num_users: max number of users to print (default = 10)
                ##    -- top_N: max number of recommendations to print per user (default = 5)
                ##
                ##    Returns: None
                ##    '''


                print()

            else:
                print ('Empty dictionary, R(ead) in some data!')

        elif file_io == 'LCV' or file_io == 'lcv':
            print()
            if len(prefs) > 0:

                metric = input ('Enter error metric: MSE, MAE, RMSE: ')
                if metric == 'MSE' or metric == 'MAE' or metric == 'RMSE' or \
                metric == 'mse' or metric == 'mae' or metric == 'rmse':
                    metric = metric.upper()
                else:
                    metric = 'MSE'
                ## add some code here to calc LOOCV
                ## write a new function to do this ..
                error1, error_list1 = loo_cv(prefs, metric,
                sim = sim_pearson, algo = "user_based")
                error2, error_list2 = loo_cv(prefs, metric, sim = sim_distance, algo = "user_based")
                print("Pearson mean squared error: %.3f:" % error1)
                print("Distance mean squared error: %.3f:" % error2)
                # print(error_list)
            else:
                print ('Empty dictionary, R(ead) in some data!')

        elif file_io == 'Sim' or file_io == 'sim':
            print()
            if len(prefs) > 0:
                ready = False # sub command in progress
                sub_cmd = input('RD(ead) distance or RP(ead) pearson or WD(rite) distance or WP(rite) pearson? ')
                try:
                    if sub_cmd == 'RD' or sub_cmd == 'rd':
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(open( "save_itemsim_distance.p", "rb" ))
                        sim_method = 'sim_distance'

                    elif sub_cmd == 'RP' or sub_cmd == 'rp':
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(open( "save_itemsim_pearson.p", "rb" ))
                        sim_method = 'sim_pearson'

                    elif sub_cmd == 'WD' or sub_cmd == 'wd':
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(prefs,similarity=sim_distance)
                        # Dump/save dictionary to a pickle file
                        pickle.dump(itemsim, open( "save_itemsim_distance.p", "wb" ))
                        sim_method = 'sim_distance'

                    elif sub_cmd == 'WP' or sub_cmd == 'wp':
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(prefs,similarity=sim_pearson)
                        # Dump/save dictionary to a pickle file
                        pickle.dump(itemsim, open( "save_itemsim_pearson.p", "wb" ))
                        sim_method = 'sim_pearson'

                    else:
                        print("Sim sub-command %s is invalid, try again" % sub_cmd)
                        continue

                    ready = True # sub command completed successfully

                except Exception as ex:
                    print ('Error!!', ex, '\nNeed to W(rite) a file before you can R(ead) it!'
                           ' Enter S(im) again and choose a Write command')
                    print()


                if len(itemsim) > 0 and ready == True:
                    # Only want to print if sub command completed successfully
                    print ('Similarity matrix based on %s, len = %d'
                           % (sim_method, len(itemsim)))
                    print()
                    ##
                    ## enter new code here, or call a new function,
                    ##    to print the sim matrix
                    ##
                    for item in itemsim:
                        for pair in itemsim[item]:
                            print("%s, %s:  %.3f" %(item, pair[1], pair[0]))

                print()

            else:
                print ('Empty dictionary, R(ead) in some data!')

        elif file_io == 'I' or file_io == 'i':
            print()
            if len(prefs) > 0 and len(itemsim) > 0:
                # print ('Example:')
                # user_name = 'Toby'
                #
                # print ('Item-based CF recs for %s, %s: ' % (user_name, sim_method),
                #        getRecommendedItems(prefs, itemsim, user_name))

                ##
                ## Example:
                ## Item-based CF recs for Toby, sim_distance:
                ##     [(3.1667425234070894, 'The Night Listener'),
                ##      (2.9366294028444346, 'Just My Luck'),
                ##      (2.868767392626467, 'Lady in the Water')]
                ##
                ## Example:
                ## Item-based CF recs for Toby, sim_pearson:
                ##     [(3.610031066802183, 'Lady in the Water')]
                ##

                # print()

                print('Item-based CF recommendations for all users:')
                # Calc Item-based CF recommendations for all users

                ## add some code above main() to calc Item-based CF recommendations
                ## ==> write a new function to do this, as follows

                get_all_II_recs(prefs, itemsim, sim_method) # num_users=10, and top_N=5 by default  '''
                # Note that the item_sim dictionry and the sim_method string are
                #   setup in the main() Sim command

                ## Expected Results ..

                ## Item-based CF recs for all users, sim_distance:
                ## Item-based CF recommendations for all users:
                ## Item-based CF recs for Lisa, sim_distance:  []
                ## Item-based CF recs for Gene, sim_distance:  []
                ## Item-based CF recs for Michael, sim_distance:  [(3.2059731906295044, 'Just My Luck'), (3.1471787551061103, 'You, Me and Dupree')]
                ## Item-based CF recs for Claudia, sim_distance:  [(3.43454674373048, 'Lady in the Water')]
                ## Item-based CF recs for Mick, sim_distance:  []
                ## Item-based CF recs for Jack, sim_distance:  [(3.5810970647618663, 'Just My Luck')]
                ## Item-based CF recs for Toby, sim_distance:  [(3.1667425234070894, 'The Night Listener'), (2.9366294028444346, 'Just My Luck'), (2.868767392626467, 'Lady in the Water')]
                ##
                ## Item-based CF recommendations for all users:
                ## Item-based CF recs for Lisa, sim_pearson:  []
                ## Item-based CF recs for Gene, sim_pearson:  []
                ## Item-based CF recs for Michael, sim_pearson:  [(4.0, 'Just My Luck'), (3.1637361366111816, 'You, Me and Dupree')]
                ## Item-based CF recs for Claudia, sim_pearson:  [(3.4436241497684494, 'Lady in the Water')]
                ## Item-based CF recs for Mick, sim_pearson:  []
                ## Item-based CF recs for Jack, sim_pearson:  [(3.0, 'Just My Luck')]
                ## Item-based CF recs for Toby, sim_pearson:  [(3.610031066802183, 'Lady in the Water')]

                print()

            else:
                if len(prefs) == 0:
                    print ('Empty dictionary, R(ead) in some data!')
                else:
                    print ('Empty similarity matrix, use Sim(ilarity) to create a sim matrix!')


        elif file_io == 'LCVSIM' or file_io == 'lcvsim':
            print()
            if len(prefs) > 0 and itemsim !={}:
                print('LOO_CV_SIM Evaluation')
                if len(prefs) == 7:
                    prefs_name = 'critics'

                metric = input ('Enter error metric: MSE, MAE, RMSE: ')
                if metric == 'MSE' or metric == 'MAE' or metric == 'RMSE' or \
                metric == 'mse' or metric == 'mae' or metric == 'rmse':
                    metric = metric.upper()
                else:
                    metric = 'MSE'
                algo = getRecommendedItems ## Item-based recommendation

                if sim_method == 'sim_pearson':
                    sim = sim_pearson
                    error_total, error_list  = loo_cv_sim(prefs, metric, sim, algo, itemsim)
                    print('%s for %s: %.5f, len(SE list): %d, using %s'
              % (metric, prefs_name, error_total, len(error_list), sim) )
                    print()
                elif sim_method == 'sim_distance':
                    sim = sim_distance
                    error_total, error_list  = loo_cv_sim(prefs, metric, sim, algo, itemsim)
                    print('%s for %s: %.5f, len(SE list): %d, using %s'
              % (metric, prefs_name, error_total, len(error_list), sim) )
                    print()
                else:
                    print('Run S(im) command to create/load Sim matrix!')
                # if prefs_name == 'critics':
                    # print(error_list)
            else:
                print ('Empty dictionary, run R(ead) OR Empty Sim Matrix, run Sim!')


        else:
            done = True

    print('\nGoodbye!')

if __name__ == '__main__':
    main()
