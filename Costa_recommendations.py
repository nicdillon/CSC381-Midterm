'''
Recommender System
Source info: PCI, TS, 2007, 978...

Author/Collaborator: Carlos Seminario

Researcher: << Hunter Costa >>

'''

import os
import matplotlib.pyplot as plt
import numpy
from math import sqrt
import pickle
import time
import pandas

def from_file_to_prefs(path, datafile, itemfile):
    ''' Load user-item matrix from specified file 
        
        Parameters:
        -- path: directory path to datafile and itemfile
        -- datafile: delimited file containing userid, itemid, rating
        -- itemfile: delimited file that maps itemid to item name
        
        Returns:
        -- prefs: a nested prefsionary containing item ratings for each user
    
    '''
    
    # Get movie titles, place into movies prefsionary indexed by itemID
    movies={}
    try:
        with open (path + '/' + itemfile, encoding='iso8859') as myfile: 
            # this encoding is required for some datasets: encoding='iso8859'
            for line in myfile:
                (id,title )=line.split('|')[0:2]
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
    
    # Load data into a nested prefsionary
    prefs={}
    for line in open(path+'/'+ datafile):
        #print(line, line.split('\t')) #debug
        (user,movieid,rating, ts)=line.split('\t')
        user = user.strip() # remove spaces
        movieid = movieid.strip() # remove spaces
        prefs.setdefault(user,{}) # make it a nested dicitonary
        prefs[user][movies[movieid]]=float(rating)
    
    #return a prefsionary of preferences
    return prefs

''' User-based recommendation calculations '''

def data_stats(prefs, data_file):

    #Initialization of prefss and lists of the ratings
    ratingList = list()
    movieList = list()
    userAvgList = list()
    userRateList = list()
    movieRatingprefs = {}
    numpyMovieList = list()
    movieRateAvgList = list()

    #calculate the average and std dev for the dataset ratings
    for user in prefs:
        for item in prefs[user]:
            ratingList.append(prefs[user][item])
            movieList.append(item)
            userRateList.append(prefs[user][item]) #add to user ratings list
            movieRatingprefs.setdefault(item, []) #sets up the movie rating prefs

            # traverse the nested disctionary and pull out the movie titles as keys and assign their ratings (values) into
            # a newly generated list
            if item in movieRatingprefs.keys(): #check to see if a movie already exists in the prefs
                movieRatingprefs[item].append(prefs[user][item]) #add rating to the value list
            else:
                movieRatingprefs[item]=[] #create a new key with the movie

        userAvgList.append(numpy.average(userRateList)) #append average user rating and std dev
        userRateList.clear()
        numpyMovieList = list(movieRatingprefs.values()) #new list with the values lists

    userAvgRate = numpy.average(userAvgList) #get avg user rating
    userstDev = numpy.std(userAvgList) #get user std dev
    rateAvg = numpy.average(ratingList) #gets avg rating score
    stDev = numpy.std(ratingList) #gets ratings std dev
    
    #Calculates average and std dev for the movie items in the data
    for index in numpyMovieList:
        movieRateAvgList.append(numpy.average(index))
    movieRatingAvg = numpy.average(movieRateAvgList)
    movieRatingStdDev = numpy.std(movieRateAvgList)

    #Matrix rating sparsity (1 - (num_of_ratings/ (num_users * num_items)))
    matSparsity = (1-(len(ratingList) / (len(prefs) * len(set(movieList))))) * 100

    #gets the total number of users in the data
    print('Number of users: %d \nNumber of items: %d \nNumber of ratings: %d' 
           % (len(prefs), len(set(movieList)), len(ratingList)))
    
    #prints the overall average rating and std dev
    print('Overall average rating: %.2f out of 5 and std dev of %.2f' % (rateAvg, stDev))

    print('Average item rating: %.2f out of 5 and std dev: %.2f' %(movieRatingAvg, movieRatingStdDev)) 

    #prints the average user rating
    print('Average user rating: %.2f out of 5 and std dev of %.2f' % (userAvgRate, userstDev))
    
    #prints the matrix sparsity percentage
    print('Matrix Sparsity: %.2f%%' % matSparsity)

    #Generate histogram from matplotlib
    plt.hist(ratingList, bins = [1.0,2.0,3.0,4.0,5.0])
    plt.title('Ratings Histogram')
    plt.ylabel('Number of users ratings')
    plt.xlabel('Rating')
    plt.show()
    
def popular_items(prefs, ratings_filename):
    popItemprefs = {}
    rateAvg = list()
    mostRated = list()
    highestRated = list()

    for user in prefs:
        for item in prefs[user]:
            popItemprefs.setdefault(item, [])
            if item in popItemprefs.keys():
                popItemprefs[item].append(prefs[user][item])
            else:
                popItemprefs[item]=[]

    counter = 0
    #Popular items - most rated (e.g. top 5)
    for movie in popItemprefs:
        rateAvg = list(popItemprefs.values())
        while counter < len(rateAvg):
            mostRated.append((len(rateAvg[counter]),movie, 
                            numpy.around(numpy.average(rateAvg[counter]), decimals=2)))
            highestRated.append((numpy.around(numpy.average(rateAvg[counter]), decimals=2), 
                                movie, len(rateAvg[counter])))
            counter+=1
            break
        
    mostRated = sorted(mostRated, reverse=True)
    highestRated = sorted(highestRated, reverse=True)

    print()
    print('Popular Items -- most rated:\n# of Ratings: |  Movie Tile: | Avg Rating:')
    for printer in mostRated[:5]:
        print(printer)
    
    #Popular items - highest rated
    print('\nPopular Items -- highest Rated:\nAvg Ratings: |  Movie Tile: | # of Ratings:')
    for printer in highestRated[:5]:
        print(printer)

    #Overall best rated items (number ratings >= x, where x = 5 for critics)
    print('\nPopular Items -- most rated:\nAvg Rating: | Movie Tile: | # of Ratings: ')
    for printer in highestRated[:25]:
        print(printer[2])
        # if(printer[2] < 15):
        #     break
        # else:
        #     print(printer)

# Returns a distance-based similarity score for person1 and person2
def sim_distance(prefs,person1,person2):
    '''
        Calculate Euclidean distance similarity 

        Parameters:
        -- prefs: prefsionary containing user-item matrix
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
     
    sum_of_squares = 0
    for item in prefs[person1]:
        if item in prefs[person2]:
            #print(item, prefs[person1][item], prefs[person2][item])
            sq = pow(prefs[person1][item]-prefs[person2][item],2)
            #print (sq)
            sum_of_squares += sq
     
    return 1/(1+sqrt(sum_of_squares))
    
def all_sim_distances(prefs):
    userList = []
    for user1 in prefs:
        for user2 in prefs:
            if user1 == user2:
                continue
            elif (user2, user1) in userList:
                continue
            else:
                userList.append((user1, user2))
                print('%s, %s:' %(user1, user2), sim_distance(prefs, user1, user2))
    

# Returns the Pearson correlation coefficient for p1 and p2
def sim_pearson(prefs,p1,p2):
    '''
        Calculate Pearson Correlation similarity 

        Parameters:
        -- prefs: prefsionary containing user-item matrix
        -- person1: string containing name of user 1
        -- person2: string containing name of user 2
        
        Returns:
        -- Pearson Correlation similarity as a float
        
    '''
     # Get the list of shared_items
    si={}
    for item in prefs[p1]: 
        if item in prefs[p2]: 
            si[item]=1
    
    # if they have no ratings in common, return 0
    if len(si)==0: 
        return 0
    
    # Get the average of each user's film ratings
    p1AvgList = list()
    p2AvgList = list()

    for rate in prefs[p1]:
        if rate in prefs[p2]:  
            p1AvgList.append(prefs[p1][rate])
            p2AvgList.append(prefs[p2][rate])
    p1Avg = float(numpy.average(p1AvgList))
    p2Avg = float(numpy.average(p2AvgList))

    # Calculate the numerator for the 2 users
    numerSum = 0
    ds1 = list()
    ds2 = list()

    for item in prefs[p1]:
        if item in prefs[p2]:
            
            ns = (prefs[p1][item]-p1Avg)*(prefs[p2][item]-p2Avg) # ind numerator calculation
            ds1.append(pow(prefs[p1][item]-p1Avg, 2)) # denom 1 calculation
            ds2.append(pow(prefs[p2][item]-p2Avg,2)) # denom 2 calculation
            numerSum += ns

    # Calculate the denomenator
    denomSum = sqrt(sum(ds1)) * sqrt(sum(ds2))

    # Calculate the Pearson Coefficient
    if denomSum == 0:
        return 0
    else:
        pCoef = numerSum/denomSum
    
    return pCoef

def all_sim_pearson(prefs):
    userList = []
    for user1 in prefs:
        for user2 in prefs:
            if user1 == user2:
                continue
            elif (user2, user1) in userList:
                continue
            else:
                userList.append((user1, user2))
                print('%s, %s:' %(user1, user2), sim_pearson(prefs, user1, user2))
    
# Gets recommendations for a person by using a weighted average
# of every other user's rankings
def getRecommendations(prefs,person,similarity):
    '''
        Calculates recommendations for a given user 

        Parameters:
        -- prefs: prefsionary containing user-item matrix
        -- person: string containing name of user
        -- similarity: function to calc similarity (sim_pearson is default)
        
        Returns:
        -- A list of recommended items with 0 or more tuples, 
           each tuple contains (preprefsed rating, item name).
           List is sorted, high to low, by preprefsed rating.
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
    # rankings=[(float('%f'%(total/simSums[item])),item) for item,total in totals.items()]
  
    # Return the sorted list
    rankings.sort()
    rankings.reverse()
    return rankings

def get_all_UU_recs(prefs, sim, num_users=10, top_N=5):
    ''' 
    Print user-based CF recommendations for all users in dataset
    
    Parameters
    -- prefs: nested prefsionary containing a U-I matrix
    -- sim: similarity function to use (default = sim_pearson)
    -- num_users: max number of users to print (default = 10)
    -- top_N: max number of recommendations to print per user (default = 5)
    
    Returns: None
    '''
    for user in prefs:
        print('User-based CF recs for %s, %s: ' % (user,sim.__name__),
            getRecommendations(prefs, user, sim))
    
# Compute Leave_One_Out evaluation
def loo_cv(prefs, metric, sim, algo):
    """
    Leave_One_Out Evaluation: evaluates recommender system ACCURACY
     
     Parameters:
        --prefs dataset: critics, ml-100K, etc.
	    --metric: MSE, MAE, RMSE, etc.
	    --sim: distance, pearson, etc.
	    --algo: user-based recommender, item-based recommender, etc.
	 
    Returns:
        error_total: MSE, MAE, RMSE totals for this set of conditions
	    error_list: list of actual-preprefsed differences
    
    Create a temp copy of prefs
    For each user in prefs:
       for item in each user's profile:
          delete this item
          get recommendation (aka preprefsion) list
	  select the recommendation for this item from the list returned
          calc error, save into error list
	  restore this item
    return mean error, error list
    """
    # creates a copy of the dict
    prefsCopy = dict()
    for user in prefs:
        for item in prefs[user]:
            prefsCopy.setdefault(user,{})
            prefsCopy[user][item] = prefs[user][item]
    
    errorList = []
    delItem = ()
    for user in prefs:
        for item in prefs[user]:
            rating = []
            delItem = (item, prefsCopy[user][item]) # Saves a copy of the item being deleted from the dict
            prefsCopy[user].pop(item) # deletes item
            rating.append(algo(prefsCopy,user,sim))# gets movie rec for user
            
            i=0
            for movie in rating:
                for title in movie:
                    if title[1] == delItem[0]:
                        theoretical = movie[i][0]
                        actual = prefs[user][item]
                        sq_error = pow(theoretical-actual,2)
                        errorList.append(sq_error)
                        print('User: %s, Item: %s, Prediction: %.5f, Actual: %.5f, Error: %.5f' 
	                            %(user, item, theoretical, actual, sq_error))
                    else:
                        i+=1
                        continue
            prefsCopy[user][item] = delItem[1] # adds item back into dict
    error_total = 0
    if metric == 'MSE' or metric == 'mse':
        error_total = numpy.around(sum(errorList)/len(errorList), decimals=4)

    return error_total

''' Item-based recommendation calculations '''

def topMatches(prefs,person,similarity, n=5):
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

    rankings=[(score/totalSim[item],item) for item,score in scores.items()]    
  
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
        print('User-based CF recs for %s, %s: ' % (user,sim_method),
            getRecommendedItems(prefs, itemsim, user))

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
    prefsCopy = dict()
    for user in prefs:
        for item in prefs[user]:
            prefsCopy.setdefault(user,{})
            prefsCopy[user][item] = prefs[user][item]
    
    errorList = []
    delItem = ()
    rating = ()
    for user in prefs:
        for item in prefs[user]:
            rating = []
            # Saves a copy of the item being deleted from the dict
            delItem = (item, prefsCopy[user][item])
             # deletes item
            prefsCopy[user].pop(item)
            
            rating.append(algo(prefsCopy,sim_matrix,user))# gets movie rec for user
            
            i=0
            for movie in rating:
                for title in movie:
                    if title[1] == delItem[0]:
                        theoretical = movie[i][0]
                        actual = prefs[user][item]
                        sq_error = pow(theoretical-actual,2)
                        errorList.append(sq_error)
                        prefsCopy[user][item] = delItem[1] # adds item back into dict
                        # print('User: %s, Item: %s, Prediction: %.5f, Actual: %.5f, Error: %.5f' 
	                            # %(user, item, theoretical, actual, sq_error))
                    else:
                        i+=1
                        continue
    error_total = 0
    if metric == 'MSE' or metric == 'mse':
        error_total = numpy.around(sum(errorList)/len(errorList), decimals=4)
    elif metric == 'RMSE' or metric == 'rmse':
        error_total = numpy.around(sqrt(sum(errorList)/len(errorList)), decimals=4)
    # elif metric == 'MAE' or metric == 'mae':
    #     error_total = numpy.around(sum(errorList)/len(errorList), decimals=4)
    return error_total, errorList
def main():
    ''' User interface for Python console '''
    
    # Load critics prefs from file
    path = os.getcwd()# this gets the current working directory
                       # you can customize path for your own computer here
    print('\npath: %s' % path) # debug
    done = False
    prefs = {}
    itemsim = {} ## make sure this in your main()
    
    while not done: 
        print()
        # Start a simple dialog
        file_io = input('R(ead) critics data from file?, '
                        'RML(ead) ML data from file?, \n'
                        'P(rint) the U-I matrix?, '
                        'V(alidate) the prefsionary?, '
                        'S(tats)?, \n'
                        'Pop(ular) items in the data, '
                        'D(istance) critics data?, '
                        'PC(earson Correlation) critics data?, '
                        'U(ser-based CF Recommendations)?, \n'
                        'LCV(eave one out cross-validation)?, '
                        'Sim(ilarity matrix) calc for Item-based recommender?, '
                        'I(tem-based CF Recommendations)?, '
                        'LCVSIM(eave one out cross-validation)?, ')
        
        if file_io == 'R' or file_io == 'r':
            print()
            file_dir = 'starter/data/'
            datafile = 'critics_ratings.data'
            itemfile = 'critics_movies.item'
            print ('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_prefs(path, file_dir+datafile, file_dir+itemfile)
            print('Number of users: %d\nList of users:' % len(prefs), 
                  list(prefs.keys()))

        elif file_io == 'RML' or file_io == 'rml':
            print()
            file_dir = 'ml-100k/'
            datafile = 'u.data'
            itemfile = 'u.item'
            print ('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_prefs(path, file_dir+datafile, file_dir+itemfile)
            print('Number of users: %d\nList of users:' % len(prefs), 
                  list(prefs.keys())[0:10])
            
        elif file_io == 'P' or file_io == 'p':
            # print the u-i matrix
            print()
            if len(prefs) > 0:
                print ('Printing "%s" prefsionary from file' % datafile)
                print ('User-item matrix contents: user, item, rating')
                for user in prefs:
                    for item in prefs[user]:
                        print(user, item, prefs[user][item])
            else:
                print ('Empty prefsionary, R(ead) in some data!')
                
        elif file_io == 'V' or file_io == 'v':      
            print()
            if len(prefs) > 0:
                # Validate the prefsionary contents ..
                print ('Validating "%s" prefsionary from file' % datafile)
                print ("critics['Lisa']['Lady in the Water'] =", 
                       prefs['Lisa']['Lady in the Water']) # ==> 2.5
                print ("critics['Toby']:", prefs['Toby']) 
                # ==> {'Snakes on a Plane': 4.5, 'You, Me and Dupree': 1.0, 
                #      'Superman Returns': 4.0}
            else:
                print ('Empty prefsionary, R(ead) in some data!')

        elif file_io == 'S' or file_io == 's': 
            data_stats(prefs, datafile)  
 
        elif file_io == 'Pop' or file_io == 'pop': 
            popular_items(prefs, datafile) 

        elif file_io == 'D' or file_io == 'd':
            print()
            if len(prefs) > 0:            
                print('Examples:')
                print ('Distance sim Lisa & Gene:', sim_distance(prefs, 'Lisa', 'Gene')) # 0.29429805508554946
                num=1
                den=(1+ sqrt( (2.5-3.0)**2 + (3.5-3.5)**2 + (3.0-1.5)**2 + (3.5-5.0)**2 + (3.0-3.0)**2 + (2.5-3.5)**2))
                print('Distance sim Lisa & Gene (check):', num/den)    
                print ('Distance sim Lisa & Michael', sim_distance(prefs, 'Lisa', 'Michael')) # 0.4721359549995794
                print()
                
                print('User-User distance similarities:')
                
                ## add some code here to calc User-User distance similarities
                all_sim_distances(prefs) 
                
                print()
            else:
                print ('Empty prefsionary, R(ead) in some data!')

        elif file_io == 'PC' or file_io == 'pc':
            print()
            if len(prefs) > 0:             
                print ('Example:')
                print ('Pearson sim Lisa & Mike:', sim_pearson(prefs, 'Lisa', 'Michael')) # 0.39605901719066977
                print()
                
                print('Pearson for all users:')

                ## for all users 
                all_sim_pearson(prefs)
                print()
                
            else:
                print ('Empty prefsionary, R(ead) in some data!') 
            
        elif file_io == 'U' or file_io == 'u':
            print()
            if len(prefs) > 0:             
                print ('Example:')
                user_name = 'Toby'
                print ('User-based CF recs for %s, sim_pearson: ' % (user_name), 
                    getRecommendations(prefs, user_name,similarity=sim_pearson)) 
                    # [(3.348, 'The Night Listener'), 
                    #  (2.833, 'Lady in the Water'), 
                    #  (2.531, 'Just My Luck')]
                print ('\nUser-based CF recs for %s, sim_distance: ' % (user_name),
                    getRecommendations(prefs, user_name, similarity=sim_distance)) 
                        # [(3.457, 'The Night Listener'), 
                        #  (2.779, 'Lady in the Water'), 
                        #  (2.422, 'Just My Luck')]
                print()
                
                print('User-based CF recommendations for all users:')
                # Calc User-based CF recommendations for all users
                print(get_all_UU_recs(prefs, sim=sim_pearson, num_users=10, top_N=5))
                print(get_all_UU_recs(prefs, sim=sim_distance, num_users=10, top_N=5))
            
            else:
                print ('Empty prefsionary, R(ead) in some data!')  
   
        elif file_io == 'LCV' or file_io == 'lcv':
            print()
            if len(prefs) > 0:
                metric = 'MSE'       
                algo = getRecommendations
                sim = input ('Enter similarity: sim_distance or sim_pearson: ')
                if sim == 'sim_distance':
                    print ('Users, Errors:')   
                    print('\nEuclidean-Distance: ', 
                    loo_cv(prefs,'MSE',sim_distance, algo))
                    print()
                elif sim == 'sim_pearson':
                    print ('Users, Errors:') 
                    print('\nPearson-Corr: ', 
                    loo_cv(prefs,metric,sim_pearson, algo))
                    print()
               
            else:
                print ('Empty prefsionary, R(ead) in some data!')        
        
        elif file_io == 'Sim' or file_io == 'sim':
            print()
            if len(prefs) > 0: 
                ready = False # sub command in progress
                sub_cmd = input('RD(ead) distance or RP(ead) pearson or WD(rite) distance or WP(rite) pearson? ')
                try:
                    if sub_cmd == 'RD' or sub_cmd == 'rd':
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(open("starter/save_itemsim_distance.p", "rb" ))
                        # print(itemsim)
                        sim_method = 'sim_distance'
    
                    elif sub_cmd == 'RP' or sub_cmd == 'rp':
                        # Load the dictionary back from the pickle file.
                        itemsim = pickle.load(open("starter/save_itemsim_pearson.p", "rb" )) 
                        print(itemsim) 
                        sim_method = 'sim_pearson'
                        
                    elif sub_cmd == 'WD' or sub_cmd == 'wd':
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(prefs,similarity=sim_distance)                     
                        # Dump/save dictionary to a pickle file
                        pickle.dump(itemsim, open("starter/save_itemsim_distance.p", "wb" ))
                        sim_method = 'sim_distance'
                        
                    elif sub_cmd == 'WP' or sub_cmd == 'wp':
                        # transpose the U-I matrix and calc item-item similarities matrix
                        itemsim = calculateSimilarItems(prefs,similarity=sim_pearson)                     
                        # Dump/save dictionary to a pickle file
                        pickle.dump(itemsim, open( "starter/save_itemsim_pearson.p", "wb" )) 
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
                print()

        elif file_io == 'I' or file_io == 'i':
                print()

                if len(prefs) > 0 and len(itemsim) > 0:                
                    print ('Example:')
                    user_name = 'Toby'
    
                    print ('Item-based CF recs for %s, %s: ' % (user_name, sim_method), 
                    getRecommendedItems(prefs, itemsim, user_name)) 
                
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
    
                    print()
                
                    print('Item-based CF recommendations for all users:')
                    # Calc Item-based CF recommendations for all users
        
                    ## add some code above main() to calc Item-based CF recommendations 
                    ## ==> write a new function to do this, as follows
                    
                    get_all_II_recs(prefs, itemsim, sim_method) # num_users=10, and top_N=5 by default  '''
                    #Note that the item_sim dictionry and the sim_method string are
                    #setup in the main() Sim command
                
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
                if prefs_name == 'critics':
                    print(error_list)
            else:
                print ('Empty dictionary, run R(ead) OR Empty Sim Matrix, run Sim!') 
                    
        else:
            done = True
    print('\nGoodbye!')
        
if __name__ == '__main__':
    main()
