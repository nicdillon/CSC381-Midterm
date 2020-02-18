'''
Recommender System
Source info: PCI, TS, 2007, 978...

Author/Collaborator: Carlos Seminario

Researcher: Charlie Caswell

'''

import statistics
import os
import numpy as np
from matplotlib import pyplot as plt 
from math import sqrt 
import pickle

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
        with open (path + '/' + itemfile, encoding='iso8859') as myfile: 
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
    
    #print("STARTING TO READ FILE")
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
    '''
    Calculates and prints important statistics 
    
    Parameters:
        --dictionaryName: the dictionary with ratings
        --ratingsFilename: the file to read from
    Returns:
        --None 
    '''
    numUsers = len(dictionaryName)
    numRatings = 0
    movies = []
    totalRatingsList = []
    itemRatings = {}
    numItemRatings = {}
    avgItemRatings = {}
    stdDevItem = {}
    userRatings = {}

    #iterate through dictionary 
    for user in dictionaryName:
        for movie in dictionaryName[user]:
            rating = dictionaryName[user][movie]
            numRatings += 1
            totalRatingsList.append(rating)
            if movie not in movies:
                movies.append(movie)

            if movie not in itemRatings:
                itemRatings[movie] = rating
                numItemRatings[movie] = 1
                stdDevItem[movie] = [rating]
            else:
                itemRatings[movie] += rating
                numItemRatings[movie] += 1
                stdDevItem[movie].append(rating)

            if user not in userRatings:
                userRatings[user] = [rating]
            else:
                userRatings[user].append(rating)
    
    for movie in itemRatings:
        avgItemRatings[movie] = itemRatings[movie]/numItemRatings[movie]

    totalRatingsAvg = statistics.mean(totalRatingsList)
    numItems = len(movies)
    stdDev = statistics.stdev(totalRatingsList)
    
    print("Number of users:     ", numUsers)
    print("Number of ratings:   ", numRatings)
    print("Number of items:     ", numItems)
    print()

    print("Overall average rating:      %.3f" % (totalRatingsAvg))
    print("Overall Standard Deviation:  %.3f" % stdDev)
    total = numItems*numUsers 
    print("Matrix Ratings Sparcity:     %.3f" % (100*((total - numRatings)/total)))

    print()
    for movie in stdDevItem:
        itemList = stdDevItem[movie]
        if len(itemList) > 2:
            itemStdDev = statistics.stdev(itemList)
        print("Avg Item Rating and Std Dev for %s:  %.3f, %.3f" % (movie, avgItemRatings[movie], itemStdDev))

    print()
    for user in userRatings:
        userList = userRatings[user]
        userStdDev = statistics.stdev(userList)
        useravgRating = statistics.mean(userList)
        print("Avg User Rating and Std Dev for %s:  %.3f, %.3f" %(user, useravgRating, userStdDev))
    print()

    array = np.array(totalRatingsList)
    bins = [0, 1, 2, 3, 4, 5]
    hist = np.histogram(array, bins)
    plt.hist(array, bins)
    plt.title("Histogram of Movie Ratings")
    plt.show()

def popular_items(dictionaryName, ratingsFilename):
    '''
    Calculates and prints information about the popular items

    Parameters:
        --dictionaryName: the dictionary with ratings
        --ratingsFilename: the file to read from
    Returns:
        --None 
    '''
    movieRatings = {}

    for user in dictionaryName:
        for movie in dictionaryName[user]:
            rating = dictionaryName[user][movie]

            if movie not in movieRatings:
                movieRatings[movie] = [rating]
            else:
                movieRatings[movie].append(rating)

    mostRated = {}
    highestRated = {}
    overallRated = {}
    #find all the lists of items for all three categories
    for movie in movieRatings:
        movieList = movieRatings[movie]
        mostRated[movie] = len(movieRatings[movie])
        highestRated[movie] = statistics.mean(movieList)
        if len(movieList) >= 15:
            overallRated[movie] = statistics.mean(movieList)

    N = 10
    mostRated5 = calcTopN(mostRated, N)
    highestRated5 = calcTopN(highestRated, N)
    overallBest5 = calcTopN(overallRated, N)

    print("The %d most rated movies were" %N)
    for movie in mostRated5:
        numRating = len(movieRatings[movie])
        print("     %s:  %d" %(movie, numRating))
    print()

    print("The %d highest rated movies were" %N)
    for movie in highestRated5:
        rating  = statistics.mean(movieRatings[movie])
        print("     %s:  %.3f" % (movie, rating))
    print()

    print("The %d overall best rated movies were" %N)
    for movie in overallBest5:
        rating  = statistics.mean(movieRatings[movie])
        numRating = len(movieRatings[movie])
        print("     %s:  %.3f, %d" % (movie, rating, numRating))
    print()


def calcTopN(data, top):
    """
    A helper function that calculates the top 5 items from a list
    
    Parameters:
        --data: the list of values to be compared 
        --top: the number of top items to find
    Returns:
        --topN: the top N best items
    """
    topN = []
    for i in range(top):
        high = 0
        name = "none"
        for movie in data:
            if data[movie] > high:
                high = data[movie]
                name = movie

        topN.append(name)
        del data[name]
    return topN
        

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
    #find shared list of movies
    movies = []
    for item in prefs[p1]: 
        if item in prefs[p2]:
            movies.append(item) 

    if len(movies) == 0:
        return 0
    
    #calculate the user averages 
    ratings1 = []
    ratings2 = []
    for movie in movies:
        ratings1.append(prefs[p1][movie])
        ratings2.append(prefs[p2][movie])
        
    p1Avg = statistics.mean(ratings1)
    p2Avg = statistics.mean(ratings2)

    num = 0
    p1Den = 0
    p2Den = 0
    #calc numerator denominator for p1 and p2
    for movie in movies:
        num += (prefs[p1][movie] - p1Avg)*(prefs[p2][movie] - p2Avg)
        p1Den += pow((prefs[p1][movie] - p1Avg),2)
        p2Den += pow((prefs[p2][movie] - p2Avg),2)

    p1Den = (p1Den)**.5
    p2Den = (p2Den)**.5
    
    if ((p1Den*p2Den) != 0):
        return num/(p1Den*p2Den)
    
    else:
        return 0

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

def get_all_UU_recs(prefs, sim=sim_distance, num_users=10, top_N=5):
    ''' 
    Print user-based CF recommendations for all users in dataset
                
    Parameters
    -- prefs: nested dictionary containing a U-I matrix
    -- sim: similarity function to use (default = sim_pearson)
    -- num_users: max number of users to print (default = 10)
    -- top_N: max number of recommendations to print per user (default = 5)
    '''
    for user in prefs:
        if num_users == 0:
            break
        else:
            num_users -= 1
        recom = getRecommendations(prefs, user, sim)
        print("The predicted recomendations for %s:" %(user))
        top_N = 5
        for rec in recom:
            if top_N == 0:
                break 
            else:
                top_N -= 1
            print("     *%s:  %.3f" %(rec[1], rec[0]))
    print()
def loo_cv(prefs, metric = "MSE", sim = sim_distance, algo = getRecommendations):
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
    """
    #make hard copy
    prefsCopy = {}
    for user in prefs:
        for item in prefs[user]:
            prefsCopy.setdefault(user, {})
            prefsCopy[user][item] = prefs[user][item]
                
    error_list = []
    error = 0.
    for user in prefs:
        for movie in prefs[user]:
            rating = prefs[user][movie]
            del prefsCopy[user][movie]
            #calc recommendation
            for item in algo(prefsCopy, user, sim):
                if (movie in item):
                    if metric == "MSE":
                        error_list.append((rating - item[0])**2)
                    elif metric == "RMSE":
                        error_list.append((rating - item[0])**2)
                    elif metric == "MAE":
                        error_list.append(abs(rating - item[0]))
            prefsCopy[user][movie] = rating
            
    error = statistics.mean(error_list)
    if metric == "RMSE":
        error = error**.5
    
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
        if num_users == 0:
            break
        else:
            num_users -= 1
        recom = getRecommendedItems(prefs, itemsim, user)
        print("The predicted recomendations for %s using %s:" %(user, sim_method))
        top_N = 5
        for rec in recom:
            if top_N == 0:
                break 
            else:
                top_N -= 1
            print("     *%s:  %.3f" %(rec[1], rec[0]))
    


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
    #make hard copy
    prefsCopy = {}
    for user in prefs:
        for item in prefs[user]:
            prefsCopy.setdefault(user, {})
            prefsCopy[user][item] = prefs[user][item]
                
    error_list = []
    error = 0.
    for user in prefs:
        for movie in prefs[user]:
            rating = prefs[user][movie]
            del prefsCopy[user][movie]
            
            for item in algo(prefsCopy, sim_matrix, user):
                if (movie in item):
                    if metric == "MSE":
                        error_list.append((rating - item[0])**2)
                    elif metric == "RMSE":
                        error_list.append((rating - item[0])**2)
                    elif metric == "MAE":
                        error_list.append(abs(rating - item[0]))
            prefsCopy[user][movie] = rating
            
    error = statistics.mean(error_list)
    if metric == "RMSE":
        error = error**.5
    
    return error, error_list

def main():
    ''' User interface for Python console '''
    
    # Load critics dict from file
    path = os.getcwd() # this gets the current working directory
                       # you can customize path for your own computer here
    print('\npath: %s' % path) # debug
    done = False ## make sure this in your main()
    prefs = {}   ## make sure this in your main()
    itemsim = {} ## make sure this in your main()
    
    while not done: 
        print()
        # Start a simple dialog
        file_io = input('R(ead) critics data from file?, \n'
                        'RML(ead ml-100k dataset) \n'
                        'P(rint) the U-I matrix?, \n'
                        'V(alidate) the dictionary? \n'
                        'S(tats)? \n'
                        'D(istance) critics data? \n'
                        'PC(earson Correlation) critics data? \n'
                        'U(ser-based CF Recommendations)? \n'
                        'LCV(eave one out cross-validation)? \n'
                        'Sim(ilarity matrix) calc for Item-based recommender? \n'
                        'I(tem-based CF Recommendations)?, \n'
                        'LCVSIM(eave one out cross-validation)?, ')
        
        if file_io == 'R' or file_io == 'r':
            #read the data
            print()
            file_dir = 'data/'
            datafile = 'critics_ratings.data'
            itemfile = 'critics_movies.item'
            print ('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_dict(path, file_dir+datafile, file_dir+itemfile)

        elif file_io == 'RML' or file_io == 'rml':
            print()
            file_dir = 'data/ml-100k/' # path from current directory
            datafile = 'u.data' # ratngs file
            itemfile = 'u.item' # movie titles file
            print ('Reading "%s" dictionary from file' % datafile)
            prefs = from_file_to_dict(path, file_dir+datafile, file_dir+itemfile) 
            print('Number of users: %d\nList of users [0:10]:'
                % len(prefs), list(prefs.keys())[0:10] )
            
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

        elif file_io == 'S' or file_io == 's':
            if len(prefs) > 0:
                data_stats(prefs, datafile)
                popular_items(prefs, datafile)
                
            else:
                print('Empty dictionary, R(ead) in some data')
        
        
        elif file_io == 'D' or file_io == 'd':
            print()
            if len(prefs) > 0:            
                 
                print('User-User distance similarities:')
               
                for userOne in prefs:
                    for userTwo in prefs:
                        if userOne != userTwo:
                            simDis = sim_distance(prefs, userOne, userTwo)
                            #shift = 20 - (len(userOne) + len(userTwo))
                            print("Distance sim for %s and %s: %.3f" % (userOne, userTwo, simDis))
                            #print("Distance sim for", userOne, "and", userTwo, shift * " ", simDis)
                print()
                
            else:
                print ('Empty dictionary, R(ead) in some data!')  
        
        elif file_io == 'PC' or file_io == 'pc':
            print()
            if len(prefs) > 0:             
                
                print('Pearson for all users:')
                
                for userOne in prefs:
                    for userTwo in prefs:
                        if userOne != userTwo:
                            pearsonVal = sim_pearson(prefs, userOne, userTwo)
                            print("Pearson value for %s and %s: %.3f" %(userOne, userTwo, pearsonVal))
                print()
                
            else:
                print ('Empty dictionary, R(ead) in some data!') 
                
        elif file_io == 'U' or file_io == 'u':
            print()
            if len(prefs) > 0:             

                print("Predicted user recommendations sim distance:")
                get_all_UU_recs(prefs, sim=sim_distance)
                print("Predicted user recommendations sim pearson:")
                get_all_UU_recs(prefs, sim=sim_pearson)
                
                print()
                
            else:
                print ('Empty dictionary, R(ead) in some data!')            
            
        elif file_io == 'LCV' or file_io == 'lcv':
            print()
            if len(prefs) > 0:             
            
                error1, error_list1 = loo_cv(prefs, metric = "MSE", sim = sim_distance, algo = getRecommendations)
                error2, error_list2 = loo_cv(prefs, metric = "MSE", sim = sim_pearson, algo = getRecommendations)
                print("Error for sim_distance %.8f" %error1)
                print()
                print("Error for sim_pearson %.8f" %error2)
                #print(len(error_list1))
                #print(len(error_list2))
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
                   
                    for item in itemsim:
                        for pair in itemsim[item]:
                            print("%s, %s:  %.5f" %(item, pair[1], pair[0]))
                print()
                
            else:
                print ('Empty dictionary, R(ead) in some data!')      
            
        elif file_io == 'I' or file_io == 'i':
            print()
            if len(prefs) > 0 and len(itemsim) > 0:                
                print('Item-based CF recommendations for all users:')
                    
                get_all_II_recs(prefs, itemsim, sim_method) 
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
			  % (metric, prefs_name, error_total, len(error_list), sim_method) )
                    print()
                elif sim_method == 'sim_distance':
                    sim = sim_distance
                    error_total, error_list  = loo_cv_sim(prefs, metric, sim, algo, itemsim)
                    print('%s for %s: %.5f, len(SE list): %d, using %s' 
			  % (metric, prefs_name, error_total, len(error_list), sim_method) )
                    print()
                else:
                    print('Run S(im) command to create/load Sim matrix!')
                if prefs_name == 'critics':
                    pass
                    #print(error_list)
            else:
                print ('Empty dictionary, run R(ead) OR Empty Sim Matrix, run Sim!')
        
        else:
            done = True
    
    print('\nGoodbye!')
    
if __name__ == '__main__':
    main()