import pandas as pd 
import numpy as np


dataset1=pd.read_csv('tmdb_5000_credits.csv')
dataset2=pd.read_csv('tmdb_5000_movies.csv')


#Dataset 1 contains the movie cast , crew and movie id
#Dataset 2 contains all other features of movies such as overview, voting, average rate, count.

dataset1.columns=['id','tittle','cast','crew']   #This method assigns names to the columns of data.

dataset2=dataset2.merge(dataset1,on='id')   #Merging the 2 datasets based on their ID's

#We need an average scoring metric which will be calculated using Weighted Rating=((v/v+m)*R)+((m/v+m)*C) where v=votes m=minima for selection 
#C=average vote R=Average Rating

C=dataset2['vote_average'].mean() #Mean is 6.09217

m=dataset2['vote_count'].quantile(0.9)

selected_movies=dataset2.copy().loc[dataset2['vote_count']>=m]

def getWeightage(x):
    v=x['vote_count']
    R=x['vote_average']
    return (v/(v+m)*R)+(m/(v+m)*C)

selected_movies['score']=selected_movies.apply(getWeightage,axis=1)

selected_movies=selected_movies.sort_values('score',ascending=False)

print(selected_movies[['title','score']].head(20))
