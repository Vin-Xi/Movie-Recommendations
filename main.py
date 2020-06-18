import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from ast import literal_eval 
from sklearn.feature_extraction.text import CountVectorizer

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

def getWeightage(x): #Function for getting weighted rating
    v=x['vote_count']
    R=x['vote_average']
    return (v/(v+m)*R)+(m/(v+m)*C)

selected_movies['score']=selected_movies.apply(getWeightage,axis=1)# Creates a new column using function.

selected_movies=selected_movies.sort_values('score',ascending=False)



pop=dataset2.sort_values('popularity',ascending=False)
plt.figure(figsize=(12,4))
plt.barh(pop['title'].head(10),pop['popularity'].head(10),align='center',color='red')
plt.gca().invert_yaxis()
plt.xlabel('Popularity')
plt.title('Popular Movies')


tfidf=TfidfVectorizer(stop_words='english') #This vectorizer will remove stop words such as 'a' and 'the' etc.
dataset2['overview']=dataset2['overview'].fillna('') #To replace Nan with empty spaces

tfidf_matrix=tfidf.fit_transform(dataset2['overview'])#Finally this will transform our overview into a forward Index

cos_sim=linear_kernel(tfidf_matrix,tfidf_matrix)#Calculating the Similarity by getting dot product

indices=pd.Series(dataset2.index,index=dataset2['title']).drop_duplicates() #Creating Inverted Index.

def find_recommendations(title,cos_sim=cos_sim):
    idx=indices[title] #Get Indices of movie with this title

    sim_scores=list(enumerate(cos_sim[idx]))#Get the list of all movies at this row(index)

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)#Sort this list by ascending


    sim_scores=sim_scores[1:11]#Top 11 movies are the similar one's.

    movie_indices=[i[0] for i in sim_scores]
    return dataset2['title'].iloc[movie_indices]


features=['cast','crew','keywords','genres']

for f in features:
    dataset2[f]=dataset2[f].apply(literal_eval)

def find_director(dataset):
    for i in dataset:
        if i['job']=='Director':
            return i['name']
    return np.nan

def get_list(dataset):
    if isinstance(dataset,list):
        names=[ i['name'] for i in dataset]
        if len(names)>3:
            names=names[:3]
        return names

    return []

dataset2['Director']=dataset2['crew'].apply(find_director)

features=['cast','keywords','genres']

for f in features:
    dataset2[f]=dataset2[f].apply(get_list)

def clean_data(dataset):
    if isinstance(dataset,list):
        return [str.lower(i.replace(" ","")) for i in dataset]
    else:
        if isinstance(dataset,str):
            return str.lower(dataset.replace(" ",""))
        else:
            return ''

features=['cast','genres','keywords','Director']

for f in features:
    dataset2[f]=dataset2[f].apply(clean_data)

def mix_data(dataset):
    return ' '.join(dataset['keywords'])+' '+' '.join(dataset['cast'])+' '+' '.join(dataset['Director'])+' '+' '.join(dataset['genres'])
dataset2['mixed']=dataset2.apply(mix_data,axis=1)

vectorizer=CountVectorizer(stop_words="english")
count_matrix=vectorizer.fit_transform(dataset2['mixed'])

cosine_sim=cosine_similarity(count_matrix,count_matrix)

dataset2=dataset2.reset_index()
indices=pd.Series(dataset2.index,index=dataset2['title'])


print(find_recommendations('Avatar',cosine_sim))