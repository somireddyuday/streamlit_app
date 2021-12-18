#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from math import sqrt
import seaborn as sns
from matplotlib import pyplot as plt
import string
import random
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_datasets as tfds
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from ast import literal_eval
import streamlit as st


def welcome():
    return "Welcome All"

# In[11]:
add_selectbox = st.sidebar.radio(
    "Select the type of SEARCH METHOD",
    ("Item_to_Item_recommended_movies", "User_to_User_recommended_movies","Movie_to_Movie_recommended_movies","New_User_Movie_recommended_movies")
)

merged_data = pd.read_csv(r'C:\Users\somir\Desktop\streamlit_app\Datasets\merged_data.csv')


# In[14]:


movies_genres_tfidf = TfidfVectorizer(token_pattern = '[a-zA-Z0-9\-]+')

#Replace NaN with an empty string
merged_data['genres'] = merged_data['genres'].fillna('').replace(to_replace="(no genres listed)", value="")

#Construct the required TF-IDF matrix by fitting and transforming the data
movies_genres_tfidf_matrix = movies_genres_tfidf.fit_transform(merged_data['genres'])
# print(movies_genres_tfidf.get_feature_names())
# Compute the cosine similarity matrix
# print(movies_genres_tfidf_matrix.shape)
# print(movies_genres_tfidf_matrix.dtype)
movies_cosine = linear_kernel(movies_genres_tfidf_matrix, movies_genres_tfidf_matrix)


# In[15]:


def get_recommendations_based_on_movies(movie_title, movies_cosine=movies_cosine):
    """
    Calculates top 2 movies to recommend based on given movie titles genres. 
    :param movie_title: title of movie to be taken for base of recommendation
    :param movies_cosine: cosine similarity between movies 
    :return: Titles of movies recommended to user
    """
    # Get the index of the movie that matches the title
    movie_idx = merged_data.loc[merged_data['title'].isin([movie_title])]
    movie_idx = movie_idx.index
    
    # Get the pairwsie similarity scores of all movies with that movie
    movies_sim_scores = list(enumerate(movies_cosine[movie_idx][0]))
    
    # Sort the movies based on the similarity scores
    movies_sim_scores = sorted(movies_sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies
    movies_sim_scores = movies_sim_scores[1:11]
    
    # Get the movie indices
    movie_indices = [i[0] for i in movies_sim_scores]
    
    # Return the top 2 most similar movies
    return merged_data['title'].iloc[movie_indices]



# In[17]:


def get_recommendation_based_on_watch_history(userId):
    """
    Calculates top movies to be recommended to user based on movie user has watched.  
    :param userId: userid of user
    :return: Titles of movies recommended to user
    """
    movie_recommended_list = []
    movie_list = []
    rating_filtered = merged_data[merged_data["userId"]== userId]
    for key, row in rating_filtered.iterrows():
#         movie_list.append((merged_data["title"][row["movieId"]==merged_data["movieId"]]).values)
        movie_list.append(merged_data[row["movieId"]==merged_data["movieId"]]['title'].to_list())
    for index, movie in enumerate(movie_list):
        for key, movie_recommended in get_recommendations_based_on_movies(movie[0]).iteritems():
            movie_recommended_list.append(movie_recommended)

    # removing already watched movie from recommended list    
    for movie_title in movie_recommended_list:
        if movie_title in movie_list:
            movie_recommended_list.remove(movie_title)
    
    return set(movie_recommended_list)



# In[18]:


from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine, correlation


# In[20]:


df_movies  = pd.read_csv(r'C:\Users\somir\Desktop\streamlit_app\Datasets\movies.csv')
df_ratings = pd.read_csv(r'C:\Users\somir\Desktop\streamlit_app\Datasets\ratings.csv')


# In[21]:


df_movies_ratings=pd.merge(df_movies, df_ratings)


# In[22]:


ratings_matrix_items = df_movies_ratings.pivot_table(index=['movieId'],columns=['userId'],values='rating').reset_index(drop=True)
ratings_matrix_items.fillna( 0, inplace = True )
movie_similarity = 1 - pairwise_distances( ratings_matrix_items.values, metric="cosine" )
np.fill_diagonal( movie_similarity, 0 ) #Filling diagonals with 0s for future use when sorting is done
ratings_matrix_items = pd.DataFrame( movie_similarity )


# In[23]:


def item_to_item_similarity(movieName): 
    """
    recomendates similar movies
   :param data: name of the movie 
   """
    try:
        #user_inp=input('Enter the reference movie title based on which recommendations are to be made: ')
        user_inp=movieName
        inp=df_movies[df_movies['title']==user_inp].index.tolist()
        inp=inp[0]

        df_movies['similarity'] = ratings_matrix_items.iloc[inp]
        df_movies.columns = ['movie_id', 'title', 'release_date','similarity']
    except:
        print("Sorry, the movie is not in the database!")


# In[24]:


def get_movie_recommend_based_on_item_similairty(user_id):
    """
     Recommending movie which user hasn't watched as per Item Similarity
    :param user_id: user_id to whom movie needs to be recommended
    :return: movieIds to user 
    """
    movie_input= df_movies_ratings[(df_movies_ratings.userId==user_id) & df_movies_ratings.rating.isin([5,4.5])][['title']]
    movie_input=movie_input.iloc[0,0]
    item_to_item_similarity(movie_input)
    movies_sorted_as_per_user_choice=df_movies.sort_values( ["similarity"], ascending = False )
    movies_sorted_as_per_user_choice=movies_sorted_as_per_user_choice[movies_sorted_as_per_user_choice['similarity'] >=0.45]['movie_id']
    recommended_movies=list()
    recommended_item_dataframe=pd.DataFrame()
    user_to_movies= df_ratings[df_ratings['userId']== user_id]['movieId']
    for movieId in movies_sorted_as_per_user_choice:
            if movieId not in user_to_movies:
                df_new= df_ratings[(df_ratings.movieId==movieId)]
                recommended_item_dataframe=pd.concat([recommended_item_dataframe,df_new])
            top_10_movies=recommended_item_dataframe.sort_values(["rating"], ascending = False )[1:10] 
    return top_10_movies['movieId']


# In[25]:


def get_movieId_To_Title(listMovieIDs):
    """
     Converting movieId to titles
    :param user_id: List of movies
    :return: movie titles
    """
    movie_titles= list()
    for id in listMovieIDs:
        movie_titles.append(df_movies[df_movies['movie_id']==id]['title'])
    return movie_titles


# In[26]:


ratings_matrix_users = df_movies_ratings.pivot_table(index=['userId'],columns=['movieId'],values='rating').reset_index(drop=True)
ratings_matrix_users.fillna( 0, inplace = True )
movie_similarity = 1 - pairwise_distances( ratings_matrix_users.values, metric="cosine" )
np.fill_diagonal( movie_similarity, 0 ) #Filling diagonals with 0s for future use when sorting is done
ratings_matrix_users = pd.DataFrame( movie_similarity )


# In[28]:


similar_user= ratings_matrix_users.idxmax(axis=1)
df_similar_user= similar_user.to_frame()
df_similar_user.columns=['similarUser']


# In[29]:


movieId_recommended=list()
def get_recommended_movies_based_on_user_similarity(userId):
    """
     Recommending movies which user hasn't watched as per User Similarity
    :param user_id: user_id to whom movie needs to be recommended
    :return: movieIds to user 
    """
    user_to_movies= df_ratings[df_ratings['userId']== userId]['movieId']
    sim_user=df_similar_user.iloc[0,0]
    df_recommended=pd.DataFrame(columns=['movieId','title','genres','userId','rating','timestamp'])
    for movieId in df_ratings[df_ratings['userId']== sim_user]['movieId']:
        if movieId not in user_to_movies:
            df_new= df_movies_ratings[(df_movies_ratings.userId==sim_user) & (df_movies_ratings.movieId==movieId)]
            df_recommended=pd.concat([df_recommended,df_new])
        top_10_movies=df_recommended.sort_values(['rating'], ascending = False )[1:10]  
    return top_10_movies['movieId']


# In[30]:


def get_user_similar_movies( user1, user2 ):
    
    """
     Returning common movies and ratings of same for both the users
    :param user1,user2: user ids of 2 users need to compare
    :return: movieIds to user 
    """
    movies_common = df_movies_ratings[df_movies_ratings.userId == user1].merge(
      df_movies_ratings[df_movies_ratings.userId == user2],
      on = "movieId",
      how = "inner" )
    movies_common.drop(['movieId','genres_x','genres_y', 'timestamp_x','timestamp_y','title_y'],axis=1,inplace=True)
    return movies_common


# In[31]:


from surprise import Reader, Dataset, SVD
from surprise.model_selection import KFold
from surprise.model_selection import cross_validate


# Load Reader library
reader = Reader()
svd=SVD()


# In[32]:


import pickle
file = open('svd_model.txt', 'rb')
svd = pickle.load(file)                     
file.close()


# In[33]:


def evaluation_collaborative_svd_model(userId,userOrItem):
    """
    hydrid the functionality of Collaborative based and svd based model to see if ratings of predicted movies 
    :param userId: userId of user, userOrItem is a boolean value if True it is User-User and if false Item-Item
    :return: dataframe of movies and ratings
    """ 
    movie_Ids_List= list()
    movie_Rating_List=list()
    movie_Id_Rating= pd.DataFrame(columns=['movieId','rating'])
    if userOrItem== True:
        movie_Ids_List=get_recommended_movies_based_on_user_similarity(userId)
    else:
        movie_Ids_List=get_movie_recommend_based_on_item_similairty(user_id)
    for movieId in movie_Ids_List:
        predict = svd.predict(userId, movieId)
        movie_Rating_List.append([movieId,predict.est])
        movie_Id_Rating = pd.DataFrame(np.array(movie_Rating_List), columns=['movieId','rating'])
        count=movie_Id_Rating[(movie_Id_Rating['rating'])>=3]['movieId'].count()
        total=movie_Id_Rating.shape[0]
        hit_ratio= count/total
    return hit_ratio


# In[35]:



def hybrid_content_svd_model(userId):
    """
    hydrid the functionality of content based and svd based model to recommend user top 10 movies. 
    :param userId: userId of user
    :return: list of movies recommended with rating given by svd model
    """
    recommended_movies_by_content_model = get_recommendation_based_on_watch_history(userId)
    recommended_movies_by_content_model = df_movies[df_movies.apply(lambda movie: movie["title"] in recommended_movies_by_content_model, axis=1)]
    for key, columns in recommended_movies_by_content_model.iterrows():
        predict = svd.predict(userId, columns["movieId"])
        recommended_movies_by_content_model.loc[key, "svd_rating"] = predict.est
#         if(predict.est < 2):
#             recommended_movies_by_content_model = recommended_movies_by_content_model.drop([key])
    return recommended_movies_by_content_model.sort_values("svd_rating", ascending=False).iloc[0:11]
        


# In[36]:
def main():
    def Item_to_Item():
        st.title("Movie Recommendation Engine")
        html_temp = """
                        <div style="background-color:tomato;padding:10px">
                        <h2 style="color:white;text-align:center;">Movie Recommendation Engine App </h2>
                        </div>
                        """
        st.markdown(html_temp, unsafe_allow_html=True)
        test_user = st.text_input("Enter the User for whom you wanna see top 10 recommendations：", "UserID")
        result = ""
        if st.button("Predict"):
            result = hybrid_content_svd_model(int(test_user))['title']
            result = result.unique()
        st.text('Top 10 movie recommendations for user id' + ' ' + str(test_user) + ' ' + 'are:')

        for i in range(len(result)):
            st.text('{0}: {1}'.format(i + 1, result[i]))


    def User_to_User():
        st.title("Movie Recommendation Engine")
        html_temp = """
                <div style="background-color:tomato;padding:10px">
                <h2 style="color:white;text-align:center;">Movie Recommendation Engine App </h2>
                </div>
                """
        st.markdown(html_temp, unsafe_allow_html=True)
        test_user_1 = st.text_input("Enter the User1 for whom you wanna see top 10 recommendations：", "USER_ID_1")
        test_user_2 = st.text_input("Enter the User2 for whom you wanna see top 10 recommendations：", "USER_ID_2")

        result = ""
        if st.button("Predict"):
            result = get_user_similar_movies(int(test_user_1), int(test_user_2))['title_x'].head(10).to_list()
        st.text('Top 10 movie recommendations for user to user ' + ' ' + str(test_user_1) + ' ' + 'are:')
        for i in range(len(result)):
            st.text('{0}: {1}'.format(i + 1, result[i]))

    def genre_based_recommendation():
        st.title("Movie Recommendation Engine")
        html_temp = """
                <div style="background-color:tomato;padding:10px">
                <h2 style="color:white;text-align:center;">Movie Recommendation Engine App </h2>
                </div>
                """
        st.markdown(html_temp, unsafe_allow_html=True)
        test_user_1 = st.text_input("Enter the Movie for you wanna see top 10 recommendations：", "USER_ID_1")
        result = ""
        if st.button("Predict"):
            result = get_recommendations_based_on_movies(str(test_user_1))
        st.text('Top 10 movie recommendations based on movie ' + ' ' + str(test_user_1) + ' ' + 'are:')
        for i in range(len(result)):
            st.text('{0}: {1}'.format(i + 1, result[i]))

    def New_User_movie_recommendation():
        st.title("Movie Recommendation Engine")
        html_temp = """
                <div style="background-color:tomato;padding:10px">
                <h2 style="color:white;text-align:center;">Movie Recommendation Engine App </h2>
                </div>
                """
        st.markdown(html_temp, unsafe_allow_html=True)
        test_user_1 = st.text_input("Enter the New User for whom you wanna see top 10 recommendations：", "USER_ID_1")
        result = ""
        if st.button("Predict"):
            result = pd.DataFrame(merged_data.groupby('title')['rating'].mean()).sort_values(by='rating', ascending=False).head(10)
            result = result.index
        for i in range(len(result)):
            st.text('{0}: {1}'.format(i + 1, result[i]))


    if add_selectbox == 'Item_to_Item_recommended_movies' :
        Item_to_Item()
    elif add_selectbox == 'User_to_User_recommended_movies':
        User_to_User()
    elif add_selectbox == 'Movie_to_Movie_recommended_movies':
        genre_based_recommendation()
    elif add_selectbox == 'New_User_Movie_recommended_movies':
        New_User_movie_recommendation()
if __name__ == '__main__':
    main()










