#!/usr/bin/env python
# coding: utf-8

# only insert where it says insert

# STEP 1: Get Spotify Client Information

# In[1]:


from sklearn.neighbors import NearestNeighbors
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import seaborn as sns
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
client_id = 'insert client id'
client_secret = 'insert client secret'
client_credentials_manager = SpotifyClientCredentials('insert client id', 'insert client secret')
sp = spotipy.Spotify(client_credentials_manager = client_credentials_manager)


# STEP 2: download playlists and convert into csv

# In[2]:


id_test = sp.user_playlist_tracks('insert spotify username', 'insert spotify uri')['items'][0]['track']['id']
columns = ['artist', 'track']
list(map(lambda x: columns.append(x), list(sp.audio_features(id_test)[0].keys())))

playlist_tracks = pd.DataFrame(columns = columns, index = range(0, 200))
playlist_tracks.head()


# In[3]:


playlist_ids = ['insert spotify uri', 'insert spotify uri']
row_counter = 0

for playlist_id in playlist_ids:
    for track in sp.user_playlist_tracks('insert spotify username', 'insert spotify uri')['items']:
        current_id = track['track']['id']
        current_row = [track['track']['artists'][0]['name'], track['track']['name']]
        (list(map(lambda x: current_row.append(x), list(sp.audio_features(current_id)[0].values()))))
        playlist_tracks.iloc[row_counter] = current_row
        row_counter += 1
playlist_tracks


# In[4]:


playlist_tracks.to_csv('file.csv', encoding='utf-8', index = False)


# In[5]:


playlist_tracks.to_csv(r'C:insert file location you want it to be stored in', index = False, header = True)


# In[ ]:




