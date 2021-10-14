# Library used to connect to sites on the internet

import requests

# Genius.com API, for finding lyrics to songs

from lyricsgenius import Genius



# ***************************** LYRICS SEARCH ***********************************


# Initialize Genius.com API with client token

genius = Genius("qNd5dzQ8FsDxg8hmOuD0MCLcfGFURUlm_B6411Sm-biulx-qQhDS_9cykvI5OsYf")

# Configuration for Genius.com API. Removing section headers such as [verse] and [Chorus]
# Also removes automaticall printed data when searces performed

genius.remove_section_headers = True
genius.verbose = False

# Search request and input of artist name and song title

artist_name = genius.search_artist(input("Enter Artist Name: "),max_songs=2)
song_title = artist_name.song(input("Enter Song Title: "))

#print(song_title.lyrics)


 
# ***************************** MACHINE LEARNING ***********************************


# This function will pass text to the machine learning model
# and return the top result with the highest confidence

def classify(text):

    # Initializing machine learning model from machinelearningforkids.co.uk; 
    # Uses link to IBM Watson for model training

    key = "16e12ff0-2c82-11ec-87dc-adee8731f028085ce5fb-f724-4926-af1d-653bccca7471"
    url = "https://machinelearningforkids.co.uk/api/scratch/"+ key + "/classify"

    response = requests.get(url, params={ "data" : text })

    if response.ok:
        responseData = response.json()
        topMatch = responseData[0]
        return topMatch
    else:
        response.raise_for_status()


# CLassifying song lyrics using machine learning model. 
# Limit lyrics to 2048 characters, whch is model input limit

demo = classify(song_title.lyrics[0:2043])

label = demo["class_name"]
confidence = demo["confidence"]


# Resultsof classification

print ("result: '%s' with %d%% confidence" % (label, confidence))