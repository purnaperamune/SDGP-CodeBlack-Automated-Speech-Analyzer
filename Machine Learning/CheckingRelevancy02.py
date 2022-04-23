#!/usr/bin/env python
# coding: utf-8

# # **1. INSTALLING AND IMPORTING REQUIRED MODULES**

# ## *1. Installing required modules*

# In[1]:


get_ipython().system('apt install libasound2-dev portaudio19-dev libportaudio2 libportaudiocpp0 ffmpeg')


# In[2]:


get_ipython().system('pip install PyAudio')


# In[3]:


get_ipython().system('pip install pydub')


# In[4]:


get_ipython().system('pip install speechrecognition')


# In[5]:


get_ipython().system('pip install pyLDAvis')


# In[6]:


get_ipython().system('pip install datasets')


# In[7]:


get_ipython().system('pip install plotly')


# ## *2. Importing required modules*

# In[8]:


import soundfile as sf
import pyloudnorm as pyln
import pyaudio
import wave
from pydub import AudioSegment
import speech_recognition as sr
from io import BytesIO
from base64 import b64decode
from IPython.display import Javascript
import IPython.display as ipd
import spacy
import numpy as np
import nltk
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
pyLDAvis.enable_notebook()
from datasets import load_dataset
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import plotly.express as px
#Libraries for preprocessing
from gensim.parsing.preprocessing import remove_stopwords
import string
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
#Download once if using NLTK for preprocessing
import nltk
nltk.download('punkt')
#Libraries for vectorisation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.decomposition import LatentDirichletAllocation
#Libraries for clustering
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')


# # **3. DATASET FOR TOPIC MODELING**

# In[9]:


#loading dataset
dataset = load_dataset('medical_questions_pairs')
dataset


# In[10]:


#creating dataframe from dataset
df = pd.DataFrame(dataset)
df.info()


# In[11]:


#overview of first 5 records
df.head(5)


# In[12]:


#creating csv from dataframe
df.to_csv("medical_data.csv")


# In[13]:


#creating text file from dataframe
with open("medical_dataset.txt", "w") as f:
  for i in df['train']:
    line = list(str(i).split("'"))
    f.write(str(line[5]) + "\n" + str(line[9]) + "\n")


# # **4. TOPIC MODELLING with LDA**

# ## *1. Text cleaning*

# We use the following function to clean our texts and return a list of tokens:

# In[14]:



#spacy.load('en')
spacy.load("en_core_web_sm")
from spacy.lang.en import English
parser = English()
def tokenize(text):
    lda_tokens = []
    tokens = parser(text)
    for token in tokens:
        if token.orth_.isspace():
            continue
        elif token.like_url:
            lda_tokens.append('URL')
        elif token.orth_.startswith('@'):
            lda_tokens.append('SCREEN_NAME')
        else:
            lda_tokens.append(token.lower_)
    return lda_tokens


# We use NLTK’s Wordnet to find the meanings of words, synonyms, antonyms, and more. In addition, we use WordNetLemmatizer to get the root word.

# In[15]:


nltk.download('wordnet')
from nltk.corpus import wordnet as wn
def get_lemma(word):
    lemma = wn.morphy(word)
    if lemma is None:
        return word
    else:
        return lemma
    
from nltk.stem.wordnet import WordNetLemmatizer
def get_lemma2(word):
    return WordNetLemmatizer().lemmatize(word)


# Filter out stop words:

# In[16]:


nltk.download('stopwords')
en_stop = set(nltk.corpus.stopwords.words('english'))


# Now we can define a function to prepare the text for topic modelling:

# In[17]:


def prepare_text_for_lda(text):
    tokens = tokenize(text)
    tokens = [token for token in tokens if len(token) > 4]
    tokens = [token for token in tokens if token not in en_stop]
    tokens = [get_lemma(token) for token in tokens]
    return tokens


# Open up our data, read line by line, for each line, prepare text for LDA, then add to a list.
# 
# Now we can see how our text data are converted:

# In[18]:


import random
text_data = []
with open('medical_dataset.txt') as f:
    for line in f:
        tokens = prepare_text_for_lda(line)
        if random.random() > .99:
            print(tokens)
            text_data.append(tokens)


# ## *2. LDA with Gensim*

# First, we are creating a dictionary from the data, then convert to bag-of-words corpus and save the dictionary and corpus for future use.

# In[19]:


from gensim import corpora
dictionary = corpora.Dictionary(text_data)
corpus = [dictionary.doc2bow(text) for text in text_data]
import pickle
pickle.dump(corpus, open('corpus.pkl', 'wb'))
dictionary.save('dictionary.gensim')


# We are asking LDA to find 1 topic in the data:

# In[20]:


import gensim
NUM_TOPICS = 1
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('model1.gensim')
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)


# Here is Topic 0 with 4 contributing words to it.
# 
# With LDA, we can see that different document with different topics, and the discriminations are obvious.

# We are asking LDA to find 5 topics in the data:

# In[21]:


import gensim
NUM_TOPICS = 5
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=15)
ldamodel.save('model5.gensim')
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)


# Let’s try a new document:

# In[22]:


new_doc = 'How to fight with severe headache'
new_doc = prepare_text_for_lda(new_doc)
new_doc_bow = dictionary.doc2bow(new_doc)
print(new_doc_bow)
print(ldamodel.get_document_topics(new_doc_bow))


# Now we are asking LDA to find 3 topics in the data:

# In[23]:


ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 3, id2word=dictionary, passes=15)
ldamodel.save('model3.gensim')
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)


# Now find 10 topics:

# In[24]:


ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics = 10, id2word=dictionary, passes=15)
ldamodel.save('model10.gensim')
topics = ldamodel.print_topics(num_words=4)
for topic in topics:
    print(topic)


# ## *3. pyLDAvis*

# pyLDAvis is designed to help users interpret the topics in a topic model that has been fit to a corpus of text data. The package extracts information from a fitted LDA topic model to inform an interactive web-based visualization.
# 
# 

# Visualizing 5 topics:

# In[25]:


dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
corpus = pickle.load(open('corpus.pkl', 'rb'))
lda5 = gensim.models.ldamodel.LdaModel.load('model5.gensim')
#import pyLDAvis.gensim
lda_display = gensimvis.prepare(lda5, corpus, dictionary, sort_topics=False)
pyLDAvis.display(lda_display)


# Saliency: a measure of how much the term tells you about the topic.
# 
# Relevance: a weighted average of the probability of the word given the
# 
# topic and the word given the topic normalized by the probability of the topic.
# The size of the bubble measures the importance of the topics, relative to the data.
# First, we got the most salient terms, means terms mostly tell us about what’s going on relative to the topics. We can also look at individual topic.
# 
# Visualizing 3 topics:

# In[26]:


lda3 = gensim.models.ldamodel.LdaModel.load('model3.gensim')
lda_display3 = gensimvis.prepare(lda3, corpus, dictionary, sort_topics=False)
pyLDAvis.display(lda_display3)


# Visualizing 10 topics:

# In[27]:


lda10 = gensim.models.ldamodel.LdaModel.load('model10.gensim')
lda_display10 = gensimvis.prepare(lda10, corpus, dictionary, sort_topics=False)
pyLDAvis.display(lda_display10)


# # **5. TEST ON SAMPLE SPEECH**

# ## *1. Show sample speech stats*

# In[28]:


def show_pydub_stats(filename):
  """Returns different audio attributes related to an audio file."""
  # Create AudioSegment instance
  audio_segment = AudioSegment.from_file(filename)
  
  # Print audio attributes and return AudioSegment instance
  print(f"Channels: {audio_segment.channels}")
  print(f"Sample width: {audio_segment.sample_width}")
  print(f"Frame rate (sample rate): {audio_segment.frame_rate}")
  print(f"Frame width: {audio_segment.frame_width}")
  print(f"Length (ms): {len(audio_segment)}")
  return audio_segment

# Test the function
print("\nFIRST SPEECH STATS :")
speech_1_stats = show_pydub_stats("FilmPurna.wav")
print("\nSECOND SPEECH STATS :")
speech_2_stats = show_pydub_stats("HealthcareChethila.wav")
print("\nTHIRD SPEECH STATS :")
speech_3_stats = show_pydub_stats("HospitalPurna.wav")
print("\nFOURTH SPEECH STATS :")
speech_4_stats = show_pydub_stats("CancerPurna.wav")


# ## *2. Show sample speech text*

# In[29]:


def transcribe_audio(filename):
  """Takes a .wav format audio file and transcribes it to text."""
  # Setup a recognizer instance
  recognizer = sr.Recognizer()
  
  # Import the audio file and convert to audio data
  audio_file = sr.AudioFile(filename)
  with audio_file as source:
    audio_data = recognizer.record(source)
  
  # Return the transcribed text
  return recognizer.recognize_google(audio_data)

# Test the function
try:
  speech_1_text = transcribe_audio("FilmPurna.wav")
  print("\nText of first speech : \n" + speech_1_text)
except:
  print("First speech is unidentifiable ")

try:
  speech_2_text = transcribe_audio("HealthcareChethila.wav")
  print("\nText of second speech : \n" + speech_2_text)
except:
  print("Second speech is unidentifiable ")

try:
  speech_3_text = transcribe_audio("HospitalPurna.wav")
  print("\nText of third speech : \n" + speech_3_text)
except:
  print("Third speech is unidentifiable")

try:
  speech_4_text = transcribe_audio("CancerPurna.wav")
  print("\nText of fourth speech : \n" + speech_4_text)
except:
  print("Fourth speech is unidentifiable ")  


# ## *3. Named entity recognition on sample speech text*

# In[30]:


# Create a spaCy language model instance
nlp = spacy.load("en_core_web_sm")

# Create a spaCy doc 
doc_1 = nlp(speech_1_text)
doc_2 = nlp(speech_2_text)
doc_3 = nlp(speech_3_text)
doc_4 = nlp(speech_4_text)

# Check the type of doc
print(type(doc_1))
print(type(doc_2))
print(type(doc_3))
print(type(doc_4))


# In[31]:


# Show tokens from first speech
for token in doc_1:
    print("\nTokens from first speech : ")
    print(token.text, token.idx)

for token in doc_2:
    print("\nTokens from second speech : ")
    print(token.text, token.idx)

for token in doc_3:
    print("\nTokens from third speech : ")
    print(token.text, token.idx)       

for token in doc_4:
    print("\nTokens from fourth speech : ")
    print(token.text, token.idx)


# In[32]:


# Show sentences in doc
for sentence in doc_1.sents:
    print("\nSentences from first speech : ")
    print(sentence)

for sentence in doc_2.sents:
    print("\nSentences from second speech : ")
    print(sentence)

for sentence in doc_3.sents:
    print("\nSentences from third speech : ")
    print(sentence)

for sentence in doc_4.sents:
    print("\nSentences from fourth speech : ")
    print(sentence)


# In[33]:


# Show named entities and their labels
for entity in doc_1.ents:
    print("\nEntities from first speech : ")
    print(entity.text, entity.label_)

for entity in doc_2.ents:
    print("\nEntities from second speech : ")
    print(entity.text, entity.label_)

for entity in doc_3.ents:
    print("\nEntities from third speech : ")
    print(entity.text, entity.label_)

for entity in doc_4.ents:
    print("\nEntities from fourth speech : ")
    print(entity.text, entity.label_)


# # **6. EXTRACT TOPICS FROM SAMPLE SPEECH**

# In[34]:


speeches = [speech_1_text, speech_2_text, speech_3_text, speech_4_text]
n=1
for x in speeches:
  print("\nTOPICS FROM SPEECH # " + str(n))
  new_doc = x
  new_doc = prepare_text_for_lda(new_doc)
  new_doc_bow = dictionary.doc2bow(new_doc)
  print(new_doc_bow)
  print(ldamodel.get_document_topics(new_doc_bow))
  n=n+1


# # **10. CLUSTERING**

# In[35]:


import matplotlib.pyplot as plt


# In[36]:


#Load cleaned data set
df = pd.read_csv('medical_data.csv', encoding= 'unicode_escape')
df.head(5)


# In[37]:


#converting text to lowercase
df['train'] = df['train'].str.lower()
#converting to series
series_text = df['train']
#converting to list
list_text = list(series_text)
#converting to string
string_text = ' '.join([str(elem) for elem in list_text])


# In[38]:


#creating function to preprocess/clean text data with join
def clean_text(text):
     tokens = prepare_text_for_lda(text)
     clean_text =  ' '.join(tokens)
     return clean_text


# In[39]:


# Python code to convert string to list
  
def Convert(string):
    li = list(string.split(" "))
    return li  
# Driver code  
converted_string = Convert(clean_text(string_text))
print(converted_string)


# In[40]:


#Bag of words
vectorizer_cv = CountVectorizer(analyzer='word')
X_cv = vectorizer_cv.fit_transform(converted_string)


# In[41]:


#show matrix
matrix = pd.concat([series_text, pd.DataFrame(X_cv.toarray(),columns=vectorizer_cv.get_feature_names())], axis=1)
matrix.head(5)


# In[42]:


#TF-IDF (word level)
vectorizer_wtf = TfidfVectorizer(analyzer='word')
X_wtf = vectorizer_wtf.fit_transform(list_text)


# In[43]:


matrix[['train']]


# In[44]:


#TF-IDF (n-gram level)
vectorizer_ntf = TfidfVectorizer(analyzer='word',ngram_range=(1,2))
X_ntf = vectorizer_ntf.fit_transform(list_text)


# In[45]:


#LDA
lda = LatentDirichletAllocation(n_components=30, learning_decay=0.9)
X_lda = lda.fit(X_cv)

#Plot topics function. Code from: https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
def plot_top_words(model, feature_names, n_top_words, title):
    fig, axes = plt.subplots(6, 5, figsize=(30, 30), sharex=True)
    axes = axes.flatten()
    for topic_idx, topic in enumerate(model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_names[i] for i in top_features_ind]
        weights = topic[top_features_ind]

        ax = axes[topic_idx]
        ax.barh(top_features, weights, height=0.7)
        ax.set_title(f'Topic {topic_idx +1}',
                     fontdict={'fontsize': 30})
        ax.invert_yaxis()
        ax.tick_params(axis='both', which='major', labelsize=20)
        for i in 'top right left'.split():
            ax.spines[i].set_visible(False)
        fig.suptitle(title, fontsize=40)
    plt.subplots_adjust(top=0.90, bottom=0.05, wspace=0.90, hspace=0.3)
    plt.show()
    
#Show topics
n_top_words = 5
feature_names = vectorizer_cv.get_feature_names()
print(feature_names)
plot_top_words(X_lda, feature_names, n_top_words, '')


# In[46]:


#creating model
kmeans = KMeans(n_clusters=100)
kmeans.fit(X_cv)
result = pd.concat([series_text,pd.DataFrame(X_cv.toarray(),columns=vectorizer_cv.get_feature_names())],axis=1)
result['cluster'] = kmeans.predict(X_cv)


# In[47]:


result[['train', 'cluster']]


# In[48]:


#Label each cluster with the word(s) that all of its food names have in common
clusters = result['cluster'].unique()
labels = []
for i in range(len(clusters)):
    subset = result[result['cluster'] == clusters[i]]
    words = ' '.join([x for x in np.where(subset.all()!=0,subset.columns,None) if x and x!='Name' and x!='cluster' and len(x.split()) == 1])
    labels.append(words)
labels_table = pd.DataFrame(zip(clusters,labels),columns=['cluster','label'])
result_labelled = pd.merge(result,labels_table,on='cluster',how='left')


# In[49]:


pd.pivot_table(result_labelled, index=['cluster'], values= ['train'], aggfunc='count').sort_values(['train'], ascending=False).rename(columns={'train':'count'})


# In[50]:


#Visualise sizes of supermarket categories (manually added to result_labelled) and clean clusters
result_summary = pd.pivot_table(result_labelled,index=['cluster'],values=['train'],aggfunc='count').reset_index().rename(columns={'train':'count'})
result_treemap = result_summary[(result_summary['cluster'] != '') & (result_summary['count'] > 1)]
fig = px.treemap(result_treemap,path=['cluster'],values='count')
fig.show();


# # **11. CHECK SIMILARITY**

# ## *On Sample Speech*

# **Marking scheme**

# 70 - Relevancy and Comparing
# 
# 20 - Voice
# 
# 10 - User input score
# 
# ___
# 
# 100 - Total Rating

# # **Relevancy and Comparing**

# In[51]:


# Relevancy and Comparing Score

speeches = [speech_1_text, speech_2_text, speech_3_text, speech_4_text]
n = 1
relevancy_temp_score = []
score_by_relevancy = []
for x in speeches:
  score = nlp(clean_text(string_text)).similarity(nlp(clean_text(x)))
  #print('\nSimiliarity score for speech #  ' + str(n) + ' : '+ str(score) )
  relevancy_temp_score.append(score)
  n = n+1

score_by_relevancy = np.array(relevancy_temp_score)
score_by_relevancy = 70*score_by_relevancy/100


# # **Voice**

# In[52]:


# Voice Score

# defining loudness function
def check_loudness(f):
  data, rate = sf.read(f) # load audio (with shape (samples, channels))
  #print(data.shape)
  meter = pyln.Meter(rate) # create BS.1770 meter
  loudness = meter.integrated_loudness(data) # measure loudness
  return loudness


# In[53]:


# calling loudness function
loudness_1 = check_loudness("FilmPurna.wav")
loudness_2 = check_loudness("HealthcareChethila.wav")
loudness_3 = check_loudness("HospitalPurna.wav")
loudness_4 = check_loudness("CancerPurna.wav")

# making list of loudness for all speeches
loudness_list = [loudness_1, loudness_2, loudness_3, loudness_4]


# In case of google API for speech into text:
# 
# Loudness should sound consistent for spoken words at -16 LUFS (stereo) or -19 LUFS (mono).

# In[54]:


# defining loudness score function
def get_loudness_score(x):
  score = (abs(abs(abs(x) - 19) - 19)/19)
  return score


# In[55]:


voice_temp_score = []
n = 1
# calling loudness score function using loop
for l in loudness_list:
  loudness_score = get_loudness_score(l)
  print("\nLoudness for speech " + str(n) + " : " + str(l) )
  print("Loudness Score for speech " + str(n) + " : " + str(loudness_score))
  voice_temp_score.append(loudness_score)
  n = n+1

# making a np array of voice score
score_by_voice = np.array(voice_temp_score)
score_by_voice = 20*score_by_voice/100


# # **Taking an overview of speech text**

# In[56]:


print("\nText of first speech : \n" + speech_1_text)
print("\nText of second speech : \n" + speech_2_text)
print("\nText of third speech : \n" + speech_3_text)
print("\nText of fourth speech : \n" + speech_4_text)


# # **User Input**

# In[57]:


# User Input Score
user_temp_score = []
score_by_user = []
n=1
while n<=4:

    value = int(input("Give score to speech " + str(n) + " text (from 0 to 10): "))
    if value not in range(0,11):
      print("Invalid score")
    else:
        user_temp_score.append(value)
        n=n+1

score_by_user = np.array(user_temp_score)
score_by_user = score_by_user/100
print('\n')


# # **Total Rating**

# In[58]:


#calculating final score
final_temp_score = np.add(score_by_relevancy, score_by_user)
final_score = np.add(final_temp_score, score_by_voice)

#calculating percentage
perc_score_by_relevancy = 100*score_by_relevancy
perc_score_by_voice = 100*score_by_voice
perc_score_by_user = 100*score_by_user
perc_final_score = 100*final_score


# In[59]:


# Calling DataFrame constructor after zipping
df = pd.DataFrame(list(zip(score_by_relevancy, score_by_voice, score_by_user, final_score)), index =['SPEECH 1', 'SPEECH 2', 'SPEECH 3', 'SPEECH 4'],
              columns =['Relevancy Score', 'Speech Voice Score', 'User Given Score', 'Final Score'])
df


# In[60]:


# Calling DataFrame constructor after zipping
perc_df = pd.DataFrame(list(zip(perc_score_by_relevancy, perc_score_by_voice, perc_score_by_user, perc_final_score)), index =['SPEECH 1', 'SPEECH 2', 'SPEECH 3', 'SPEECH 4'],
               columns =['Relevancy Score(%)', 'Speech Voice Score(%)', 'User Given Score(%)', 'Final Score(%)'])
perc_df

