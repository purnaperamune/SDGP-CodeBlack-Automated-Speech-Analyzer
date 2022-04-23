#!/usr/bin/env python
# coding: utf-8

# In[1]:


path_medical_word_list = 'wordlist.txt'


# In[2]:


def get_medical_sector_words(pth):
  Medical_words = []
  with open(pth) as f:
    Medical_words.append(f.readlines())
  Medical_sector_words = ""
  for i in Medical_words[0]:
    Medical_sector_words  = Medical_sector_words + " "+(i.replace('\n',''))
  return Medical_sector_words


# In[3]:


Medical_sector_words_lest = get_medical_sector_words(path_medical_word_list)


# In[4]:


speech_one = '''The moving images of a film are created by photographing actual scenes with a motion-picture camera, by photographing drawings or miniature models using traditional animation techniques, by means of CGI and computer animation, or by a combination of some or all of these techniques, and other visual effects.
Before the introduction of digital production, series of still images were recorded on a strip of chemically sensitized celluloid (photographic film stock), usually at the rate of 24 frames per second. The images are transmitted through a movie projector at the same rate as they were recorded, with a Geneva drive ensuring that each frame remains still during its short projection time. A rotating shutter causes stroboscopic intervals of darkness, but the viewer does not notice the interruptions due to flicker fusion.'''


# In[5]:


speech_two = '''Doctors play a pivotal role in building the society. They are the lifelines of the community.
This term can be used very literally for the Doctors who shape culture and save those that are diseased and unhealthy. 
Doctors work hard to save the lives of patients with serious ailments. 
They act as an inspiration to society.'''


# In[6]:


speech_three = ''' Common signs of infection include respiratory symptoms, fever, cough, 
shortness of breath and breathing difficulties. In more severe cases, 
infection can cause pneumonia, severe acute respiratory syndrome, 
kidney failure and even death.'''


# In[7]:


speech_four ='''Cricket is a sport. 
 '''


# In[8]:


from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import numpy as np
import numpy.linalg as LA
from nltk.corpus import stopwords
import nltk
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize,pos_tag
from nltk.stem import PorterStemmer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('stopwords')


# In[9]:


def remove_whitespace(text):
    return  " ".join(text.split())


# In[10]:



def tokenization(x):
  return word_tokenize(x)


# In[11]:



en_stopwords = stopwords.words('english')
def remove_stopwords(text):
    result = []
    for token in text:
        if token not in en_stopwords:
            result.append(token)
            
    return result


# In[12]:


def remove_punct(text):
    
    tokenizer = RegexpTokenizer(r"\w+")
    lst=tokenizer.tokenize(' '.join(text))
    return lst


# In[13]:


def lemmatization(text):
    
    result=[]
    wordnet = WordNetLemmatizer()
    for token,tag in pos_tag(text):
        pos=tag[0].lower()
        
        if pos not in ['a', 'r', 'n', 'v']:
            pos='n'
            
        result.append(wordnet.lemmatize(token,pos))
    
    return result


# In[14]:


def getstring(txt_lst):
  wrd = ""
  for i in txt_lst:
    wrd  = wrd+" "+i
  return wrd.strip()


# In[15]:


def preprocessing(text):
  txt = remove_whitespace(text).lower()
  txt = tokenization(txt)
  txt = remove_stopwords(txt)
  txt = remove_punct(txt)
  txt = lemmatization(txt)
  return getstring(txt)


# In[16]:


s1 = preprocessing(speech_one)
s2 = preprocessing(speech_two)
s3 = preprocessing(speech_three)
s4 = preprocessing(speech_four)
m_wrds = preprocessing(Medical_sector_words_lest)


# In[17]:


train = [m_wrds]
test = [s1,s2,s3,s4]


# In[18]:


tfidfvectorizer = TfidfVectorizer(analyzer='word',stop_words= 'english')


# In[19]:


tfidf_wm = tfidfvectorizer.fit_transform(train)
tfidf_wm_test = tfidfvectorizer.transform(test)


# In[20]:


cx = lambda a, b : round(np.inner(a, b)/(LA.norm(a)*LA.norm(b)), 3)


# In[21]:


cx(tfidf_wm.toarray()[0],tfidf_wm_test.toarray()[0])


# In[22]:


similarity_1 = cx(tfidf_wm_test.toarray()[0],tfidf_wm.toarray()[0])
similarity_2 = cx(tfidf_wm.toarray()[0],tfidf_wm_test.toarray()[1])
similarity_3 = cx(tfidf_wm.toarray()[0],tfidf_wm_test.toarray()[2])
similarity_4 = cx(tfidf_wm.toarray()[0],tfidf_wm_test.toarray()[3])


# In[23]:


print('Simalarity of Speech 1 to medical sector is '+ str(similarity_1))
print('Simalarity of Speech 2 to medical sector is '+ str(similarity_2))
print('Simalarity of Speech 3 to medical sector is '+ str(similarity_3))
print('Simalarity of Speech 4 to medical sector is '+ str(similarity_4))


# In[24]:


#more similarity mean more relevance to medical. 
#so speech 4 is less similar or not relevance to medical

