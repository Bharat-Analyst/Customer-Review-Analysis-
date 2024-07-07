#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('vader_lexicon')
import seaborn as sns
import matplotlib.pyplot as plt


# https://www.youtube.com/watch?v=QpzMWQvxXWk

# In[2]:


df = pd.read_csv("Women_s_E-Commerce_Clothing_Reviews_1594_1.csv", sep=';', on_bad_lines='skip')


# In[3]:


df['Review.Text'].fillna('',inplace = True)
df.insert(0, 'Serial Number', range(1, len(df) + 1))
df.head()


# In[4]:


example = df["Review.Text"][50]
print(example)
tokens = nltk.word_tokenize(example)
tagged = nltk.pos_tag(tokens)
tagged[:10]


# ## VADER SENTIMENT ANALYIS

# In[5]:


from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm

sia = SentimentIntensityAnalyzer()
sia.polarity_scores("Most happy")


# In[6]:


example = "I can't find a job I hate it"
sia.polarity_scores(example)


# ## Run Polarity score for the entire dataset

# In[7]:


res = {}
for i,rows in tqdm(df.iterrows(),total = len(df)):
    text = rows['Review.Text']
    myid = rows['Serial Number']
    res[myid] = sia.polarity_scores(text)


# In[8]:


vader_polarity_scores = pd.DataFrame(res).T
vader_polarity_scores = vader_polarity_scores.reset_index().rename(columns = {"index" : "Serial Number"}).merge(df,how='left')
# Now we have orginial dataset and Vader Score
vader_polarity_scores


# ## Plot the Vader Scores

# In[9]:


ax_main = sns.barplot(data= vader_polarity_scores,x="Rating",y="compound")
ax_main.set_title("Compound Vader Score against Rating")
plt.show()

fig, axs = plt.subplots(1, 3, figsize=(15,5))
sns.barplot(data= vader_polarity_scores,x="Rating",y="pos",ax=axs[0])
sns.barplot(data= vader_polarity_scores,x="Rating",y="neu",ax=axs[1])
sns.barplot(data= vader_polarity_scores,x="Rating",y="neg",ax=axs[2])
axs[0].set_title("Positive")
axs[1].set_title("Neutral")
axs[2].set_title("Negative")
plt.show()


# ## Roberta Pretained Model

# In[10]:


from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = f'cardiffnlp/twitter-roberta-base-sentiment-latest'
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


# In[11]:


#old Vader Model Score
print(example)

sia.polarity_scores(example)


# In[12]:


# Roberta Model Score
#example
encoded_text = tokenizer(example, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
score_dictonary = {
    "roberta_neg" : scores[0],
    "roberta_neu" : scores[1],
    "roberta_pos" : scores[2]
}
score_dictonary


# In[13]:


#function to apply to our dataset
def polarity_score_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    score_dictonary = {
    "roberta_neg" : scores[0],
    "roberta_neu" : scores[1],
    "roberta_pos" : scores[2]
    }
    return score_dictonary


# In[19]:


roberta_res = {}
for i,rows in tqdm(df.iterrows(),total = len(df)):
    text = rows['Review.Text']
    myid = rows['Serial Number']
    roberta_res[myid] = polarity_score_roberta(text)
    
roberta_res


# In[20]:


roberta_polarity_scores = pd.DataFrame(roberta_res).T
roberta_polarity_scores = roberta_polarity_scores.reset_index().rename(columns = {"index" : "Serial Number"}).merge(vader_polarity_scores,how='left')
# Now we have orginial dataset and Vader Score
roberta_polarity_scores


# In[30]:


fig, axs = plt.subplots(3, 2, figsize=(15, 15))  # Adjusted figsize for better visibility

sns.barplot(data=roberta_polarity_scores, x="Rating", y="pos", ax=axs[0, 0])
sns.barplot(data=roberta_polarity_scores, x="Rating", y="neu", ax=axs[0, 1])
sns.barplot(data=roberta_polarity_scores, x="Rating", y="neg", ax=axs[1, 0])
sns.barplot(data=roberta_polarity_scores, x="Rating", y="roberta_pos", ax=axs[1, 1])
sns.barplot(data=roberta_polarity_scores, x="Rating", y="roberta_neu", ax=axs[2, 0])
sns.barplot(data=roberta_polarity_scores, x="Rating", y="roberta_neg", ax=axs[2, 1])

axs[0, 0].set_title("Positive")
axs[0, 1].set_title("Neutral")
axs[1, 0].set_title("Negative")
axs[1, 1].set_title("Roberta_Positive")
axs[2, 0].set_title("Roberta_Neutral")
axs[2, 1].set_title("Roberta_Negative")

plt.tight_layout()  # Adjusts subplots to fit into the figure area.
plt.show()


# In[33]:


df.to_csv("roberta_polarity_scores", index=False) 
##So I can furthur use it for Recommendation System and Exploratory data Analysis 


# In[ ]:




