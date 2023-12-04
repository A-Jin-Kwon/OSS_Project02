#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd


# In[9]:


df = pd.read_csv('2019_kbo_for_kaggle_v2.csv')


# In[10]:


top_ten_players = df[['H', 'avg', 'HR', 'OBP']]


# In[11]:


cols1 = ['H', 'avg', 'HR', 'OBP']

for year in range(2015, 2019):
    print(f"<In {year}>")
    y = df[df['year'] == year]
    
    for col in cols1:
        data = y.nlargest(10, col)[['batter_name', col]]
        print(f"\n{data}" )


# In[28]:


data_2018 = df[df['year'] == 2018]

positions = ['포수', '1루수', '2루수', '3루수', '유격수', '좌익수', '중견수', '우익수']

for position in positions:
    data = data_2018[data_2018['cp'] == position].nlargest(1, 'war')
    top_player_name = data['batter_name'].values[0]
    top_player_cp = data['war'].values[0]
    print(f"2018 Highest war {position}: {top_player_name} {top_player_cp}\n")


# In[32]:


cols = ['salary', 'R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG']
data = df[cols]

correlations = data.corr().loc['salary', cols[1:]]
print(f"{correlations}\n")

max_col = correlations.idxmax()

print(f"Highest correlation with salary: {max_col}")


# In[ ]:




