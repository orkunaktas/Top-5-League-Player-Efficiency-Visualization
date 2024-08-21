#!/usr/bin/env python
# coding: utf-8

# # Top 5 League Best Players

# In[157]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# Scrape the data from fbref.com from the 2023/2024 season
url = 'https://fbref.com/en/comps/Big5/2023-2024/stats/players/2023-2024-Big-5-European-Leagues-Stats'

df = pd.read_html(url, attrs={'id': 'stats_standard'})[0]
df.head()


# In[159]:


df.columns = df.columns.droplevel()


# In[161]:


df[(df["Player"] == "Player")]


# In[163]:


df.columns = ['Rk', 'Player', 'Nation', 'Pos', 'Squad', 'Comp', 'Age', 'Born', 'MP',
              'Starts', 'Min', '90s', 'Gls', 'Ast', 'G+A', 'G-PK', 'PK', 'PKatt',
              'CrdY', 'CrdR', 'xG', 'npxG', 'xAG', 'npxG+xAG', 'PrgC', 'PrgP', 'PrgR',
              'Gls_90', 'Ast_90', 'G+A_90', 'G-PK_90', 'G+A-PK_90', 'xG_90', 'xAG_90',
              'xG+xAG_90', 'npxG_90', 'npxG+xAG_90', 'Matches']


# In[165]:


df = df[df['Player'] != 'Player']


# In[167]:


df.drop(columns=['Matches'], inplace=True)


# In[171]:


len(df)


# In[169]:


df.to_excel("top5-players.xlsx",index=False)


# In[175]:


df['xG'] = pd.to_numeric(df['xG'], errors='coerce')
df['Gls'] = pd.to_numeric(df['Gls'], errors='coerce')

fig, ax = plt.subplots()
ax.scatter(df['xG'], df['Gls'], alpha=0.5)
ax.set_xlabel('Expected Goals')
ax.set_ylabel('Goals Scored')
ax.set_title('Goals vs Expected Goals in 2023/2024\nEuropean Big 5 leagues')
ax.plot([15, 80], [15, 80], color='black', linestyle='--')

df['difference'] = df['Gls'] - df['xG']
overperformers = df.nlargest(5, 'difference')
underperformers = df.nsmallest(5, 'difference')

for i in range(5):
    ax.text(overperformers.iloc[i]['xG'], overperformers.iloc[i]['Gls'], overperformers.iloc[i]['Player'])
    ax.text(underperformers.iloc[i]['xG'], underperformers.iloc[i]['Gls'], underperformers.iloc[i]['Player'])

plt.show()


# In[133]:


fig, ax = plt.subplots()

ax.scatter(df['xG'], df['Gls'], alpha=0.5)
ax.set_xlim(0, 40)
ax.set_ylim(0, 40)


# In[135]:


ax.plot([0, 60], [0, 60], color='black', linestyle='--')


# In[137]:


fig


# In[139]:


df['difference'] = df['Gls'] - df['xG']
overperformers = df.nlargest(5, 'difference')
underperformers = df.nsmallest(5, 'difference')

for i in range(5):
    ax.text(overperformers.iloc[i]['xG'], overperformers.iloc[i]['Gls'], overperformers.iloc[i]['Player'])
    ax.text(underperformers.iloc[i]['xG'], underperformers.iloc[i]['Gls'], underperformers.iloc[i]['Player'])


# In[141]:


fig


# In[143]:


ax.text(2, 35, 'Overperforming xG', ha='left', va='bottom', fontsize=10)
ax.text(38, 2, 'Underperforming xG', ha='right', va='bottom', fontsize=10)


# In[145]:


fig


# In[147]:


import pandas as pd
import matplotlib.pyplot as plt

df['xG'] = pd.to_numeric(df['xG'], errors='coerce')
df['Gls'] = pd.to_numeric(df['Gls'], errors='coerce')

fig, ax = plt.subplots()

ax.scatter(df['xG'], df['Gls'], alpha=0.5)
ax.set_xlim(0, 40)
ax.set_ylim(0, 40)

ax.set_xlabel('Expected Goals')
ax.set_ylabel('Goals Scored')
ax.set_title('Goals vs Expected Goals in 2023/2024\nEuropean Big 5 leagues')

ax.plot([0, 60], [0, 60], color='black', linestyle='--')

# Farkı hesaplayarak aşırı performans gösteren ve altında kalan oyuncuları bulma
df['difference'] = df['Gls'] - df['xG']
overperformers = df.nlargest(15, 'difference')
underperformers = df.nsmallest(15, 'difference')

for i in range(5):
    ax.text(overperformers.iloc[i]['xG'], overperformers.iloc[i]['Gls'], overperformers.iloc[i]['Player'],
            fontsize=10, ha='right', va='bottom')
    ax.text(underperformers.iloc[i]['xG'], underperformers.iloc[i]['Gls'], underperformers.iloc[i]['Player'],
            fontsize=10, ha='left', va='bottom')

ax.text(2, 36, 'Overperforming xG', ha='left', va='bottom', fontsize=10)
ax.text(38, 2, 'Underperforming xG', ha='right', va='bottom', fontsize=10)

plt.show()


# In[155]:


overperformers[["Player","Pos","difference"]]


# In[153]:


underperformers[["Player","Pos","difference"]]


# In[ ]:





# In[ ]:





# In[ ]:





# In[321]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

fig, axs = plt.subplots(2, 3, figsize=(18, 10))
metrics = ['Gls', 'Ast', 'xG', 'npxG', 'xAG', 'npxG+xAG']

for i, metric in enumerate(metrics):
    numeric_data = pd.to_numeric(df[metric], errors='coerce')
    
    sns.histplot(numeric_data.dropna(), kde=True, ax=axs[i//3, i%3])
    axs[i//3, i%3].set_title(f'Distribution of {metric}')
    
    # Xtick
    xticks = np.linspace(numeric_data.min(), numeric_data.max(), 5)  # 5 xtick
    axs[i//3, i%3].set_xticks(xticks)

plt.tight_layout()
plt.show()


# In[323]:


import matplotlib.pyplot as plt

team_performance = df.groupby('Team').agg({
    'Gls': 'sum',
    'Ast': 'sum',
    'xG': 'sum',
    'npxG': 'sum',
    'xAG': 'sum'
}).reset_index()



top_teams = team_performance.sort_values(by='Gls', ascending=False).head(20)

top_teams_gls_ast = top_teams.sort_values(by=['Gls', 'Ast'], ascending=[True, True])

top_teams_xg_npxg = top_teams.sort_values(by=['xG', 'npxG'], ascending=[True, True])

fig, axs = plt.subplots(1, 2, figsize=(14, 6))

top_teams_gls_ast.plot(kind='barh', x='Squad', y=['Gls', 'Ast'], ax=axs[0], legend=True)
axs[0].set_title('Goals and Assists of the Top 20 Teams')

top_teams_xg_npxg.plot(kind='barh', x='Squad', y=['xG', 'npxG'], ax=axs[1], legend=True)
axs[1].set_title('xG and npxG of the Top 20 Teams')

plt.tight_layout()
plt.show()


# In[325]:


import matplotlib.pyplot as plt
import seaborn as sns

alpha = 1.5 
beta = 1.2  
gamma = 1.4  
delta = 1.1 
epsilon = 0.8 
zeta = 0.7   
eta = 0.3    
theta = 1.2  

numeric_columns = ['Gls', 'Ast', 'xG', 'xAG', '90s', 'npxG', 'PrgC', 'G+A']
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

df['Efficiency_Score'] = (alpha * df['Gls'].fillna(0) +
                          beta * df['Ast'].fillna(0) +
                          gamma * df['xG'].fillna(0) +
                          delta * df['xAG'].fillna(0) +
                          epsilon * df['90s'].fillna(0) +
                          zeta * df['npxG'].fillna(0) +
                          eta * df['PrgC'].fillna(0) +
                          theta * df['G+A'].fillna(0))

top_20_players = df[['Player', 'Pos', 'Efficiency_Score']].sort_values(by='Efficiency_Score', ascending=False).head(20)

plt.figure(figsize=(14, 10))
sns.set(style="whitegrid")

bar_plot = sns.barplot(x='Efficiency_Score', y='Player', data=top_20_players, edgecolor='black')

plt.xlabel('Efficiency Score')
plt.ylabel('Player')
plt.title('Top 20 player with the highest Efficiency score')

plt.show()


# In[252]:


top_20_players


# In[327]:


seasonal_performance = df.groupby('Pos')['Efficiency_Score'].mean()

plt.figure(figsize=(12, 8))
sns.lineplot(x=seasonal_performance.index, y=seasonal_performance.values)
plt.title('Average Efficiency Scores of Positions')
plt.xlabel('Position')
plt.ylabel('Efficiency Score')
plt.show()


# In[294]:


import matplotlib.pyplot as plt
import seaborn as sns

team_avg_efficiency = df.groupby('Squad')['Efficiency_Score'].mean().reset_index()
team_avg_efficiency = team_avg_efficiency.sort_values(by='Efficiency_Score', ascending=True).tail(20)


plt.figure(figsize=(12, 8))
plt.barh(team_avg_efficiency['Squad'], team_avg_efficiency['Efficiency_Score'], color='skyblue', edgecolor='black')
plt.xlabel('Efficiency Score')
plt.ylabel('Team')
plt.title('Teams Average Efficiency Score')
plt.show()

team_total_goals = df.groupby('Squad')['Gls'].sum().reset_index()
team_total_goals = team_total_goals.sort_values(by='Gls', ascending=True).tail(20)


plt.figure(figsize=(12, 8))
plt.barh(team_total_goals['Squad'], team_total_goals['Gls'], color='lightcoral', edgecolor='black')
plt.xlabel('Goal')
plt.ylabel('Team')
plt.title('Teams Total Goals')
plt.tight_layout()
plt.show()


# In[303]:


import matplotlib.pyplot as plt

team_avg_xg = df.groupby('Squad')['xG'].mean().reset_index()
team_avg_xg = team_avg_xg.sort_values(by='xG', ascending=True).tail(20)

plt.figure(figsize=(12, 8))
plt.barh(team_avg_xg['Squad'], team_avg_xg['xG'], color='deepskyblue', edgecolor='black')
plt.xlabel('Average xG')
plt.ylabel('Team')
plt.title('Average xG by Team')
plt.show()

team_avg_xag = df.groupby('Squad')['xAG'].mean().reset_index()
team_avg_xag = team_avg_xag.sort_values(by='xAG', ascending=True).tail(20)

plt.figure(figsize=(12, 8))
plt.barh(team_avg_xag['Squad'], team_avg_xag['xAG'], color='gold', edgecolor='black')
plt.xlabel('Average xAG')
plt.ylabel('Team')
plt.title('Average xAG by Team')
plt.show()


# In[ ]:




