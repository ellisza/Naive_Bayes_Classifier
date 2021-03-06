# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import re

# %%
# Get the html
html = requests.get(
    'https://www.hotslogs.com/Sitewide/ScoreResultStatistics?Tab2=roleTabBruiser'
)
soup = BeautifulSoup(html.text, 'html5lib')

# %%
# sift the the html table to get the rows and columns
# to build the DataFrame
rows = []
columns = []

for row in soup.find_all('tr'):
    for column in row.find_all('td'):
        value = column.text.replace('\n', '')
        columns.append(value)
    columns = []
    rows.append(columns)

rows
# %%
# Build the DataFrame
headers = ['junk1', 'hero', 'games', 'winPercent', 'junk2',  'takedownDeathRatio', 'takedowns', 'kills', 'deaths', 'heroDamage', 'siegeDamage', 'healing', 'selfHeal', 'damageTaken', 'expGain']
df = pd.DataFrame(data=rows, columns=headers)
df = df.drop(df.index[[0,1]])
df = df.drop(df.index[90:])
df
# %%
# Save to csv 
df.to_csv(r'hotsData.csv', index=False, header=True) # saves to local working directory
df = pd.read_csv('hotsData.csv') # reload from file in local working directory

#%%
df