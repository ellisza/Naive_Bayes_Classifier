# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import re
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns; sns.set()

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

#%%
# clean data
df = df.drop(df.index[[0,1]])
df = df.drop(df.index[90:])
#%%
df = df.drop('junk1', 1)
df = df.drop('junk2', 1)
#%%
df.games = df.games.str.replace(',','').astype('float')
#%%
df.heroDamage = df.heroDamage.str.replace(',','').astype('float')
#%%
df.siegeDamage = df.siegeDamage.str.replace(',','').astype('float')
#%%
df.healing = df.healing.astype('float')
#%%
df.selfHeal = df.selfHeal.str.replace(',','').astype('float')
#%%
df.damageTaken = df.damageTaken.astype('float')
#%%
df.expGain = df.expGain.str.replace(',','').astype('float')
#%%
df.winPercent = df.winPercent.str.replace(',','').astype('float')
#%%
df.head()
# %%
# Save to csv 
df.to_csv(r'hotsData.csv', index=False, header=True) # saves to local working directory

#%%
df = pd.read_csv('hotsData.csv') # reload from file in local working directory
#%%
# add roles column
for index, row in df.iterrows():
    if row.hero == 'Diablo' or row.hero == 'Cho' or row.hero == 'Stitches' or row.hero == 'Muradin' or row.hero == 'Johanna' or row.hero == 'Tyrael' or row.hero == 'Mal\'Ganis' or row.hero == 'Arthas' or row.hero == 'Mei' or row.hero == 'Blaze' or row.hero == 'E.T.C.' or row.hero == 'Anub\'arak' or row.hero == 'Garrosh':
        df.at[index, 'roles'] = 'tank'
    if row.hero == 'L\u00FAcio' or row.hero == 'Anduin' or row.hero == 'Brightwing' or row.hero == 'Alexstrasza' or row.hero == 'Li Li' or row.hero == 'Stukov' or row.hero == 'Whitemane' or row.hero == 'Malfurion' or row.hero == 'Lt. Morales' or row.hero == 'Deckard' or row.hero == 'Rehgar' or row.hero == 'Auriel' or row.hero == 'Kharazim' or row.hero == 'Ana' or row.hero == 'Tyrande' or row.hero == 'Uther':
        df.at[index, 'roles'] = 'healer'
    if  row.hero == 'Abathur' or row.hero == 'Nazeebo' or row.hero == 'The Lost Vikings' or row.hero == 'Chen' or row.hero == 'Medivh' or row.hero == 'Zarya' or row.hero == 'Rexxar' or row.hero == 'Leoric' or row.hero == 'D.Va' or row.hero == 'Varian' or row.hero == 'Dehaka' or row.hero == 'Yrel' or row.hero == 'Artanis' or row.hero == 'Imperius' or row.hero == 'Sonya' or row.hero == 'Hogger' or row.hero == 'Malthael' or row.hero == 'Thrall' or row.hero == 'Xul' or row.hero == 'Ragnaros' or row.hero == 'Deathwing' or row.hero == 'Alarak' or row.hero == 'Murky' or row.hero == 'Samuro' or row.hero == 'Valeera' or row.hero == 'Maiev' or row.hero == 'Gazlowe' or row.hero == 'Zeratul' or row.hero == 'Kerrigan' or row.hero == 'Qhira' or row.hero == 'The Butcher' or row.hero == 'Illidan' or row.hero == 'Li-Ming' or row.hero == 'Chromie' or row.hero == 'Gall' or row.hero == 'Nova' or row.hero == 'Hanzo' or row.hero == 'Mephisto' or row.hero == 'Junkrat' or row.hero == 'Fenix' or row.hero == 'Tracer' or row.hero == 'Lunara' or row.hero == 'Tychus' or row.hero == 'Falstad' or row.hero == 'Kael\'thas' or row.hero == 'Genji' or row.hero == 'Tassadar' or row.hero == 'Cassia' or row.hero == 'Orphea' or row.hero == 'Jaina' or row.hero == 'Kel\'Thuzad' or row.hero == 'Greymane' or row.hero == 'Sylvanas' or row.hero == 'Raynor' or row.hero == 'Gul\'dan' or row.hero == 'Valla' or row.hero == 'Azmodan' or row.hero == 'Zul\'jin' or row.hero == 'Probius' or row.hero == 'Nezeebo' or row.hero == 'Sgt. Hammer' or row.hero == 'Zagara':
        df.at[index, 'roles'] = 'dps'
    # df.at[index, 'healing'] = df.at[index,'healing'] + '.0'
    # df.at[index, 'damageTaken'] = df.at[index,'damageTaken'] + '.0'

    print(row.hero, row.roles)


#%%

# df['roles'].loc[df['hero'] == 'Nazeebo'] = 'dps'
df.head(30)
#%%

testData = df.drop(['hero','roles'], axis=1)
testData.head()
#%%
testDataFix = testData.convert_dtypes()
testDataFix = testDataFix.astype({'healing': 'Float64'}).dtypes

#%%
xtrain = df.drop(['roles', 'hero'], axis=1)
ytrain = df.loc[:,'roles']

xtest = df.drop(['roles', 'hero'], axis=1)
ytest = df.loc[:,'roles']

# %%
model = GaussianNB()
# %%
model.fit(xtrain, ytrain)
# %%
pred = model.predict(xtest)
print(pred)
#%%
mat = confusion_matrix(pred, ytest)
names = np.unique(pred)
sns.heatmap(mat, square=True, annot=True, fmt='d', cbar=False, xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')
# %%
