#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 19:21:23 2022

@author: john_rabovich
"""

from bs4 import BeautifulSoup as Soup
import requests
import pandas as pd
import numpy as np


# 08/20/22 pull data from fantasy football calculator into python object
ffc_response = requests.get('https://fantasyfootballcalculator.com/adp/ppr/12-team/all')

#08/20/22 parse html data from fantasy football calculator
adp_soup = Soup(ffc_response.text)

#08/20/22 below 2 lines of code are just to do some analysis and to see what it looks like ie how many tables we have
# we see table has size 1 which means there is one table 
tables = adp_soup.find_all('table')
len(tables)

# 08/20/22 below we are assigning python table the value fo the first table in the html data we retrieved but we didnt really have
# to do this step sicne there is only table in this pull
adp_table = tables[0]

# 08/20/22 simple analyisis to see what the 0 row looks like (headers) ad the 1st row is jonathan taylor data
rows = adp_table.find_all('tr')
rows[0]
first_data_row = rows[1]
first_data_row.find_all('td')

# 08/20/22 making function that uses list comprehension to parse rows
def parse_row(row):
    return [str(x.string) for x in row.find_all('td')]

list_of_parsed_rows = [parse_row(row) for row in rows[1:]]

# 08/20/22 converting list of parsed rows which is a list of lists into a dataframe
df = pd.DataFrame(list_of_parsed_rows)

# 08/20/22 assign column names to df based on column from ff calculator website page
# and then going to modify the columns to their appropriate data types(they are all currently strings). will also drop graph col
df.columns = ['ovr','pick','name','pos','team','bye','adp','std','high','low','drafted','graph']

float_cols = ['adp','std']
int_cols = ['ovr','drafted']
df[float_cols] = df[float_cols].astype(float)
df[int_cols] = df[int_cols].astype(int)

df.drop('graph',axis = 1, inplace = True)



#08/21/22 working on pulling last season player data from yahoo
url = 'https://football.fantasysports.yahoo.com/f1/317997/players?status=ALL&pos=O&cut_type=9&stat1=S_S_2021&myteam=0&sort=AR&sdir=1&count=475'
yahoo_response = requests.get(url)

yahoo_soup = Soup(yahoo_response.text)
yahoo_tables = yahoo_soup.find_all('table')
yahoo_table1 = yahoo_tables[0]
yahoo_table2 = yahoo_tables[1]
yahoo_table3 = yahoo_tables[2]
yahoo_table4 = yahoo_tables[3]

rows_t1 = yahoo_table1.find_all('tr')
rows_t2 = yahoo_table2.find_all('tr')
rows_t3= yahoo_table3.find_all('tr')
rows_t4 = yahoo_table4.find_all('tr')

def parse_data_row(row):
    return [str(x.string) for x in row.find_all('td')]

list_of_yahoo_parsed_rows = [parse_data_row(row) for row in rows_t1[2:]]

# 08/21/22 below is code to get player name from a row( rows_t1[2] siginifies josh allen)
rows_t1[2].find_all('a')[2].string

# 08/21/22 below is code to get team name and pos in one string from yahoo data(had to separate on div and span tags). 
# this locates the team and pos string on the first data row ie the josh allen row rows_t1[2]
rows_t1[2].find_all('div')[1].find_all('span')[2].string

#08/21/2022 going top make parse row functions to get player names columns and additional team-pos column
def parse_name(row):
    return [str(x.string) for x in row.find_all('a')[2]]

list_of_player_names = [parse_name(row) for row in rows_t1[2:]]

#player_names = pd.Series(list_of_player_names)

def parse_team_and_pos(row):
    return [str(x.string) for x in row.find_all('div')[1].find_all('span')[2]]

team_and_pos_list = [parse_team_and_pos(row) for row in rows_t1[2:]]

practice_df = pd.DataFrame(list_of_yahoo_parsed_rows)
practice_df2 = pd.DataFrame(list_of_player_names)
practice_df3 = pd.DataFrame(team_and_pos_list)

#08/21/22 making name and team/pos columns on practice_df the pd.DataFrame versions of the parsed a, div, and span rows
practice_df[['name','team_and_pos']] = practice_df2, practice_df3

practice_df.columns = ['row_num','drop1','drop2','drop3','fantasy_team','games_played','bye','fantasy_points',
                       'preseason_ranking','ranking_actual_2021','rostered_pct','passing_yds','passing_tds','ints'
                       ,'carries','rush_yds','rush_tds','targets','receptions','receiving_yds','receiving_tds'
                       ,'return_yds','return_tds','two_pt_conversions','fumbles_lost','drop4','name','team_and_pos']

practice_df['season'] = '2021'

#08/22/22 used below code to assign first 25 to dataframe dataset but i need to keep a note because now i am overwriting practice_df
# with each subsequent 25 values
#dataset = practice_df

#08/22/22 doing first merge dataset here. then recursively merging merge_dataset
#merge_dataset = pd.concat([dataset,practice_df], ignore_index = True)

merge_dataset = pd.concat([merge_dataset,practice_df],ignore_index = True)

#08/22/22 making ff_dataset_final = the final rendition of merge_dayaset with top 500 players and then making csv and saving it on desktop 
# for future reference

ff_dataset_final = merge_dataset

from pathlib import Path 

filepath = Path('/Users/john_rabovich/Desktop/Data science/fantasy football 2021 nb/ff_dataset_final.csv')
ff_dataset_final.to_csv(filepath)

# 08/22/22 chekc for unique values of the columns i labled dropn to determine if there is any reason i shoudl keep them
# and them dropping them if i dont need to keep them
ff_dataset_final['drop1'].unique()
ff_dataset_final['drop2'].unique()
ff_dataset_final['drop3'].unique()
ff_dataset_final['drop4'].unique()

ff_dataset_final = ff_dataset_final.drop(columns =['drop1','drop2','drop3','drop4','row_num'])

ff_dataset_final['team_and_pos']

ff_dataset_final['nfl_team'] = [x[:3].strip() for x in ff_dataset_final['team_and_pos']]
ff_dataset_final['pos'] = [x[5:].strip() for x in ff_dataset_final['team_and_pos']]

ff_dataset_final2 = ff_dataset_final[['name','pos', 'nfl_team','fantasy_points',
       'preseason_ranking', 'ranking_actual_2021', 'rostered_pct',
       'passing_yds', 'passing_tds', 'ints', 'carries', 'rush_yds', 'rush_tds',
       'targets', 'receptions', 'receiving_yds', 'receiving_tds', 'return_yds',
       'return_tds', 'two_pt_conversions', 'fumbles_lost', 
       'team_and_pos', 'season', 'nfl_team','fantasy_team','games_played', 'bye']]

#08/28/22 going to try to get more years of data and make dependent variable of preseason ranking meeting/exceed actual ranking
# and convert necessary variables to numeric/other types of feature engineering

ff_dataset_final2['fantasy_points']= ff_dataset_final2['fantasy_points'].astype(float)
ff_dataset_final2['preseason_ranking']= ff_dataset_final2['preseason_ranking'].astype(int)
ff_dataset_final2['ranking_actual_2021']= ff_dataset_final2['ranking_actual_2021'].astype(int)

ff_dataset_final2['rostered_pct']= [x.replace('%','') for x in ff_dataset_final2['rostered_pct']]
ff_dataset_final2['rostered_pct'] = ff_dataset_final2['rostered_pct'].astype(float)

ff_dataset_final2['passing_yds'] = ff_dataset_final2['passing_yds'].astype(int)
ff_dataset_final2['passing_tds'] = ff_dataset_final2['passing_tds'].astype(int)
ff_dataset_final2['ints'] = ff_dataset_final2['ints'].astype(int)

ff_dataset_final2['carries'] = ff_dataset_final2['carries'].astype(int)
ff_dataset_final2['rush_yds'] = ff_dataset_final2['rush_yds'].astype(int)
ff_dataset_final2['rush_tds'] = ff_dataset_final2['rush_tds'].astype(int)

ff_dataset_final2['targets'] = ff_dataset_final2['targets'].astype(int)
ff_dataset_final2['receptions'] = ff_dataset_final2['receptions'].astype(int)
ff_dataset_final2['receiving_yds'] = ff_dataset_final2['receiving_yds'].astype(int)
ff_dataset_final2['receiving_tds'] = ff_dataset_final2['receiving_tds'].astype(int)

ff_dataset_final2['return_yds'] = ff_dataset_final2['return_yds'].astype(int)
ff_dataset_final2['return_tds'] = ff_dataset_final2['return_tds'].astype(int)

ff_dataset_final2['two_pt_conversions'] = ff_dataset_final2['two_pt_conversions'].astype(int)
ff_dataset_final2['fumbles_lost'] = ff_dataset_final2['fumbles_lost'].astype(int)
ff_dataset_final2['season'] = ff_dataset_final2['season'].astype(int)
ff_dataset_final2['games_played'] = ff_dataset_final2['games_played'].astype(int)
ff_dataset_final2['bye'] = ff_dataset_final2['bye'].astype(int)

#08/28/22 make dependent var beat expectations below

ff_dataset_final2['beat_expectations'] = ff_dataset_final2['ranking_actual'] <= ff_dataset_final2['preseason_ranking']
ff_dataset_final2['beat_expectations'] = ff_dataset_final2['beat_expectations'].replace({True: 1,False:0})


#08/28/22 make other potentially usefull variables here
ff_dataset_final2['catch_rate'] = ff_dataset_final2['receptions'] / ff_dataset_final2['targets']


#08/28/22 to make skill pos var, need to make function and then apply function to pos column
def is_skill(pos):
    return pos in ['RB','WR','TE','WR,TE','QB,TE']

ff_dataset_final2['skill_pos'] = ff_dataset_final2['pos'].apply(is_skill)
ff_dataset_final2['skill_pos'] = ff_dataset_final2['skill_pos'].replace({True:1, False:0})


#08/30/22 making more variables like fantasy ppg and finding average ppg by posiiton so i can calculate value over replacement
ff_dataset_final2['fantasy_ppg'] = ff_dataset_final2['fantasy_points'] / ff_dataset_final2['games_played']

ff_dataset_final2['ppg_by_pos'] = np.nan
ff_dataset_final2['avg_ppg_by_pos'] = ff_dataset_final2['ppg_by_pos'] 

rb_mean = ff_dataset_final2.loc[ff_dataset_final2['pos'] == 'RB','fantasy_ppg'].mean()
wr_mean = ff_dataset_final2.loc[ff_dataset_final2['pos'] == 'WR','fantasy_ppg'].mean()
qb_mean = ff_dataset_final2.loc[ff_dataset_final2['pos'] == 'QB','fantasy_ppg'].mean()
te_mean = ff_dataset_final2.loc[ff_dataset_final2['pos'] == 'TE','fantasy_ppg'].mean()
qb_te_mean = ff_dataset_final2.loc[ff_dataset_final2['pos'] == 'QB,TE','fantasy_ppg'].mean()
wr_te_mean = ff_dataset_final2.loc[ff_dataset_final2['pos'] == 'WR,TE','fantasy_ppg'].mean()

ff_dataset_final2.loc[ff_dataset_final2['pos'] == 'QB','avg_ppg_by_pos'] = qb_mean
ff_dataset_final2.loc[ff_dataset_final2['pos'] == 'RB','avg_ppg_by_pos'] = rb_mean
ff_dataset_final2.loc[ff_dataset_final2['pos'] == 'WR','avg_ppg_by_pos'] = wr_mean
ff_dataset_final2.loc[ff_dataset_final2['pos'] == 'TE','avg_ppg_by_pos'] = te_mean
ff_dataset_final2.loc[ff_dataset_final2['pos'] == 'QB,TE','avg_ppg_by_pos'] = qb_te_mean
ff_dataset_final2.loc[ff_dataset_final2['pos'] == 'WR,TE','avg_ppg_by_pos'] = wr_te_mean

#08/30/22 making value over replacement variable
ff_dataset_final2['VOR'] = ff_dataset_final2['fantasy_ppg'] - ff_dataset_final2['avg_ppg_by_pos']


from statistics import stdev

rb_stdev = stdev(ff_dataset_final2.loc[ff_dataset_final2['pos'] == 'RB','fantasy_ppg'])
wr_stdev = stdev(ff_dataset_final2.loc[ff_dataset_final2['pos'] == 'WR','fantasy_ppg'])
qb_stdev = stdev(ff_dataset_final2.loc[ff_dataset_final2['pos'] == 'QB','fantasy_ppg'])
te_stdev = stdev(ff_dataset_final2.loc[ff_dataset_final2['pos'] == 'TE','fantasy_ppg'])
qb_te_stdev = 0
wr_te_stdev = 0

ff_dataset_final2['ppg_stdev_by_pos'] = np.nan
ff_dataset_final2.loc[ff_dataset_final2['pos'] == 'QB','ppg_stdev_by_pos'] = qb_stdev
ff_dataset_final2.loc[ff_dataset_final2['pos'] == 'RB','ppg_stdev_by_pos'] = rb_stdev
ff_dataset_final2.loc[ff_dataset_final2['pos'] == 'WR','ppg_stdev_by_pos'] = wr_stdev
ff_dataset_final2.loc[ff_dataset_final2['pos'] == 'TE','ppg_stdev_by_pos'] = te_stdev
ff_dataset_final2.loc[ff_dataset_final2['pos'] == 'QB,TE','ppg_stdev_by_pos'] = qb_te_stdev
ff_dataset_final2.loc[ff_dataset_final2['pos'] == 'WR,TE','ppg_stdev_by_pos'] = wr_te_stdev

#09/06/22 goal of todays session is to remove the players i draftged and use taht as a test set. then try out some ml models
player_list = ['Dalvin Cook','Joe Burrow','Mike Evans','Terry McLaurin','Jerry Jeudy','Michael Carter','Dalton Schultz',
               'Adam Thielen','Rashod Bateman','Rashaad Penny','Breece Hall','Aaron Rodgers','Chris Godwin','James Cook',
               'Nyheim Hines','Zach Wilson','Jahan Dotson','Dawson Knox','Darrel Williams','Dak Prescott','Nick Chubb',
               'Davante Adams','Darren Waller','Gabe Davis','Miles Sanders','Amari Cooper','Matthew Stafford','Garret Wilson',
               'Alexander Mattison']

#09/06/22 having some difficulty pulling only rows i want based on player names so im going to try some sql
import sqlite3 
from os import path
conn = sqlite3.connect('/Users/john_rabovich/Desktop/Data science/fantasy football 2021 nb/ff_dataset_final2.sqlite')

#ff_dataset_final2['on_fantasy_team'] = [x for x in ff_dataset_final2['name'] if x in player_list]

#09/07/22 needed to remove duplicate column nfl_team with below step
ff_dataset_final3 = ff_dataset_final2.drop(ff_dataset_final2.columns[3],axis = 1)
ff_dataset_final3['nfl_team'] = ff_dataset_final['nfl_team']

#09/08/22 already figured out how to make the atasets in sql but im going to convert cat vars to numeric before i do the sql step

ff_dataset_final3['pos_n'] = ff_dataset_final3['pos']
ff_dataset_final3['pos_n'].replace(['QB','RB','WR','TE','WR,TE','QB,TE'],[1,2,3,4,5,6], inplace=True)

ff_dataset_final3['fantasy_team_n'] = ff_dataset_final3['fantasy_team']
ff_dataset_final3['fantasy_team_n'].replace(['FA','el mariachi','The Dome Lebodome','One Ring Reitz','Kim Jong Warren-Moon',
                                             "Colin's Rebellion",'Here to kick some Bass','3rd String Qbs','Boobie','Chubby chasers',
                                             'Pants or No Pants','The Romping Rabos','Comeback SZN'],
                                            [0,1,2,3,4,5,6,7,8,9,10,11,12],inplace = True)

ff_dataset_final3['nfl_team_n'] = ff_dataset_final3['nfl_team']
ff_dataset_final3['nfl_team_n'].replace(['Det','NO','Buf','LV','Ari','Was','Mia','GB','SF','TB','Hou','Sea','Car','Phi','Chi',
                                         'LAC','NYJ','Cin','NYG','Pit','Ind','KC','ATL','NE','Den','Cle','Jax','LAR','Min','Bal',
                                         'Dal','Ten'],[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32],
                                        inplace = True)

#09/14/22 need to convert col nfl_team_n to numeric
ff_dataset_final3['nfl_team_n'] = pd.to_numeric(ff_dataset_final3['nfl_team_n'],errors = 'coerce')


conn = sqlite3.connect('/Users/john_rabovich/Desktop/Data science/fantasy football 2021 nb/ff_dataset_final3.sqlite')


filepath = Path('/Users/john_rabovich/Desktop/Data science/fantasy football 2021 nb/ff_dataset_final3.csv')
ff_dataset_final3.to_csv(filepath)

ff_dataset_final3.to_sql('ff_dataset_final3',conn, index = False, if_exists ='replace')



my_players_df = pd.read_sql("""
         SELECT *
         FROM ff_dataset_final3
         WHERE name in ('Dalvin Cook','Joe Burrow','Mike Evans','Terry McLaurin','Jerry Jeudy','Michael Carter','Dalton Schultz',
               'Adam Thielen','Rashod Bateman','Rashaad Penny','Breece Hall','Aaron Rodgers','Chris Godwin','James Cook',
               'Nyheim Hines','Zach Wilson','Jahan Dotson','Dawson Knox','Darrel Williams','Dak Prescott','Nick Chubb',
               'Davante Adams','Darren Waller','Gabe Davis','Miles Sanders','Amari Cooper','Matthew Stafford','Garret Wilson',
               'Alexander Mattison')
         
         """,conn)
         
train =  pd.read_sql("""
         SELECT *
         FROM ff_dataset_final3
         WHERE name NOT IN ('Dalvin Cook','Joe Burrow','Mike Evans','Terry McLaurin','Jerry Jeudy','Michael Carter','Dalton Schultz',
               'Adam Thielen','Rashod Bateman','Rashaad Penny','Breece Hall','Aaron Rodgers','Chris Godwin','James Cook',
               'Nyheim Hines','Zach Wilson','Jahan Dotson','Dawson Knox','Darrel Williams','Dak Prescott','Nick Chubb',
               'Davante Adams','Darren Waller','Gabe Davis','Miles Sanders','Amari Cooper','Matthew Stafford','Garret Wilson',
               'Alexander Mattison')
         
         """,conn)
         
train.drop('ppg_by_pos', axis = 1, inplace = True)
my_players_df.drop('ppg_by_pos', axis = 1, inplace = True)



import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import statsmodels.formula.api as smf

#09/21/22 filling missing values columns w appropriate values
train['nfl_team_n'] = train['nfl_team_n'].fillna(33)
my_players_df['nfl_team_n'] = my_players_df['nfl_team_n'].fillna(33)

train['catch_rate'] = train['catch_rate'].fillna(0)
my_players_df['catch_rate'] = my_players_df['catch_rate'].fillna(0)

X_train = train[['fantasy_points', 'preseason_ranking',
       'ranking_actual_2021', 'rostered_pct', 'passing_yds', 'passing_tds',
       'ints', 'carries', 'rush_yds', 'rush_tds', 'targets', 'receptions',
       'receiving_yds', 'receiving_tds', 'return_yds', 'return_tds',
       'two_pt_conversions', 'fumbles_lost', 'season',
        'games_played', 'bye', 'ranking_actual',
        'catch_rate', 'skill_pos', 'fantasy_ppg',
       'avg_ppg_by_pos', 'VOR', 'ppg_stdev_by_pos', 
       'pos_n', 'fantasy_team_n', 'nfl_team_n']]

X_valid = my_players_df[['fantasy_points', 'preseason_ranking',
       'ranking_actual_2021', 'rostered_pct', 'passing_yds', 'passing_tds',
       'ints', 'carries', 'rush_yds', 'rush_tds', 'targets', 'receptions',
       'receiving_yds', 'receiving_tds', 'return_yds', 'return_tds',
       'two_pt_conversions', 'fumbles_lost', 'season',
        'games_played', 'bye', 'ranking_actual',
        'catch_rate', 'skill_pos', 'fantasy_ppg',
        'avg_ppg_by_pos', 'VOR', 'ppg_stdev_by_pos', 
       'pos_n', 'fantasy_team_n', 'nfl_team_n']]

y_train = train['beat_expectations']
y_valid = my_players_df['beat_expectations']


logistic_reg_model = LogisticRegression(penalty='l2',solver = 'liblinear',random_state = 0)

"""
log reg parameters: 
    penalty-decides whether there is regularization and which type to use. L2 reg is default
    solver-type of minimization algorithm(cost function) you use
    random state- random seed number

"""
logistic_reg_model.fit(X_train,y_train)


#09/24/22 get the intercept and slope of the reg model below
logistic_reg_model.intercept_
logistic_reg_model.coef_

#09/24/22 below i have predictions of probabilities of outcomes and also how model competes against oos predictions
logistic_reg_model.predict_proba(X_valid)

logistic_reg_model.score(X_valid,y_valid)

#09/26/2022 importing confusion matrix for post validation metrics
from sklearn.metrics import confusion_matrix
from sklearn import metrics

y_pred = logistic_reg_model.predict(X_valid)

confusion_matrix(y_valid,y_pred)

#09/27/22 want to do some data viz, think i need to do feature correlation to target first
print(metrics.r2_score(y_valid,y_pred))

feature_corr_2_target = X_train.apply(lambda x: x.corr(y_train))

feature_corr_2_target.to_csv('/Users/john_rabovich/Desktop/Data science/fantasy football 2021 nb/feature_corr_2_target.csv')

feature_corr_2_target.columns = ['features','corr2target']


# 09/27/22 going to make auc graph and get auc metrics
# calculate the fpr and tpr for all thresholds of the classification

fpr, tpr, threshold = metrics.roc_curve(y_valid, y_pred)
roc_auc = metrics.auc(fpr, tpr)

import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()










