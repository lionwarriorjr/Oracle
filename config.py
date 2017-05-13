import os
import sys
import numpy as np
from scipy import stats
import pandas as pd
import nltk
from collections import defaultdict
import re
import itertools
import jellyfish
from functools import reduce
import plotly

desc_filename = os.path.join(os.getcwd(), 'sample_query.csv')
X = pd.read_csv(desc_filename) # read season data and store in config.X

# domain specific feature coercing
X['inning'] = X['inning'].astype(str)
X['Date'] = pd.to_datetime(X['Date'])
months = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
          7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
X['month'] = [months[t.month] for t in X.Date]

plotly.tools.set_credentials_file(username='smohan', api_key='UGKT8GURMTb64LlCxpBm') # Plotly API Credentials
filtered = X.copy() # filtered updated automatically on user query
qry = '' # store current user query as config.qry

# initialize keyword set
keywords = defaultdict(str)
keywords['and'] = 'AND'
keywords['or'] = 'OR'
keywords['not'] = 'NOT'
keywords['of'] = '->'
keywords['that'] = '->'
keywords['in'] = '=>*(%filter%)'
keywords['on'] = '=>*(%filter%)'
keywords['for'] = '=>*(%filter%)'
keywords['from'] = '=>*(%filter%)'
keywords['when'] = '=>*(%filter%)'
keywords['where'] = '=>*(%filter%)'
keywords['with'] = '=>*(%filter%)'
keywords['against'] = '=>*(%filter%)'
keywords['by'] = '=>*(%by%)'
keywords['along'] = '=>*(%by%)'
keywords['over'] = '=>*(%over%)'
keywords['under'] = '=>*(%under%)'
keywords['above'] = '=>*(%over%)'
keywords['below'] = '=>*(%under%)'
keywords['between'] = '=>*(%between%)'
keywords['except'] = '=>*(%except%)'
keywords['without'] = '=>*(%except%)'
keywords['near'] = '=>*(%near%)'
keywords['through'] = '=>*(%range%)'
keywords['until'] = '=>*(%until%)'
keywords['to'] = '=>*(%to%)'
keywords['after'] = '=>*(%after%)'
keywords['before'] = '=>*(%before%)'
keywords['compare'] = '=>*(%compare%)'

filters = set()
GENITIVE = '->'

# initialize stoplist set
stoplist = {}
stoplist['V'] = set()
stoplist['V'].add('is')
stoplist['V'].add('are')
stoplist['V'].add('am')
stoplist['V'].add('be')
stoplist['V'].add('was')
stoplist['V'].add('were')
stoplist['V'].add('do')
stoplist['V'].add('does')
stoplist['V'].add('did')

# Initialize preset substitution list
subs = defaultdict(str)
subs['(time)->(percentage)'] = '(*PCT*)'
subs['(time)->(percent)'] = '(*PCT*)'

# add to domain-specific background information to the system's domain knowledge
# account for common modifiers that reference features in the dataset
DOMAIN_KNOWLEDGE = defaultdict(lambda : defaultdict(list))
DOMAIN_KNOWLEDGE['month']['month'] = list(X['month'].unique())
DOMAIN_KNOWLEDGE['team_id_pitcher']['rangers'] = ['texmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['angels'] = ['anamlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['diamondbacks'] = ['arimlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['white sox'] = ['chamlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['reds'] = ['cinmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['rockies'] = ['colmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['tigers'] = ['detmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['brewers'] = ['milmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['yankees'] = ['nyamlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['athletics'] = ['oakmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['phillies'] = ['phimlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['pirates'] = ['pitmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['mariners'] = ['seamlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['rays'] = ['tbamlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['blue jays'] = ['tormlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['nationals'] = ['wasmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['cubs'] = ['chnmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['dodgers'] = ['lanmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['marlins'] = ['miamlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['padres'] = ['sdnmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['orioles'] = ['balmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['royals'] = ['kcamlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['twins'] = ['minmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['mets'] = ['nynmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['cardinals'] = ['slnmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['giants'] = ['sfnmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['texas'] = ['texmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['los angeles'] = ['anamlb', 'lanmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['arizona'] = ['arimlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['chicago'] = ['chamlb','chnmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['cincinatti'] = ['cinmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['colorado'] = ['colmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['detroit'] = ['detmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['milwaukee'] = ['milmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['new york'] = ['nyamlb', 'nynmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['oakland'] = ['oakmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['philadelphia'] = ['phimlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['pittsburgh'] = ['pitmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['seattle'] = ['seamlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['tampa bay'] = ['tbamlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['toronto'] = ['tormlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['washington'] = ['wasmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['miami'] = ['miamlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['san diego'] = ['sdnmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['baltimore'] = ['balmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['kansas city'] = ['kcamlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['minnesota'] = ['minmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['st louis'] = ['stlmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['san francisco'] = ['sfnmlb']
DOMAIN_KNOWLEDGE['team_id_pitcher']['pitchers'] = list(set(reduce(lambda x, y: x + y, 
                                                           DOMAIN_KNOWLEDGE['team_id_pitcher'].values(), [])))
DOMAIN_KNOWLEDGE['team_id_batter']['batters'] = DOMAIN_KNOWLEDGE['team_id_pitcher']['pitchers']
DOMAIN_KNOWLEDGE['type']['strike'].append('S')
DOMAIN_KNOWLEDGE['type']['ball'].append('B')
DOMAIN_KNOWLEDGE['type']['in play'].append('X')
DOMAIN_KNOWLEDGE['pitch_type']['pitch type'] = list(X['pitch_type'].unique())
DOMAIN_KNOWLEDGE['pitch_type']['fastball'] = ['FA', 'FF', 'FT', 'FC', 'FS']
DOMAIN_KNOWLEDGE['pitch_type']['4-seam'] = ['FF']
DOMAIN_KNOWLEDGE['pitch_type']['four-seam'] = ['FF']
DOMAIN_KNOWLEDGE['pitch_type']['two-seam'] = ['FT']
DOMAIN_KNOWLEDGE['pitch_type']['2-seam'] = ['FT']
DOMAIN_KNOWLEDGE['pitch_type']['cutter'] = ['FC']
DOMAIN_KNOWLEDGE['pitch_type']['sinker'] = ['SI']
DOMAIN_KNOWLEDGE['pitch_type']['split'] = ['SF']
DOMAIN_KNOWLEDGE['pitch_type']['fingered'] = ['SF']
DOMAIN_KNOWLEDGE['pitch_type']['slider'] = ['SL']
DOMAIN_KNOWLEDGE['pitch_type']['changeup'] = ['CH']
DOMAIN_KNOWLEDGE['pitch_type']['curveball'] = ['CB', 'CU']
DOMAIN_KNOWLEDGE['pitch_type']['knuckleball'] = ['KC', 'KN']
DOMAIN_KNOWLEDGE['pitch_type']['knucklers'] = ['KC', 'KN']
DOMAIN_KNOWLEDGE['pitch_type']['eephus'] = ['EP']
DOMAIN_KNOWLEDGE['event']['groundout'] = ['Groundout']
DOMAIN_KNOWLEDGE['event']['strikeout'] = ['Strikeout']
DOMAIN_KNOWLEDGE['event']['homerun'] = ['Home Run']
DOMAIN_KNOWLEDGE['event']['walk'] = ['Walk']
DOMAIN_KNOWLEDGE['event']['single'] = ['Single']
DOMAIN_KNOWLEDGE['event']['double'] = ['Double']
DOMAIN_KNOWLEDGE['event']['triple'] = ['Triple']
DOMAIN_KNOWLEDGE['event']['lineout'] = ['Lineout']
DOMAIN_KNOWLEDGE['event']['flyout'] = ['Flyout']
DOMAIN_KNOWLEDGE['event']['pop-out'] = ['Pop Out']
DOMAIN_KNOWLEDGE['event']['bunt'] = ['Bunt Groundout', 'Sac Bunt', 'Bunt Pop Out']
DOMAIN_KNOWLEDGE['event']['field'] = ['Field Error']
DOMAIN_KNOWLEDGE['event']['error'] = ['Field Error']
DOMAIN_KNOWLEDGE['stand']['batters'] = ['L', 'R']
DOMAIN_KNOWLEDGE['stand']['lefty'] = ['L']
DOMAIN_KNOWLEDGE['stand']['left-handed'] = ['L']
DOMAIN_KNOWLEDGE['stand']['righty'] = ['R']
DOMAIN_KNOWLEDGE['stand']['right-handed'] = ['R']
DOMAIN_KNOWLEDGE['inning']['inning'] = list(X['inning'].unique())
DOMAIN_KNOWLEDGE['inning']['first'] = ['1']
DOMAIN_KNOWLEDGE['inning']['second'] = ['2']
DOMAIN_KNOWLEDGE['inning']['third'] = ['3']
DOMAIN_KNOWLEDGE['inning']['fourth'] = ['4']
DOMAIN_KNOWLEDGE['inning']['fifth'] = ['5']
DOMAIN_KNOWLEDGE['inning']['sixth'] = ['6']
DOMAIN_KNOWLEDGE['inning']['seventh'] = ['7']
DOMAIN_KNOWLEDGE['inning']['eighth'] = ['8']
DOMAIN_KNOWLEDGE['inning']['ninth'] = ['9']
DOMAIN_KNOWLEDGE['inning_side']['top'] = ['top']
DOMAIN_KNOWLEDGE['inning_side']['bottom'] = ['bottom']
DOMAIN_KNOWLEDGE['o']['outs'] = ['0', '1', '2']
DOMAIN_KNOWLEDGE['px']['x'] = ['is.numeric']
DOMAIN_KNOWLEDGE['px']['location'] = ['is.numeric']
DOMAIN_KNOWLEDGE['px']['left/right'] = ['is.numeric']
DOMAIN_KNOWLEDGE['px']['horizontal'] = ['is.numeric']
DOMAIN_KNOWLEDGE['pz']['z'] = ['is.numeric']
DOMAIN_KNOWLEDGE['pz']['location'] = ['is.numeric']
DOMAIN_KNOWLEDGE['pz']['pitch'] = ['is.numeric']
DOMAIN_KNOWLEDGE['pz']['height'] = ['is.numeric']
DOMAIN_KNOWLEDGE['start_speed']['start'] = ['is.numeric']
DOMAIN_KNOWLEDGE['start_speed']['pitch'] = ['is.numeric']
DOMAIN_KNOWLEDGE['start_speed']['ball'] = ['is.numeric']
DOMAIN_KNOWLEDGE['start_speed']['speed'] = ['is.numeric']
DOMAIN_KNOWLEDGE['start_speed']['velocity'] = ['is.numeric']
DOMAIN_KNOWLEDGE['end_speed']['end'] = ['is.numeric']
DOMAIN_KNOWLEDGE['end_speed']['pitch'] = ['is.numeric']
DOMAIN_KNOWLEDGE['end_speed']['ball'] = ['is.numeric']
DOMAIN_KNOWLEDGE['end_speed']['speed'] = ['is.numeric']
DOMAIN_KNOWLEDGE['end_speed']['velocity'] = ['is.numeric']
DOMAIN_KNOWLEDGE['pfx_x']['x'] = ['is.numeric']
DOMAIN_KNOWLEDGE['pfx_x']['movement'] = ['is.numeric']
DOMAIN_KNOWLEDGE['pfx_x']['left/right'] = ['is.numeric']
DOMAIN_KNOWLEDGE['pfx_x']['horizontal'] = ['is.numeric']
DOMAIN_KNOWLEDGE['pfx_x']['z'] = ['is.numeric']
DOMAIN_KNOWLEDGE['pfx_z']['movement'] = ['is.numeric']
DOMAIN_KNOWLEDGE['pfx_z']['vertical'] = ['is.numeric']
DOMAIN_KNOWLEDGE['ax']['x'] = ['is.numeric']
DOMAIN_KNOWLEDGE['ax']['acceleration'] = ['is.numeric']
DOMAIN_KNOWLEDGE['ax']['left/right'] = ['is.numeric']
DOMAIN_KNOWLEDGE['ax']['horizontal'] = ['is.numeric']
DOMAIN_KNOWLEDGE['az']['z'] = ['is.numeric']
DOMAIN_KNOWLEDGE['az']['acceleration'] = ['is.numeric']
DOMAIN_KNOWLEDGE['az']['vertical'] = ['is.numeric']
DOMAIN_KNOWLEDGE['break_angle']['break'] = ['is.numeric']
DOMAIN_KNOWLEDGE['break_angle']['angle'] = ['is.numeric']
DOMAIN_KNOWLEDGE['break_length']['break'] = ['is.numeric']
DOMAIN_KNOWLEDGE['break_length']['length'] = ['is.numeric']
DOMAIN_KNOWLEDGE['type_confidence']['type'] = ['is.numeric']
DOMAIN_KNOWLEDGE['type_confidence']['confidence'] = ['is.numeric']
name_ids = ['batter_name', 'pitcher_name']
IDENTIFIERS = defaultdict(list)
#for name_id in name_ids:
    #IDENTIFIERS[name_id] = X[name_id]

CORE = { # CORE initialized to hold uninformative modifiers for each observation in the dataset
    'pitch': None,
    'pitches': None,
    'atbats': None
}

class Recommendation:
    
    '''Recommendation Object that Holds Query Metadata'''
    
    def __init__(self, qry = None):
        self.data = {}
        if qry is not None:
            self.data[qry] = RecommendationItem()
        
class RecommendationItem:
    
    '''Recommendation Item for Automated Query Suggestion'''
    
    def __init__(self, feature = None, index = []):
        self.index_hash = {}
        if feature is not None:
            self.index_hash[feature] = index

added = [] # list of features added to dataframe during analysis
COOCCURENCE_HASH = defaultdict(lambda : defaultdict(float)) # cooccurence hash over features
MOST_RECENT_QUERY = set() # holds set of features in most recent query
EXP_DECAY = defaultdict(int) # hash of each feature on exponentially decaying weight as distance since last fetched increases
ITEMSETS = [] # frequent itemsets
RECOMMENDATION = Recommendation() # global RECOMMENDATION object
RECOMMENDATIONS = defaultdict(list)

CONF_THRESHOLD = 0.9 # confidence that a token must match an entry in DOMAIN_KNOWLEDGE before fetching that feature directly
NAME_THRESHOLD = 0.9 # confidence that a token must match an entry in IDENTIFIERS before being labeled an player id
RELEVANCE_FEEDBACK_THRESHOLD = 0.5 # threshold normalized on [0, 1] to label if a feature is relevant after relevance feedback
ASSOC_MIN_SUPPORT = 2 # frequent itemsets support threshold
ASSOC_MIN_CONFIDENCE = 0.5 # frequent itemsets confidence threshold
DECAY = 2e-4 # exponential decay cost