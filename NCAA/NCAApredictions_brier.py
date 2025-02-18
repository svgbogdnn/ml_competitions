import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
import category_encoders as ce
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from catboost import CatBoostRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import brier_score_loss
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

import warnings
import time
start = time.time()

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

''' DATA CONNECTION '''

#                                       male info
MConferenceTourneyGames = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/MConferenceTourneyGames.csv')
MGameCities = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/MGameCities.csv')
MMasseyOrdinals = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/MMasseyOrdinals.csv')
MNCAATourneyCompactResults = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/MNCAATourneyCompactResults.csv')
MNCAATourneyDetailedResults = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/MNCAATourneyDetailedResults.csv')
MNCAATourneySeedRoundSlots = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/MNCAATourneySeedRoundSlots.csv')
MNCAATourneySeeds = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/MNCAATourneySeeds.csv')
MNCAATourneySlots = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/MNCAATourneySlots.csv')
MRegularSeasonCompactResults = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/MRegularSeasonCompactResults.csv')
MRegularSeasonDetailedResults = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/MRegularSeasonDetailedResults.csv')
MSeasons = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/MSeasons.csv')
MSecondaryTourneyCompactResults = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/MSecondaryTourneyCompactResults.csv')
MSecondaryTourneyTeams = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/MSecondaryTourneyTeams.csv')
MTeamCoaches = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/MTeamCoaches.csv')
MTeamConferences = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/MTeamConferences.csv')
MTeamSpellings = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/MTeamSpellings.csv', encoding='latin1')
MTeams = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/MTeams.csv')
#                                        female info
WConferenceTourneyGames = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/WConferenceTourneyGames.csv')
WGameCities = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/WGameCities.csv')
WNCAATourneyCompactResults = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/WNCAATourneyCompactResults.csv')
WNCAATourneyDetailedResults = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/WNCAATourneyDetailedResults.csv')
WNCAATourneySeeds = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/WNCAATourneySeeds.csv')
WNCAATourneySlots = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/WNCAATourneySlots.csv')
WRegularSeasonCompactResults = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/WRegularSeasonCompactResults.csv')
WRegularSeasonDetailedResults = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/WRegularSeasonDetailedResults.csv')
WSeasons = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/WSeasons.csv')
WSecondaryTourneyCompactResults = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/WSecondaryTourneyCompactResults.csv')
WSecondaryTourneyTeams = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/WSecondaryTourneyTeams.csv')
WTeamConferences = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/WTeamConferences.csv')
WTeamSpellings = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/WTeamSpellings.csv', encoding='latin1')
WTeams = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/WTeams.csv')
#                                           both info
SampleSubmissionStage1 = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/SampleSubmissionStage1.csv')
SeedBenchmarkStage1 = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/SeedBenchmarkStage1.csv')
Cities = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/Cities.csv') # 2/3 cat
Conferences = pd.read_csv('D:/Train Data/march-machine-learning-mania-2025/Conferences.csv') # for example full cat

# check for omissions
# print(Cities.isnull().sum())
# print(Conferences.isnull().sum())
# print(MConferenceTourneyGames.isnull().sum())
# print(MGameCities.isnull().sum())
# print(MMasseyOrdinals.isnull().sum())
# print(MNCAATourneyCompactResults.isnull().sum())
# print(MNCAATourneyDetailedResults.isnull().sum())
# print(MNCAATourneySeedRoundSlots.isnull().sum())
# print(MNCAATourneySeeds.isnull().sum())
# print(MNCAATourneySlots.isnull().sum())
# print(MRegularSeasonCompactResults.isnull().sum())
# print(MRegularSeasonDetailedResults.isnull().sum())
# print(MSeasons.isnull().sum())
# print(MSecondaryTourneyCompactResults.isnull().sum())
# print(MSecondaryTourneyTeams.isnull().sum())
# print(MTeamCoaches.isnull().sum())
# print(MTeamConferences.isnull().sum())
# print(MTeamSpellings.isnull().sum())
# print(MTeams.isnull().sum())
# print(SampleSubmissionStage1.isnull().sum())
# print(SeedBenchmarkStage1.isnull().sum())
# print(WConferenceTourneyGames.isnull().sum())
# print(WGameCities.isnull().sum())
# print(WNCAATourneyCompactResults.isnull().sum())
# print(WNCAATourneyDetailedResults.isnull().sum())
# print(WNCAATourneySeeds.isnull().sum())
# print(WNCAATourneySlots.isnull().sum())
# print(WRegularSeasonCompactResults.isnull().sum())
# print(WRegularSeasonDetailedResults.isnull().sum())
# print(WSeasons.isnull().sum())
# print(WSecondaryTourneyCompactResults.isnull().sum())
# print(WSecondaryTourneyTeams.isnull().sum())
# print(WTeamConferences.isnull().sum())
# print(WTeamSpellings.isnull().sum())
# print(WTeams.isnull().sum())

''' Relation of Numerical?Categorical Data '''
# because it need later, to do encoding (view which df we dont need to encode) and just for overall
datasets = {
    "Cities": Cities,
    "Conferences": Conferences,
    "MConferenceTourneyGames": MConferenceTourneyGames,
    "MGameCities": MGameCities,
    "MMasseyOrdinals": MMasseyOrdinals,
    "MNCAATourneyCompactResults": MNCAATourneyCompactResults,
    "MNCAATourneyDetailedResults": MNCAATourneyDetailedResults,
    "MNCAATourneySeedRoundSlots": MNCAATourneySeedRoundSlots,
    "MNCAATourneySeeds": MNCAATourneySeeds,
    "MNCAATourneySlots": MNCAATourneySlots,
    "MRegularSeasonCompactResults": MRegularSeasonCompactResults,
    "MRegularSeasonDetailedResults": MRegularSeasonDetailedResults,
    "MSeasons": MSeasons,
    "MSecondaryTourneyCompactResults": MSecondaryTourneyCompactResults,
    "MSecondaryTourneyTeams": MSecondaryTourneyTeams,
    "MTeamCoaches": MTeamCoaches,
    "MTeamConferences": MTeamConferences,
    "MTeamSpellings": MTeamSpellings,
    "MTeams": MTeams,
    "SampleSubmissionStage1": SampleSubmissionStage1,
    "SeedBenchmarkStage1": SeedBenchmarkStage1,
    "WConferenceTourneyGames": WConferenceTourneyGames,
    "WGameCities": WGameCities,
    "WNCAATourneyCompactResults": WNCAATourneyCompactResults,
    "WNCAATourneyDetailedResults": WNCAATourneyDetailedResults,
    "WNCAATourneySeeds": WNCAATourneySeeds,
    "WNCAATourneySlots": WNCAATourneySlots,
    "WRegularSeasonCompactResults": WRegularSeasonCompactResults,
    "WRegularSeasonDetailedResults": WRegularSeasonDetailedResults,
    "WSeasons": WSeasons,
    "WSecondaryTourneyCompactResults": WSecondaryTourneyCompactResults,
    "WSecondaryTourneyTeams": WSecondaryTourneyTeams,
    "WTeamConferences": WTeamConferences,
    "WTeamSpellings": WTeamSpellings,
    "WTeams": WTeams
}

def calculate_percentage(dataframe):
    num_cols = dataframe.select_dtypes(include=['number']).shape[1]
    cat_cols = dataframe.select_dtypes(include=['object']).shape[1]
    total_cols = num_cols + cat_cols
    num_percentage = (num_cols / total_cols) * 100 if total_cols else 0
    cat_percentage = (cat_cols / total_cols) * 100 if total_cols else 0
    return num_percentage, cat_percentage
# for name, dataset in datasets.items():
#     num_percentage, cat_percentage = calculate_percentage(dataset)
#     print(f"{name} - Numeric ~ {num_percentage:.2f}% | Categorical ~ {cat_percentage:.2f}%")
'''
Cities - Numeric ~ 33.33% | Categorical ~ 66.67%
Conferences - Numeric ~ 0.00% | Categorical ~ 100.00%
MConferenceTourneyGames - Numeric ~ 80.00% | Categorical ~ 20.00%
MGameCities - Numeric ~ 83.33% | Categorical ~ 16.67%
MMasseyOrdinals - Numeric ~ 80.00% | Categorical ~ 20.00%
MNCAATourneyCompactResults - Numeric ~ 87.50% | Categorical ~ 12.50%
MNCAATourneyDetailedResults - Numeric ~ 97.06% | Categorical ~ 2.94%
MNCAATourneySeedRoundSlots - Numeric ~ 60.00% | Categorical ~ 40.00%
MNCAATourneySeeds - Numeric ~ 66.67% | Categorical ~ 33.33%
MNCAATourneySlots - Numeric ~ 25.00% | Categorical ~ 75.00%
MRegularSeasonCompactResults - Numeric ~ 87.50% | Categorical ~ 12.50%
MRegularSeasonDetailedResults - Numeric ~ 97.06% | Categorical ~ 2.94%
MSeasons - Numeric ~ 16.67% | Categorical ~ 83.33%
MSecondaryTourneyCompactResults - Numeric ~ 77.78% | Categorical ~ 22.22%
MSecondaryTourneyTeams - Numeric ~ 66.67% | Categorical ~ 33.33%
MTeamCoaches - Numeric ~ 80.00% | Categorical ~ 20.00%
MTeamConferences - Numeric ~ 66.67% | Categorical ~ 33.33%
MTeamSpellings - Numeric ~ 50.00% | Categorical ~ 50.00%
MTeams - Numeric ~ 75.00% | Categorical ~ 25.00%
SampleSubmissionStage1 - Numeric ~ 50.00% | Categorical ~ 50.00%
SeedBenchmarkStage1 - Numeric ~ 50.00% | Categorical ~ 50.00%
WConferenceTourneyGames - Numeric ~ 80.00% | Categorical ~ 20.00%
WGameCities - Numeric ~ 83.33% | Categorical ~ 16.67%
WNCAATourneyCompactResults - Numeric ~ 87.50% | Categorical ~ 12.50%
WNCAATourneyDetailedResults - Numeric ~ 97.06% | Categorical ~ 2.94%
WNCAATourneySeeds - Numeric ~ 66.67% | Categorical ~ 33.33%
WNCAATourneySlots - Numeric ~ 25.00% | Categorical ~ 75.00%
WRegularSeasonCompactResults - Numeric ~ 87.50% | Categorical ~ 12.50%
WRegularSeasonDetailedResults - Numeric ~ 97.06% | Categorical ~ 2.94%
WSeasons - Numeric ~ 16.67% | Categorical ~ 83.33%
WSecondaryTourneyCompactResults - Numeric ~ 77.78% | Categorical ~ 22.22%
WSecondaryTourneyTeams - Numeric ~ 66.67% | Categorical ~ 33.33%
WTeamConferences - Numeric ~ 66.67% | Categorical ~ 33.33%
WTeamSpellings - Numeric ~ 50.00% | Categorical ~ 50.00%
WTeams - Numeric ~ 50.00% | Categorical ~ 50.00%
'''

# division into categorical & numerical values
num_cols_Cities = Cities.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_Cities = Cities.select_dtypes(include=['object']).columns.tolist()
num_cols_MTeams = MTeams.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_MTeams = MTeams.select_dtypes(include=['object']).columns.tolist()
num_cols_Conferences = Conferences.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_Conferences = Conferences.select_dtypes(include=['object']).columns.tolist()
num_cols_MConferenceTourneyGames = MConferenceTourneyGames.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_MConferenceTourneyGames = MConferenceTourneyGames.select_dtypes(include=['object']).columns.tolist()
num_cols_MGameCities = MGameCities.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_MGameCities = MGameCities.select_dtypes(include=['object']).columns.tolist()
num_cols_MMasseyOrdinals = MMasseyOrdinals.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_MMasseyOrdinals = MMasseyOrdinals.select_dtypes(include=['object']).columns.tolist()
num_cols_MNCAATourneyCompactResults = MNCAATourneyCompactResults.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_MNCAATourneyCompactResults = MNCAATourneyCompactResults.select_dtypes(include=['object']).columns.tolist()
num_cols_MNCAATourneyDetailedResults = MNCAATourneyDetailedResults.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_MNCAATourneyDetailedResults = MNCAATourneyDetailedResults.select_dtypes(include=['object']).columns.tolist()
num_cols_MNCAATourneySeedRoundSlots = MNCAATourneySeedRoundSlots.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_MNCAATourneySeedRoundSlots = MNCAATourneySeedRoundSlots.select_dtypes(include=['object']).columns.tolist()
num_cols_MNCAATourneySeeds = MNCAATourneySeeds.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_MNCAATourneySeeds = MNCAATourneySeeds.select_dtypes(include=['object']).columns.tolist()
num_cols_MNCAATourneySlots = MNCAATourneySlots.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_MNCAATourneySlots = MNCAATourneySlots.select_dtypes(include=['object']).columns.tolist()
num_cols_MRegularSeasonCompactResults = MRegularSeasonCompactResults.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_MRegularSeasonCompactResults = MRegularSeasonCompactResults.select_dtypes(include=['object']).columns.tolist()
num_cols_MRegularSeasonDetailedResults = MRegularSeasonDetailedResults.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_MRegularSeasonDetailedResults = MRegularSeasonDetailedResults.select_dtypes(include=['object']).columns.tolist()
num_cols_MSeasons = MSeasons.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_MSeasons = MSeasons.select_dtypes(include=['object']).columns.tolist()
num_cols_MSecondaryTourneyCompactResults = MSecondaryTourneyCompactResults.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_MSecondaryTourneyCompactResults = MSecondaryTourneyCompactResults.select_dtypes(include=['object']).columns.tolist()
num_cols_MSecondaryTourneyTeams = MSecondaryTourneyTeams.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_MSecondaryTourneyTeams = MSecondaryTourneyTeams.select_dtypes(include=['object']).columns.tolist()
num_cols_MTeamCoaches = MTeamCoaches.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_MTeamCoaches = MTeamCoaches.select_dtypes(include=['object']).columns.tolist()
num_cols_MTeamConferences = MTeamConferences.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_MTeamConferences = MTeamConferences.select_dtypes(include=['object']).columns.tolist()
num_cols_MTeamSpellings = MTeamSpellings.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_MTeamSpellings = MTeamSpellings.select_dtypes(include=['object']).columns.tolist()
num_cols_MTeams = MTeams.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_MTeams = MTeams.select_dtypes(include=['object']).columns.tolist()
num_cols_SampleSubmissionStage1 = SampleSubmissionStage1.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_SampleSubmissionStage1 = SampleSubmissionStage1.select_dtypes(include=['object']).columns.tolist()
num_cols_SeedBenchmarkStage1 = SeedBenchmarkStage1.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_SeedBenchmarkStage1 = SeedBenchmarkStage1.select_dtypes(include=['object']).columns.tolist()
num_cols_WConferenceTourneyGames = WConferenceTourneyGames.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_WConferenceTourneyGames = WConferenceTourneyGames.select_dtypes(include=['object']).columns.tolist()
num_cols_WGameCities = WGameCities.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_WGameCities = WGameCities.select_dtypes(include=['object']).columns.tolist()
num_cols_WNCAATourneyCompactResults = WNCAATourneyCompactResults.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_WNCAATourneyCompactResults = WNCAATourneyCompactResults.select_dtypes(include=['object']).columns.tolist()
num_cols_WNCAATourneyDetailedResults = WNCAATourneyDetailedResults.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_WNCAATourneyDetailedResults = WNCAATourneyDetailedResults.select_dtypes(include=['object']).columns.tolist()
num_cols_WNCAATourneySeeds = WNCAATourneySeeds.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_WNCAATourneySeeds = WNCAATourneySeeds.select_dtypes(include=['object']).columns.tolist()
num_cols_WNCAATourneySlots = WNCAATourneySlots.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_WNCAATourneySlots = WNCAATourneySlots.select_dtypes(include=['object']).columns.tolist()
num_cols_WRegularSeasonCompactResults = WRegularSeasonCompactResults.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_WRegularSeasonCompactResults = WRegularSeasonCompactResults.select_dtypes(include=['object']).columns.tolist()
num_cols_WRegularSeasonDetailedResults = WRegularSeasonDetailedResults.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_WRegularSeasonDetailedResults = WRegularSeasonDetailedResults.select_dtypes(include=['object']).columns.tolist()
num_cols_WSeasons = WSeasons.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_WSeasons = WSeasons.select_dtypes(include=['object']).columns.tolist()
num_cols_WSecondaryTourneyCompactResults = WSecondaryTourneyCompactResults.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_WSecondaryTourneyCompactResults = WSecondaryTourneyCompactResults.select_dtypes(include=['object']).columns.tolist()
num_cols_WSecondaryTourneyTeams = WSecondaryTourneyTeams.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_WSecondaryTourneyTeams = WSecondaryTourneyTeams.select_dtypes(include=['object']).columns.tolist()
num_cols_WTeamConferences = WTeamConferences.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_WTeamConferences = WTeamConferences.select_dtypes(include=['object']).columns.tolist()
num_cols_WTeamSpellings = WTeamSpellings.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_WTeamSpellings = WTeamSpellings.select_dtypes(include=['object']).columns.tolist()
num_cols_WTeams = WTeams.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_cols_WTeams = WTeams.select_dtypes(include=['object']).columns.tolist()

# fill categorical & numerical omissions
num_imputer = SimpleImputer(strategy='mean')
cat_imputer = SimpleImputer(strategy='constant', fill_value='Unknown')

MTeamSpellings[num_cols_MTeamSpellings] = num_imputer.fit_transform(MTeamSpellings[num_cols_MTeamSpellings])
MTeamSpellings[cat_cols_MTeamSpellings] = cat_imputer.fit_transform(MTeamSpellings[cat_cols_MTeamSpellings])

Cities[num_cols_Cities] = num_imputer.fit_transform(Cities[num_cols_Cities])
Cities[cat_cols_Cities] = cat_imputer.fit_transform(Cities[cat_cols_Cities])

# Conferences[num_cols_Conferences] = num_imputer.fit_transform(Conferences[num_cols_Conferences]) # because empty
Conferences[cat_cols_Conferences] = cat_imputer.fit_transform(Conferences[cat_cols_Conferences])

MConferenceTourneyGames[num_cols_MConferenceTourneyGames] = num_imputer.fit_transform(MConferenceTourneyGames[num_cols_MConferenceTourneyGames])
MConferenceTourneyGames[cat_cols_MConferenceTourneyGames] = cat_imputer.fit_transform(MConferenceTourneyGames[cat_cols_MConferenceTourneyGames])

MGameCities[num_cols_MGameCities] = num_imputer.fit_transform(MGameCities[num_cols_MGameCities])
MGameCities[cat_cols_MGameCities] = cat_imputer.fit_transform(MGameCities[cat_cols_MGameCities])

MMasseyOrdinals[num_cols_MMasseyOrdinals] = num_imputer.fit_transform(MMasseyOrdinals[num_cols_MMasseyOrdinals])
MMasseyOrdinals[cat_cols_MMasseyOrdinals] = cat_imputer.fit_transform(MMasseyOrdinals[cat_cols_MMasseyOrdinals])

MNCAATourneyCompactResults[num_cols_MNCAATourneyCompactResults] = num_imputer.fit_transform(MNCAATourneyCompactResults[num_cols_MNCAATourneyCompactResults])
MNCAATourneyCompactResults[cat_cols_MNCAATourneyCompactResults] = cat_imputer.fit_transform(MNCAATourneyCompactResults[cat_cols_MNCAATourneyCompactResults])

MNCAATourneyDetailedResults[num_cols_MNCAATourneyDetailedResults] = num_imputer.fit_transform(MNCAATourneyDetailedResults[num_cols_MNCAATourneyDetailedResults])
MNCAATourneyDetailedResults[cat_cols_MNCAATourneyDetailedResults] = cat_imputer.fit_transform(MNCAATourneyDetailedResults[cat_cols_MNCAATourneyDetailedResults])

MNCAATourneySeedRoundSlots[num_cols_MNCAATourneySeedRoundSlots] = num_imputer.fit_transform(MNCAATourneySeedRoundSlots[num_cols_MNCAATourneySeedRoundSlots])
MNCAATourneySeedRoundSlots[cat_cols_MNCAATourneySeedRoundSlots] = cat_imputer.fit_transform(MNCAATourneySeedRoundSlots[cat_cols_MNCAATourneySeedRoundSlots])

MNCAATourneySeeds[num_cols_MNCAATourneySeeds] = num_imputer.fit_transform(MNCAATourneySeeds[num_cols_MNCAATourneySeeds])
MNCAATourneySeeds[cat_cols_MNCAATourneySeeds] = cat_imputer.fit_transform(MNCAATourneySeeds[cat_cols_MNCAATourneySeeds])

MNCAATourneySlots[num_cols_MNCAATourneySlots] = num_imputer.fit_transform(MNCAATourneySlots[num_cols_MNCAATourneySlots])
MNCAATourneySlots[cat_cols_MNCAATourneySlots] = cat_imputer.fit_transform(MNCAATourneySlots[cat_cols_MNCAATourneySlots])

MRegularSeasonCompactResults[num_cols_MRegularSeasonCompactResults] = num_imputer.fit_transform(MRegularSeasonCompactResults[num_cols_MRegularSeasonCompactResults])
MRegularSeasonCompactResults[cat_cols_MRegularSeasonCompactResults] = cat_imputer.fit_transform(MRegularSeasonCompactResults[cat_cols_MRegularSeasonCompactResults])

MRegularSeasonDetailedResults[num_cols_MRegularSeasonDetailedResults] = num_imputer.fit_transform(MRegularSeasonDetailedResults[num_cols_MRegularSeasonDetailedResults])
MRegularSeasonDetailedResults[cat_cols_MRegularSeasonDetailedResults] = cat_imputer.fit_transform(MRegularSeasonDetailedResults[cat_cols_MRegularSeasonDetailedResults])

MSeasons[num_cols_MSeasons] = num_imputer.fit_transform(MSeasons[num_cols_MSeasons])
MSeasons[cat_cols_MSeasons] = cat_imputer.fit_transform(MSeasons[cat_cols_MSeasons])

MSecondaryTourneyCompactResults[num_cols_MSecondaryTourneyCompactResults] = num_imputer.fit_transform(MSecondaryTourneyCompactResults[num_cols_MSecondaryTourneyCompactResults])
MSecondaryTourneyCompactResults[cat_cols_MSecondaryTourneyCompactResults] = cat_imputer.fit_transform(MSecondaryTourneyCompactResults[cat_cols_MSecondaryTourneyCompactResults])

MSecondaryTourneyTeams[num_cols_MSecondaryTourneyTeams] = num_imputer.fit_transform(MSecondaryTourneyTeams[num_cols_MSecondaryTourneyTeams])
MSecondaryTourneyTeams[cat_cols_MSecondaryTourneyTeams] = cat_imputer.fit_transform(MSecondaryTourneyTeams[cat_cols_MSecondaryTourneyTeams])

MTeamCoaches[num_cols_MTeamCoaches] = num_imputer.fit_transform(MTeamCoaches[num_cols_MTeamCoaches])
MTeamCoaches[cat_cols_MTeamCoaches] = cat_imputer.fit_transform(MTeamCoaches[cat_cols_MTeamCoaches])

MTeamConferences[num_cols_MTeamConferences] = num_imputer.fit_transform(MTeamConferences[num_cols_MTeamConferences])
MTeamConferences[cat_cols_MTeamConferences] = cat_imputer.fit_transform(MTeamConferences[cat_cols_MTeamConferences])

MTeamSpellings[num_cols_MTeamSpellings] = num_imputer.fit_transform(MTeamSpellings[num_cols_MTeamSpellings])
MTeamSpellings[cat_cols_MTeamSpellings] = cat_imputer.fit_transform(MTeamSpellings[cat_cols_MTeamSpellings])

MTeams[num_cols_MTeams] = num_imputer.fit_transform(MTeams[num_cols_MTeams])
MTeams[cat_cols_MTeams] = cat_imputer.fit_transform(MTeams[cat_cols_MTeams])

SampleSubmissionStage1[num_cols_SampleSubmissionStage1] = num_imputer.fit_transform(SampleSubmissionStage1[num_cols_SampleSubmissionStage1])
SampleSubmissionStage1[cat_cols_SampleSubmissionStage1] = cat_imputer.fit_transform(SampleSubmissionStage1[cat_cols_SampleSubmissionStage1])

SeedBenchmarkStage1[num_cols_SeedBenchmarkStage1] = num_imputer.fit_transform(SeedBenchmarkStage1[num_cols_SeedBenchmarkStage1])
SeedBenchmarkStage1[cat_cols_SeedBenchmarkStage1] = cat_imputer.fit_transform(SeedBenchmarkStage1[cat_cols_SeedBenchmarkStage1])

WConferenceTourneyGames[num_cols_WConferenceTourneyGames] = num_imputer.fit_transform(WConferenceTourneyGames[num_cols_WConferenceTourneyGames])
WConferenceTourneyGames[cat_cols_WConferenceTourneyGames] = cat_imputer.fit_transform(WConferenceTourneyGames[cat_cols_WConferenceTourneyGames])

WGameCities[num_cols_WGameCities] = num_imputer.fit_transform(WGameCities[num_cols_WGameCities])
WGameCities[cat_cols_WGameCities] = cat_imputer.fit_transform(WGameCities[cat_cols_WGameCities])

WNCAATourneyCompactResults[num_cols_WNCAATourneyCompactResults] = num_imputer.fit_transform(WNCAATourneyCompactResults[num_cols_WNCAATourneyCompactResults])
WNCAATourneyCompactResults[cat_cols_WNCAATourneyCompactResults] = cat_imputer.fit_transform(WNCAATourneyCompactResults[cat_cols_WNCAATourneyCompactResults])

WNCAATourneyDetailedResults[num_cols_WNCAATourneyDetailedResults] = num_imputer.fit_transform(WNCAATourneyDetailedResults[num_cols_WNCAATourneyDetailedResults])
WNCAATourneyDetailedResults[cat_cols_WNCAATourneyDetailedResults] = cat_imputer.fit_transform(WNCAATourneyDetailedResults[cat_cols_WNCAATourneyDetailedResults])

WNCAATourneySeeds[num_cols_WNCAATourneySeeds] = num_imputer.fit_transform(WNCAATourneySeeds[num_cols_WNCAATourneySeeds])
WNCAATourneySeeds[cat_cols_WNCAATourneySeeds] = cat_imputer.fit_transform(WNCAATourneySeeds[cat_cols_WNCAATourneySeeds])

WNCAATourneySlots[num_cols_WNCAATourneySlots] = num_imputer.fit_transform(WNCAATourneySlots[num_cols_WNCAATourneySlots])
WNCAATourneySlots[cat_cols_WNCAATourneySlots] = cat_imputer.fit_transform(WNCAATourneySlots[cat_cols_WNCAATourneySlots])

WRegularSeasonCompactResults[num_cols_WRegularSeasonCompactResults] = num_imputer.fit_transform(WRegularSeasonCompactResults[num_cols_WRegularSeasonCompactResults])
WRegularSeasonCompactResults[cat_cols_WRegularSeasonCompactResults] = cat_imputer.fit_transform(WRegularSeasonCompactResults[cat_cols_WRegularSeasonCompactResults])

WRegularSeasonDetailedResults[num_cols_WRegularSeasonDetailedResults] = num_imputer.fit_transform(WRegularSeasonDetailedResults[num_cols_WRegularSeasonDetailedResults])
WRegularSeasonDetailedResults[cat_cols_WRegularSeasonDetailedResults] = cat_imputer.fit_transform(WRegularSeasonDetailedResults[cat_cols_WRegularSeasonDetailedResults])

WSeasons[num_cols_WSeasons] = num_imputer.fit_transform(WSeasons[num_cols_WSeasons])
WSeasons[cat_cols_WSeasons] = cat_imputer.fit_transform(WSeasons[cat_cols_WSeasons])

WSecondaryTourneyCompactResults[num_cols_WSecondaryTourneyCompactResults] = num_imputer.fit_transform(WSecondaryTourneyCompactResults[num_cols_WSecondaryTourneyCompactResults])
WSecondaryTourneyCompactResults[cat_cols_WSecondaryTourneyCompactResults] = cat_imputer.fit_transform(WSecondaryTourneyCompactResults[cat_cols_WSecondaryTourneyCompactResults])

WSecondaryTourneyTeams[num_cols_WSecondaryTourneyTeams] = num_imputer.fit_transform(WSecondaryTourneyTeams[num_cols_WSecondaryTourneyTeams])
WSecondaryTourneyTeams[cat_cols_WSecondaryTourneyTeams] = cat_imputer.fit_transform(WSecondaryTourneyTeams[cat_cols_WSecondaryTourneyTeams])

WTeamConferences[num_cols_WTeamConferences] = num_imputer.fit_transform(WTeamConferences[num_cols_WTeamConferences])
WTeamConferences[cat_cols_WTeamConferences] = cat_imputer.fit_transform(WTeamConferences[cat_cols_WTeamConferences])

WTeamSpellings[num_cols_WTeamSpellings] = num_imputer.fit_transform(WTeamSpellings[num_cols_WTeamSpellings])
WTeamSpellings[cat_cols_WTeamSpellings] = cat_imputer.fit_transform(WTeamSpellings[cat_cols_WTeamSpellings])

WTeams[num_cols_WTeams] = num_imputer.fit_transform(WTeams[num_cols_WTeams])
WTeams[cat_cols_WTeams] = cat_imputer.fit_transform(WTeams[cat_cols_WTeams])
''' In fact, in these datasets NO emissions at all, idk how, but this true'''

''' ENCODING one-hot '''
onehot_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

encoded_onehot = onehot_encoder.fit_transform(Conferences[cat_cols_Conferences])
encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_Conferences), index=Conferences.index)
Conferences = pd.concat([Conferences[num_cols_Conferences], encoded_onehot_df], axis=1)

encoded_onehot = onehot_encoder.fit_transform(Cities[cat_cols_Cities])
encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_Cities), index=Cities.index)
Cities = pd.concat([Cities[num_cols_Cities], encoded_onehot_df], axis=1)

encoded_onehot = onehot_encoder.fit_transform(MConferenceTourneyGames[cat_cols_MConferenceTourneyGames])
encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_MConferenceTourneyGames), index=MConferenceTourneyGames.index)
MConferenceTourneyGames = pd.concat([MConferenceTourneyGames[num_cols_MConferenceTourneyGames], encoded_onehot_df], axis=1)

encoded_onehot = onehot_encoder.fit_transform(MGameCities[cat_cols_MGameCities])
encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_MGameCities), index=MGameCities.index)
MGameCities = pd.concat([MGameCities[num_cols_MGameCities], encoded_onehot_df], axis=1)

encoded_onehot = onehot_encoder.fit_transform(MMasseyOrdinals[cat_cols_MMasseyOrdinals])
encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_MMasseyOrdinals), index=MMasseyOrdinals.index)
MMasseyOrdinals = pd.concat([MMasseyOrdinals[num_cols_MMasseyOrdinals], encoded_onehot_df], axis=1)

encoded_onehot = onehot_encoder.fit_transform(MNCAATourneyCompactResults[cat_cols_MNCAATourneyCompactResults])
encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_MNCAATourneyCompactResults), index=MNCAATourneyCompactResults.index)
MNCAATourneyCompactResults = pd.concat([MNCAATourneyCompactResults[num_cols_MNCAATourneyCompactResults], encoded_onehot_df], axis=1)

encoded_onehot = onehot_encoder.fit_transform(MNCAATourneyDetailedResults[cat_cols_MNCAATourneyDetailedResults])
encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_MNCAATourneyDetailedResults), index=MNCAATourneyDetailedResults.index)
MNCAATourneyDetailedResults = pd.concat([MNCAATourneyDetailedResults[num_cols_MNCAATourneyDetailedResults], encoded_onehot_df], axis=1)

encoded_onehot = onehot_encoder.fit_transform(MNCAATourneySeedRoundSlots[cat_cols_MNCAATourneySeedRoundSlots])
encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_MNCAATourneySeedRoundSlots), index=MNCAATourneySeedRoundSlots.index)
MNCAATourneySeedRoundSlots = pd.concat([MNCAATourneySeedRoundSlots[num_cols_MNCAATourneySeedRoundSlots], encoded_onehot_df], axis=1)

encoded_onehot = onehot_encoder.fit_transform(MNCAATourneySeeds[cat_cols_MNCAATourneySeeds])
encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_MNCAATourneySeeds), index=MNCAATourneySeeds.index)
MNCAATourneySeeds = pd.concat([MNCAATourneySeeds[num_cols_MNCAATourneySeeds], encoded_onehot_df], axis=1)

encoded_onehot = onehot_encoder.fit_transform(MNCAATourneySlots[cat_cols_MNCAATourneySlots])
encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_MNCAATourneySlots), index=MNCAATourneySlots.index)
MNCAATourneySlots = pd.concat([MNCAATourneySlots[num_cols_MNCAATourneySlots], encoded_onehot_df], axis=1)

encoded_onehot = onehot_encoder.fit_transform(MRegularSeasonCompactResults[cat_cols_MRegularSeasonCompactResults])
encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_MRegularSeasonCompactResults), index=MRegularSeasonCompactResults.index)
MRegularSeasonCompactResults = pd.concat([MRegularSeasonCompactResults[num_cols_MRegularSeasonCompactResults], encoded_onehot_df], axis=1)

encoded_onehot = onehot_encoder.fit_transform(MRegularSeasonDetailedResults[cat_cols_MRegularSeasonDetailedResults])
encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_MRegularSeasonDetailedResults), index=MRegularSeasonDetailedResults.index)
MRegularSeasonDetailedResults = pd.concat([MRegularSeasonDetailedResults[num_cols_MRegularSeasonDetailedResults], encoded_onehot_df], axis=1)

encoded_onehot = onehot_encoder.fit_transform(MSeasons[cat_cols_MSeasons])
encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_MSeasons), index=MSeasons.index)
MSeasons = pd.concat([MSeasons[num_cols_MSeasons], encoded_onehot_df], axis=1)

encoded_onehot = onehot_encoder.fit_transform(MSecondaryTourneyCompactResults[cat_cols_MSecondaryTourneyCompactResults])
encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_MSecondaryTourneyCompactResults), index=MSecondaryTourneyCompactResults.index)
MSecondaryTourneyCompactResults = pd.concat([MSecondaryTourneyCompactResults[num_cols_MSecondaryTourneyCompactResults], encoded_onehot_df], axis=1)

encoded_onehot = onehot_encoder.fit_transform(MSecondaryTourneyTeams[cat_cols_MSecondaryTourneyTeams])
encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_MSecondaryTourneyTeams), index=MSecondaryTourneyTeams.index)
MSecondaryTourneyTeams = pd.concat([MSecondaryTourneyTeams[num_cols_MSecondaryTourneyTeams], encoded_onehot_df], axis=1)

# encoded_onehot = onehot_encoder.fit_transform(MTeamCoaches[cat_cols_MTeamCoaches])
# encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_MTeamCoaches), index=MTeamCoaches.index)
# MTeamCoaches = pd.concat([MTeamCoaches[num_cols_MTeamCoaches], encoded_onehot_df], axis=1)

hash_encoder = ce.HashingEncoder(cols=cat_cols_MTeamCoaches, n_components=32) #field
MTeamCoaches_encoded = hash_encoder.fit_transform(MTeamCoaches[cat_cols_MTeamCoaches])
MTeamCoaches = pd.concat([MTeamCoaches[num_cols_MTeamCoaches], MTeamCoaches_encoded], axis=1)

encoded_onehot = onehot_encoder.fit_transform(MTeamConferences[cat_cols_MTeamConferences])
encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_MTeamConferences), index=MTeamConferences.index)
MTeamConferences = pd.concat([MTeamConferences[num_cols_MTeamConferences], encoded_onehot_df], axis=1)

encoded_onehot = onehot_encoder.fit_transform(MTeamSpellings[cat_cols_MTeamSpellings])
encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_MTeamSpellings), index=MTeamSpellings.index)
MTeamSpellings = pd.concat([MTeamSpellings[num_cols_MTeamSpellings], encoded_onehot_df], axis=1)

encoded_onehot = onehot_encoder.fit_transform(MTeams[cat_cols_MTeams])
encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_MTeams), index=MTeams.index)
MTeams = pd.concat([MTeams[num_cols_MTeams], encoded_onehot_df], axis=1)

# encoded_onehot = onehot_encoder.fit_transform(SampleSubmissionStage1[cat_cols_SampleSubmissionStage1])
# encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_SampleSubmissionStage1), index=SampleSubmissionStage1.index)
# SampleSubmissionStage1 = pd.concat([SampleSubmissionStage1[num_cols_SampleSubmissionStage1], encoded_onehot_df], axis=1)
#
# encoded_onehot = onehot_encoder.fit_transform(SeedBenchmarkStage1[cat_cols_SeedBenchmarkStage1])
# encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_SeedBenchmarkStage1), index=SeedBenchmarkStage1.index)
# SeedBenchmarkStage1 = pd.concat([SeedBenchmarkStage1[num_cols_SeedBenchmarkStage1], encoded_onehot_df], axis=1)

encoded_onehot = onehot_encoder.fit_transform(WConferenceTourneyGames[cat_cols_WConferenceTourneyGames])
encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_WConferenceTourneyGames), index=WConferenceTourneyGames.index)
WConferenceTourneyGames = pd.concat([WConferenceTourneyGames[num_cols_WConferenceTourneyGames], encoded_onehot_df], axis=1)

encoded_onehot = onehot_encoder.fit_transform(WGameCities[cat_cols_WGameCities])
encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_WGameCities), index=WGameCities.index)
WGameCities = pd.concat([WGameCities[num_cols_WGameCities], encoded_onehot_df], axis=1)

encoded_onehot = onehot_encoder.fit_transform(WNCAATourneyCompactResults[cat_cols_WNCAATourneyCompactResults])
encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_WNCAATourneyCompactResults), index=WNCAATourneyCompactResults.index)
WNCAATourneyCompactResults = pd.concat([WNCAATourneyCompactResults[num_cols_WNCAATourneyCompactResults], encoded_onehot_df], axis=1)

encoded_onehot = onehot_encoder.fit_transform(WNCAATourneyDetailedResults[cat_cols_WNCAATourneyDetailedResults])
encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_WNCAATourneyDetailedResults), index=WNCAATourneyDetailedResults.index)
WNCAATourneyDetailedResults = pd.concat([WNCAATourneyDetailedResults[num_cols_WNCAATourneyDetailedResults], encoded_onehot_df], axis=1)

encoded_onehot = onehot_encoder.fit_transform(WNCAATourneySeeds[cat_cols_WNCAATourneySeeds])
encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_WNCAATourneySeeds), index=WNCAATourneySeeds.index)
WNCAATourneySeeds = pd.concat([WNCAATourneySeeds[num_cols_WNCAATourneySeeds], encoded_onehot_df], axis=1)

encoded_onehot = onehot_encoder.fit_transform(WNCAATourneySlots[cat_cols_WNCAATourneySlots])
encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_WNCAATourneySlots), index=WNCAATourneySlots.index)
WNCAATourneySlots = pd.concat([WNCAATourneySlots[num_cols_WNCAATourneySlots], encoded_onehot_df], axis=1)

encoded_onehot = onehot_encoder.fit_transform(WRegularSeasonCompactResults[cat_cols_WRegularSeasonCompactResults])
encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_WRegularSeasonCompactResults), index=WRegularSeasonCompactResults.index)
WRegularSeasonCompactResults = pd.concat([WRegularSeasonCompactResults[num_cols_WRegularSeasonCompactResults], encoded_onehot_df], axis=1)

encoded_onehot = onehot_encoder.fit_transform(WRegularSeasonDetailedResults[cat_cols_WRegularSeasonDetailedResults])
encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_WRegularSeasonDetailedResults), index=WRegularSeasonDetailedResults.index)
WRegularSeasonDetailedResults = pd.concat([WRegularSeasonDetailedResults[num_cols_WRegularSeasonDetailedResults], encoded_onehot_df], axis=1)

encoded_onehot = onehot_encoder.fit_transform(WSeasons[cat_cols_WSeasons])
encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_WSeasons), index=WSeasons.index)
WSeasons = pd.concat([WSeasons[num_cols_WSeasons], encoded_onehot_df], axis=1)

encoded_onehot = onehot_encoder.fit_transform(WSecondaryTourneyCompactResults[cat_cols_WSecondaryTourneyCompactResults])
encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_WSecondaryTourneyCompactResults), index=WSecondaryTourneyCompactResults.index)
WSecondaryTourneyCompactResults = pd.concat([WSecondaryTourneyCompactResults[num_cols_WSecondaryTourneyCompactResults], encoded_onehot_df], axis=1)

encoded_onehot = onehot_encoder.fit_transform(WSecondaryTourneyTeams[cat_cols_WSecondaryTourneyTeams])
encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_WSecondaryTourneyTeams), index=WSecondaryTourneyTeams.index)
WSecondaryTourneyTeams = pd.concat([WSecondaryTourneyTeams[num_cols_WSecondaryTourneyTeams], encoded_onehot_df], axis=1)

encoded_onehot = onehot_encoder.fit_transform(WTeamConferences[cat_cols_WTeamConferences])
encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_WTeamConferences), index=WTeamConferences.index)
WTeamConferences = pd.concat([WTeamConferences[num_cols_WTeamConferences], encoded_onehot_df], axis=1)

encoded_onehot = onehot_encoder.fit_transform(WTeamSpellings[cat_cols_WTeamSpellings])
encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_WTeamSpellings), index=WTeamSpellings.index)
WTeamSpellings = pd.concat([WTeamSpellings[num_cols_WTeamSpellings], encoded_onehot_df], axis=1)

encoded_onehot = onehot_encoder.fit_transform(WTeams[cat_cols_WTeams])
encoded_onehot_df = pd.DataFrame(encoded_onehot, columns=onehot_encoder.get_feature_names_out(cat_cols_WTeams), index=WTeams.index)
WTeams = pd.concat([WTeams[num_cols_WTeams], encoded_onehot_df], axis=1)

''' NORMALIZATION standardscaler'''

std_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()

data_std = std_scaler.fit_transform(MTeamSpellings)
MTeamSpellings = pd.DataFrame(data_std, columns=MTeamSpellings.columns, index=MTeamSpellings.index)

data_std = std_scaler.fit_transform(Conferences)
Conferences = pd.DataFrame(data_std, columns=Conferences.columns, index=Conferences.index)

data_std = std_scaler.fit_transform(MConferenceTourneyGames)
MConferenceTourneyGames = pd.DataFrame(data_std, columns=MConferenceTourneyGames.columns, index=MConferenceTourneyGames.index)

data_std = std_scaler.fit_transform(MGameCities)
MGameCities = pd.DataFrame(data_std, columns=MGameCities.columns, index=MGameCities.index)

data_std = std_scaler.fit_transform(MMasseyOrdinals)
MMasseyOrdinals = pd.DataFrame(data_std, columns=MMasseyOrdinals.columns, index=MMasseyOrdinals.index)

data_std = std_scaler.fit_transform(MNCAATourneyCompactResults)
MNCAATourneyCompactResults = pd.DataFrame(data_std, columns=MNCAATourneyCompactResults.columns, index=MNCAATourneyCompactResults.index)

data_std = std_scaler.fit_transform(MNCAATourneyDetailedResults)
MNCAATourneyDetailedResults = pd.DataFrame(data_std, columns=MNCAATourneyDetailedResults.columns, index=MNCAATourneyDetailedResults.index)

data_std = std_scaler.fit_transform(MNCAATourneySeedRoundSlots)
MNCAATourneySeedRoundSlots = pd.DataFrame(data_std, columns=MNCAATourneySeedRoundSlots.columns, index=MNCAATourneySeedRoundSlots.index)

data_std = std_scaler.fit_transform(MNCAATourneySeeds)
MNCAATourneySeeds = pd.DataFrame(data_std, columns=MNCAATourneySeeds.columns, index=MNCAATourneySeeds.index)

data_std = std_scaler.fit_transform(MNCAATourneySlots)
MNCAATourneySlots = pd.DataFrame(data_std, columns=MNCAATourneySlots.columns, index=MNCAATourneySlots.index)

data_std = std_scaler.fit_transform(MRegularSeasonCompactResults)
MRegularSeasonCompactResults = pd.DataFrame(data_std, columns=MRegularSeasonCompactResults.columns, index=MRegularSeasonCompactResults.index)

data_std = std_scaler.fit_transform(MRegularSeasonDetailedResults)
MRegularSeasonDetailedResults = pd.DataFrame(data_std, columns=MRegularSeasonDetailedResults.columns, index=MRegularSeasonDetailedResults.index)

data_std = std_scaler.fit_transform(MSeasons)
MSeasons = pd.DataFrame(data_std, columns=MSeasons.columns, index=MSeasons.index)

data_std = std_scaler.fit_transform(MSecondaryTourneyCompactResults)
MSecondaryTourneyCompactResults = pd.DataFrame(data_std, columns=MSecondaryTourneyCompactResults.columns, index=MSecondaryTourneyCompactResults.index)

data_std = std_scaler.fit_transform(MSecondaryTourneyTeams)
MSecondaryTourneyTeams = pd.DataFrame(data_std, columns=MSecondaryTourneyTeams.columns, index=MSecondaryTourneyTeams.index)

data_std = std_scaler.fit_transform(MTeamCoaches)
MTeamCoaches = pd.DataFrame(data_std, columns=MTeamCoaches.columns, index=MTeamCoaches.index)

data_std = std_scaler.fit_transform(MTeamConferences)
MTeamConferences = pd.DataFrame(data_std, columns=MTeamConferences.columns, index=MTeamConferences.index)

data_std = std_scaler.fit_transform(MTeamSpellings)
MTeamSpellings = pd.DataFrame(data_std, columns=MTeamSpellings.columns, index=MTeamSpellings.index)

data_std = std_scaler.fit_transform(MTeams)
MTeams = pd.DataFrame(data_std, columns=MTeams.columns, index=MTeams.index)

data_std = std_scaler.fit_transform(WConferenceTourneyGames)
WConferenceTourneyGames = pd.DataFrame(data_std, columns=WConferenceTourneyGames.columns, index=WConferenceTourneyGames.index)

data_std = std_scaler.fit_transform(WGameCities)
WGameCities = pd.DataFrame(data_std, columns=WGameCities.columns, index=WGameCities.index)

data_std = std_scaler.fit_transform(WNCAATourneyCompactResults)
WNCAATourneyCompactResults = pd.DataFrame(data_std, columns=WNCAATourneyCompactResults.columns, index=WNCAATourneyCompactResults.index)

data_std = std_scaler.fit_transform(WNCAATourneyDetailedResults)
WNCAATourneyDetailedResults = pd.DataFrame(data_std, columns=WNCAATourneyDetailedResults.columns, index=WNCAATourneyDetailedResults.index)

data_std = std_scaler.fit_transform(WNCAATourneySeeds)
WNCAATourneySeeds = pd.DataFrame(data_std, columns=WNCAATourneySeeds.columns, index=WNCAATourneySeeds.index)

data_std = std_scaler.fit_transform(WNCAATourneySlots)
WNCAATourneySlots = pd.DataFrame(data_std, columns=WNCAATourneySlots.columns, index=WNCAATourneySlots.index)

data_std = std_scaler.fit_transform(WRegularSeasonCompactResults)
WRegularSeasonCompactResults = pd.DataFrame(data_std, columns=WRegularSeasonCompactResults.columns, index=WRegularSeasonCompactResults.index)

data_std = std_scaler.fit_transform(WRegularSeasonDetailedResults)
WRegularSeasonDetailedResults = pd.DataFrame(data_std, columns=WRegularSeasonDetailedResults.columns, index=WRegularSeasonDetailedResults.index)

data_std = std_scaler.fit_transform(WSeasons)
WSeasons = pd.DataFrame(data_std, columns=WSeasons.columns, index=WSeasons.index)

data_std = std_scaler.fit_transform(WSecondaryTourneyCompactResults)
WSecondaryTourneyCompactResults = pd.DataFrame(data_std, columns=WSecondaryTourneyCompactResults.columns, index=WSecondaryTourneyCompactResults.index)

data_std = std_scaler.fit_transform(WSecondaryTourneyTeams)
WSecondaryTourneyTeams = pd.DataFrame(data_std, columns=WSecondaryTourneyTeams.columns, index=WSecondaryTourneyTeams.index)

data_std = std_scaler.fit_transform(WTeamConferences)
WTeamConferences = pd.DataFrame(data_std, columns=WTeamConferences.columns, index=WTeamConferences.index)

data_std = std_scaler.fit_transform(WTeamSpellings)
WTeamSpellings = pd.DataFrame(data_std, columns=WTeamSpellings.columns, index=WTeamSpellings.index)

data_std = std_scaler.fit_transform(WTeams)
WTeams = pd.DataFrame(data_std, columns=WTeams.columns, index=WTeams.index)

# MinMax Scaler
# data_mm = minmax_scaler.fit_transform(MTeamSpellings)
# MTeamSpellings = pd.DataFrame(data_mm, columns=MTeamSpellings.columns, index=MTeamSpellings.index)
#
# data_mm = minmax_scaler.fit_transform(Conferences)
# Conferences = pd.DataFrame(data_mm, columns=Conferences.columns, index=Conferences.index)
#
# data_mm = minmax_scaler.fit_transform(MConferenceTourneyGames)
# MConferenceTourneyGames = pd.DataFrame(data_mm, columns=MConferenceTourneyGames.columns, index=MConferenceTourneyGames.index)
#
# data_mm = minmax_scaler.fit_transform(MGameCities)
# MGameCities = pd.DataFrame(data_mm, columns=MGameCities.columns, index=MGameCities.index)
#
# data_mm = minmax_scaler.fit_transform(MMasseyOrdinals)
# MMasseyOrdinals = pd.DataFrame(data_mm, columns=MMasseyOrdinals.columns, index=MMasseyOrdinals.index)
#
# data_mm = minmax_scaler.fit_transform(MNCAATourneyCompactResults)
# MNCAATourneyCompactResults = pd.DataFrame(data_mm, columns=MNCAATourneyCompactResults.columns, index=MNCAATourneyCompactResults.index)
#
# data_mm = minmax_scaler.fit_transform(MNCAATourneyDetailedResults)
# MNCAATourneyDetailedResults = pd.DataFrame(data_mm, columns=MNCAATourneyDetailedResults.columns, index=MNCAATourneyDetailedResults.index)
#
# data_mm = minmax_scaler.fit_transform(MNCAATourneySeedRoundSlots)
# MNCAATourneySeedRoundSlots = pd.DataFrame(data_mm, columns=MNCAATourneySeedRoundSlots.columns, index=MNCAATourneySeedRoundSlots.index)
#
# data_mm = minmax_scaler.fit_transform(MNCAATourneySeeds)
# MNCAATourneySeeds = pd.DataFrame(data_mm, columns=MNCAATourneySeeds.columns, index=MNCAATourneySeeds.index)
#
# data_mm = minmax_scaler.fit_transform(MNCAATourneySlots)
# MNCAATourneySlots = pd.DataFrame(data_mm, columns=MNCAATourneySlots.columns, index=MNCAATourneySlots.index)
#
# data_mm = minmax_scaler.fit_transform(MRegularSeasonCompactResults)
# MRegularSeasonCompactResults = pd.DataFrame(data_mm, columns=MRegularSeasonCompactResults.columns, index=MRegularSeasonCompactResults.index)
#
# data_mm = minmax_scaler.fit_transform(MRegularSeasonDetailedResults)
# MRegularSeasonDetailedResults = pd.DataFrame(data_mm, columns=MRegularSeasonDetailedResults.columns, index=MRegularSeasonDetailedResults.index)
#
# data_mm = minmax_scaler.fit_transform(MSeasons)
# MSeasons = pd.DataFrame(data_mm, columns=MSeasons.columns, index=MSeasons.index)
#
# data_mm = minmax_scaler.fit_transform(MSecondaryTourneyCompactResults)
# MSecondaryTourneyCompactResults = pd.DataFrame(data_mm, columns=MSecondaryTourneyCompactResults.columns, index=MSecondaryTourneyCompactResults.index)
#
# data_mm = minmax_scaler.fit_transform(MSecondaryTourneyTeams)
# MSecondaryTourneyTeams = pd.DataFrame(data_mm, columns=MSecondaryTourneyTeams.columns, index=MSecondaryTourneyTeams.index)
#
# data_mm = minmax_scaler.fit_transform(MTeamCoaches)
# MTeamCoaches = pd.DataFrame(data_mm, columns=MTeamCoaches.columns, index=MTeamCoaches.index)
#
# data_mm = minmax_scaler.fit_transform(MTeamConferences)
# MTeamConferences = pd.DataFrame(data_mm, columns=MTeamConferences.columns, index=MTeamConferences.index)
#
# data_mm = minmax_scaler.fit_transform(MTeamSpellings)
# MTeamSpellings = pd.DataFrame(data_mm, columns=MTeamSpellings.columns, index=MTeamSpellings.index)
#
# data_mm = minmax_scaler.fit_transform(MTeams)
# MTeams = pd.DataFrame(data_mm, columns=MTeams.columns, index=MTeams.index)
#
# data_mm = minmax_scaler.fit_transform(WConferenceTourneyGames)
# WConferenceTourneyGames = pd.DataFrame(data_mm, columns=WConferenceTourneyGames.columns, index=WConferenceTourneyGames.index)
#
# data_mm = minmax_scaler.fit_transform(WGameCities)
# WGameCities = pd.DataFrame(data_mm, columns=WGameCities.columns, index=WGameCities.index)
#
# data_mm = minmax_scaler.fit_transform(WNCAATourneyCompactResults)
# WNCAATourneyCompactResults = pd.DataFrame(data_mm, columns=WNCAATourneyCompactResults.columns, index=WNCAATourneyCompactResults.index)
#
# data_mm = minmax_scaler.fit_transform(WNCAATourneyDetailedResults)
# WNCAATourneyDetailedResults = pd.DataFrame(data_mm, columns=WNCAATourneyDetailedResults.columns, index=WNCAATourneyDetailedResults.index)
#
# data_mm = minmax_scaler.fit_transform(WNCAATourneySeeds)
# WNCAATourneySeeds = pd.DataFrame(data_mm, columns=WNCAATourneySeeds.columns, index=WNCAATourneySeeds.index)
#
# data_mm = minmax_scaler.fit_transform(WNCAATourneySlots)
# WNCAATourneySlots = pd.DataFrame(data_mm, columns=WNCAATourneySlots.columns, index=WNCAATourneySlots.index)
#
# data_mm = minmax_scaler.fit_transform(WRegularSeasonCompactResults)
# WRegularSeasonCompactResults = pd.DataFrame(data_mm, columns=WRegularSeasonCompactResults.columns, index=WRegularSeasonCompactResults.index)
#
# data_mm = minmax_scaler.fit_transform(WRegularSeasonDetailedResults)
# WRegularSeasonDetailedResults = pd.DataFrame(data_mm, columns=WRegularSeasonDetailedResults.columns, index=WRegularSeasonDetailedResults.index)
#
# data_mm = minmax_scaler.fit_transform(WSeasons)
# WSeasons = pd.DataFrame(data_mm, columns=WSeasons.columns, index=WSeasons.index)
#
# data_mm = minmax_scaler.fit_transform(WSecondaryTourneyCompactResults)
# WSecondaryTourneyCompactResults = pd.DataFrame(data_mm, columns=WSecondaryTourneyCompactResults.columns, index=WSecondaryTourneyCompactResults.index)
#
# data_mm = minmax_scaler.fit_transform(WSecondaryTourneyTeams)
# WSecondaryTourneyTeams = pd.DataFrame(data_mm, columns=WSecondaryTourneyTeams.columns, index=WSecondaryTourneyTeams.index)
#
# data_mm = minmax_scaler.fit_transform(WTeamConferences)
# WTeamConferences = pd.DataFrame(data_mm, columns=WTeamConferences.columns, index=WTeamConferences.index)
#
# data_mm = minmax_scaler.fit_transform(WTeamSpellings)
# WTeamSpellings = pd.DataFrame(data_mm, columns=WTeamSpellings.columns, index=WTeamSpellings.index)
#
# data_mm = minmax_scaler.fit_transform(WTeams)
# WTeams = pd.DataFrame(data_mm, columns=WTeams.columns, index=WTeams.index)

''' IMPACT ON MODEL '''
''' High
Teams ~ key for identification teams
NCAATourneyCompactResults ~ all games in history
Seasons ~ all about date
NCAATourneySeeds ~ draws
RegularSeasonCompactResults ~ analyze 

    Middle 
MasseyOrdinals ~ teams rating 
ConferenceTourneyGames ~ some competitions
RegularSeasonDetailedResults ~ detail stats
ConferenceTourneyGames, TeamCoaches, TeamConferences, Cities all, SecondaryTourneyCompactResults

    Low
Cities, MGameCities, WGameCities, MTeamSpellings, WTeamSpellings, SeedBenchmarkStage1,
MSecondaryTourneyCompactResults, WSecondaryTourneyCompactResults & lists
'''

''' MAIN DATA '''
''' TRAIN TEST VALIDATION FOR '''

# aggregate rating - av. rating on team and season
def aggregate_massey(massey_df):
    massey_agg = massey_df.groupby(['Season', 'TeamID'])['OrdinalRank'].mean().reset_index()
    massey_agg.rename(columns={'OrdinalRank': 'AvgOrdinalRank'}, inplace=True)
    return massey_agg


# male
def merge_mens_data():
    # 1. Базовый датасет – результаты турнирных игр
    m_games = MNCAATourneyCompactResults.copy()

    # 2. Добавляем информацию о командах: для выигравшей и проигравшей команды
    m_games = m_games.merge(MTeams, left_on='WTeamID', right_on='TeamID', how='left',
                            suffixes=('', '_W'))
    m_games = m_games.merge(MTeams, left_on='LTeamID', right_on='TeamID', how='left',
                            suffixes=('', '_L'))

    # 3. Добавляем информацию о сезоне (например, регионы, DayZero)
    m_games = m_games.merge(MSeasons, on='Season', how='left')

    # 4. Добавляем посевы турнира для выигравшей и проигравшей команд
    m_games = m_games.merge(MNCAATourneySeeds, left_on=['Season', 'WTeamID'],
                            right_on=['Season', 'TeamID'], how='left',
                            suffixes=('', '_seed_W'))
    m_games = m_games.merge(MNCAATourneySeeds, left_on=['Season', 'LTeamID'],
                            right_on=['Season', 'TeamID'], how='left',
                            suffixes=('', '_seed_L'))

    # 5. Добавляем агрегированные рейтинги (Massey Ordinals)
    massey_m = aggregate_massey(MMasseyOrdinals)
    m_games = m_games.merge(massey_m, left_on=['Season', 'WTeamID'],
                            right_on=['Season', 'TeamID'], how='left',
                            suffixes=('', '_massey_W'))
    m_games = m_games.merge(massey_m, left_on=['Season', 'LTeamID'],
                            right_on=['Season', 'TeamID'], how='left',
                            suffixes=('', '_massey_L'))

    # 6. Добавляем агрегированные данные регулярного сезона (например, общее число побед и поражений)
    m_wins = MRegularSeasonCompactResults.groupby(['Season', 'WTeamID']).size().reset_index(name='Wins')
    m_losses = MRegularSeasonCompactResults.groupby(['Season', 'LTeamID']).size().reset_index(name='Losses')
    m_record = m_wins.merge(m_losses, left_on=['Season', 'WTeamID'], right_on=['Season', 'LTeamID'], how='outer')
    m_record['Wins'] = m_record['Wins'].fillna(0)
    m_record['Losses'] = m_record['Losses'].fillna(0)
    m_record['TeamID'] = m_record['WTeamID'].combine_first(m_record['LTeamID'])
    m_record = m_record[['Season', 'TeamID', 'Wins', 'Losses']]
    m_games = m_games.merge(m_record, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'],
                            how='left', suffixes=('', '_reg_W'))
    m_record_lose = m_record.rename(columns={'TeamID': 'TeamID_L', 'Wins': 'Wins_reg_L', 'Losses': 'Losses_reg_L'})
    m_games = m_games.merge(m_record_lose, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID_L'],
                            how='left')

    # 7. Добавляем данные детализированных результатов регулярного сезона
    m_detail_w = MRegularSeasonDetailedResults.groupby(['Season', 'WTeamID'])['WScore'].mean().reset_index()
    m_detail_w.rename(columns={'WScore': 'AvgWScore'}, inplace=True)
    m_games = m_games.merge(m_detail_w, left_on=['Season', 'WTeamID'], right_on=['Season', 'WTeamID'], how='left')

    # 8. Добавляем данные конференционных турниров (например, число побед в конференции)
    m_conf = MConferenceTourneyGames.groupby(['Season', 'WTeamID']).size().reset_index(name='ConfWins')
    m_games = m_games.merge(m_conf, left_on=['Season', 'WTeamID'], right_on=['Season', 'WTeamID'], how='left')

    # 9. Добавляем информацию по тренерам (например, последний тренер в сезоне)
    # После применения HashingEncoder к MTeamCoaches исходный столбец 'CoachName' заменён на столбцы 'col_0', 'col_1', ..., 'col_31'
    coach_hash_cols = [col for col in MTeamCoaches.columns if col.startswith('col_')]
    m_coaches = MTeamCoaches.sort_values('LastDayNum').groupby(['Season', 'TeamID']).tail(1)
    m_coaches = m_coaches[['Season', 'TeamID'] + coach_hash_cols]
    m_games = m_games.merge(m_coaches, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'],
                            how='left', suffixes=('', '_coach_W'))
    m_games = m_games.merge(m_coaches, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'],
                            how='left', suffixes=('', '_coach_L'))

    # 10. Добавляем данные конференций
    m_games = m_games.merge(MTeamConferences, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'],
                            how='left', suffixes=('', '_conf_W'))
    m_games = m_games.merge(MTeamConferences, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'],
                            how='left', suffixes=('', '_conf_L'))

    # 11. Добавляем данные о городах игр (например, ID города, где проходила игра)
    m_games = m_games.merge(MGameCities, on=['Season', 'DayNum', 'WTeamID', 'LTeamID'], how='left',
                            suffixes=('', '_city'))

    # 12. Добавляем данные вторичных турниров
    m_games = m_games.merge(MSecondaryTourneyCompactResults, on=['Season', 'DayNum', 'WTeamID', 'LTeamID'], how='left',
                            suffixes=('', '_secondary'))

    # Добавляем индикатор мужских игр
    m_games['Gender'] = 'M'

    return m_games

# female
def merge_womens_data():
    # 1. Базовый датасет – результаты турнирных игр
    w_games = WNCAATourneyCompactResults.copy()

    # 2. Добавляем информацию о командах
    w_games = w_games.merge(WTeams, left_on='WTeamID', right_on='TeamID', how='left',
                            suffixes=('', '_W'))
    w_games = w_games.merge(WTeams, left_on='LTeamID', right_on='TeamID', how='left',
                            suffixes=('', '_L'))

    # 3. Добавляем информацию о сезоне
    w_games = w_games.merge(WSeasons, on='Season', how='left')

    # 4. Добавляем посевы турнира
    w_games = w_games.merge(WNCAATourneySeeds, left_on=['Season', 'WTeamID'],
                            right_on=['Season', 'TeamID'], how='left',
                            suffixes=('', '_seed_W'))
    w_games = w_games.merge(WNCAATourneySeeds, left_on=['Season', 'LTeamID'],
                            right_on=['Season', 'TeamID'], how='left',
                            suffixes=('', '_seed_L'))

    # 5. Для женских данных рейтинги Massey обычно не предоставляются, поэтому этот шаг пропускаем или можно добавить,
    # если появятся сторонние рейтинги.

    # 6. Добавляем агрегированные данные регулярного сезона
    w_wins = WRegularSeasonCompactResults.groupby(['Season', 'WTeamID']).size().reset_index(name='Wins')
    w_losses = WRegularSeasonCompactResults.groupby(['Season', 'LTeamID']).size().reset_index(name='Losses')
    w_record = w_wins.merge(w_losses, left_on=['Season', 'WTeamID'], right_on=['Season', 'LTeamID'], how='outer')
    w_record['Wins'] = w_record['Wins'].fillna(0)
    w_record['Losses'] = w_record['Losses'].fillna(0)
    w_record['TeamID'] = w_record['WTeamID'].combine_first(w_record['LTeamID'])
    w_record = w_record[['Season', 'TeamID', 'Wins', 'Losses']]
    w_games = w_games.merge(w_record, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left',
                            suffixes=('', '_reg_W'))
    w_record_lose = w_record.rename(columns={'TeamID': 'TeamID_L', 'Wins': 'Wins_reg_L', 'Losses': 'Losses_reg_L'})
    w_games = w_games.merge(w_record_lose, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID_L'], how='left')

    # 7. Добавляем агрегированные детализированные результаты регулярного сезона
    w_detail_w = WRegularSeasonDetailedResults.groupby(['Season', 'WTeamID'])['WScore'].mean().reset_index()
    w_detail_w.rename(columns={'WScore': 'AvgWScore'}, inplace=True)
    w_games = w_games.merge(w_detail_w, on=['Season', 'WTeamID'], how='left')

    # 8. Добавляем данные конференционных турниров
    w_conf = WConferenceTourneyGames.groupby(['Season', 'WTeamID']).size().reset_index(name='ConfWins')
    w_games = w_games.merge(w_conf, on=['Season', 'WTeamID'], how='left')

    # 9. Женские данные по тренерам обычно отсутствуют в базовом наборе, поэтому этот шаг можно пропустить
    # (либо добавить, если есть соответствующий датасет)

    # 10. Добавляем данные конференций
    w_games = w_games.merge(WTeamConferences, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'],
                            how='left', suffixes=('', '_conf_W'))
    w_games = w_games.merge(WTeamConferences, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'],
                            how='left', suffixes=('', '_conf_L'))

    # 11. Добавляем информацию о городах игр
    w_games = w_games.merge(WGameCities, on=['Season', 'DayNum', 'WTeamID', 'LTeamID'], how='left')

    # 12. Добавляем данные вторичных турниров
    w_games = w_games.merge(WSecondaryTourneyCompactResults, on=['Season', 'DayNum', 'WTeamID', 'LTeamID'], how='left',
                            suffixes=('', '_secondary'))
    w_games['Gender'] = 'W'

    return w_games

# merge female w. male; major dataframe
def builddata():
    mens_df = merge_mens_data()
    womens_df = merge_womens_data()
    full_df = pd.concat([mens_df, womens_df], ignore_index=True)
    # Можно провести дополнительные операции по очистке, заполнению пропусков и т.д.
    return full_df
chief_data = builddata()
# print(chief_data.shape[1]) # dimension 1302

# drop columns with all zeros in data, so models cant work
missing_counts = chief_data.isnull().sum()
cols_to_drop = missing_counts[missing_counts == len(chief_data)].index.tolist()
chief_data.drop(columns=cols_to_drop, inplace=True)
chief_data.drop(columns='WLoc_A', inplace=True)
chief_data.drop(columns='WLoc_H', inplace=True)

# target
chief_data['target'] = np.where(chief_data['WTeamID'] == chief_data[['WTeamID','LTeamID']].min(axis=1), 1, 0)
# if team with lowest id win --> target==1, otherwise 0

''' (actually) SEPARATION, 80~10~10 '''

train_data, temp_data = train_test_split(chief_data, test_size=0.2, random_state=42)
validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
# 4168 --> 3334-417-417

X_train = train_data.drop('target', axis=1)
y_train = train_data['target']

X_val = validation_data.drop('target', axis=1)
y_val = validation_data['target']

X_test = test_data.drop('target', axis=1)
y_test = test_data['target']

# Decrease features
# X_train = pd.get_dummies(train_data.drop('target', axis=1), drop_first=True).copy()
# X_val = pd.get_dummies(validation_data.drop('target', axis=1), drop_first=True).copy()
# X_test = pd.get_dummies(test_data.drop('target', axis=1), drop_first=True).copy()
# # Приводим столбцы валидационного и тестового наборов к тем же, что и у тренировочного
# X_val = X_val.reindex(columns=X_train.columns, fill_value=0).copy()
# X_test = X_test.reindex(columns=X_train.columns, fill_value=0).copy()
# # Обучаем RandomForestClassifier
# rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
# rf.fit(X_train, train_data['target'])
# # Посмотрим распределение важностей признаков
# max_importance = np.max(rf.feature_importances_)
# median_importance = np.median(rf.feature_importances_)
# print("Max feature importance:", max_importance)
# print("Median feature importance:", median_importance)
# # Задаем порог отбора признаков на уровне 75-го процентиля
# threshold_value = np.percentile(rf.feature_importances_, 75)
# print("Threshold (75th percentile):", threshold_value)
# selector = SelectFromModel(rf, threshold=threshold_value, prefit=True)
# # Преобразуем данные
# X_train_reduced = selector.transform(X_train)
# X_val_reduced = selector.transform(X_val)
# X_test_reduced = selector.transform(X_test)
# # Получаем имена выбранных признаков
# selected_features = X_train.columns[selector.get_support()]
# print("Исходное число признаков:", X_train.shape[1])
# print("После отбора признаков:", X_train_reduced.shape[1])
# print("Оставшиеся признаки:", len(list(selected_features)))

''' CROSS-VALIDATION '''
X = train_data.drop('target', axis=1)
y = train_data['target']
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# # just check if our model worked with dummy ass
# y_true = validation_data['target']
# y_prob_dummy = np.full(len(y_true), 0.5)
# brier = brier_score_loss(y_true, y_prob_dummy)
# print("Brier Score (dummy classifier):", brier)


''' METRICS brier-score '''
def custom_brier_score(y_true, y_prob, **kwargs):
    # Если y_prob одномерный, предполагаем, что это вероятности для класса 1
    if y_prob.ndim == 1:
        prob = y_prob
    else:
        prob = y_prob[:, 1]
    return  brier_score_loss(y_true, prob)

brier_scorer = make_scorer(custom_brier_score, needs_proba=True)

''' HYPERPARAMETERS choosing '''

# Отделяем признаки и таргет из тренировочного набора
X_train = pd.get_dummies(X_train, drop_first=True)
# Если в X_train еще остались категориальные признаки, обязательно их закодируйте числовым образом
# Определяем сетку гиперпараметров
# param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10],'max_features': ['sqrt', 'log2', None]} # was
param_grid = {
    'n_estimators': [200],     # best option
    'max_depth': [None],       # best option
    'min_samples_split': [2],  # best option
    'max_features': [None]     # best option
}
# Создаем базовую модель
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
# Настраиваем GridSearchCV с 5-кратной кросс-валидацией и выбранным scorer
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring=brier_scorer, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
# print("Лучшие параметры:", grid_search.best_params_) # {'max_depth': None, 'max_features': None, 'min_samples_split': 2, 'n_estimators': 200}
# print("Лучшая (отрицательная) оценка Brier Score:", grid_search.best_score_) #-0.009298954126540334

''' BASIC MODEL '''
X_val = pd.get_dummies(X_val, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)
# Убедимся, что все наборы данных имеют одинаковые признаки
X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
# После GridSearchCV:
final_model = grid_search.best_estimator_
# Обучаем финальную модель на полном тренировочном наборе
final_model.fit(X_train, y_train)
# Сохраняем результаты оценки на валидационном наборе
val_score = -brier_score_loss(validation_data['target'], final_model.predict_proba(X_val)[:, 1])
# print("Final Validation Brier Score:", val_score) #-0.00946294964028777

rf_best = RandomForestClassifier(
    max_depth=None,
    max_features=None,
    min_samples_split=2,
    n_estimators=200,
    random_state=42
)
rf_best.fit(X_train, y_train)

y_pred_val = rf_best.predict(X_val)
brier_val = custom_brier_score(y_val, y_pred_val)
# print(f"Brier Score на валидации: {brier_val}") # 0.009592326139088728

y_pred_test = rf_best.predict(X_test)
brier_test = custom_brier_score(y_test, y_pred_test)
# print(f"Brier Score на тесте: {brier_test}") # 0.016786570743405275

''' MODELS '''
imputer = SimpleImputer(strategy='mean')

X_train_imputed = imputer.fit_transform(X_train)
X_val_imputed = imputer.transform(X_val)
X_test_imputed = imputer.transform(X_test)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
xgb = XGBClassifier(n_estimators=100, random_state=42)
catboost = CatBoostClassifier(n_estimators=100, random_state=42, verbose=0)

meta_model = LogisticRegression()

stacking_model = StackingClassifier(
    estimators=[('rf', rf), ('gb', gb), ('xgb', xgb), ('catboost', catboost)],
    final_estimator=meta_model
)

stacking_model.fit(X_train_imputed, y_train)
stacking_score = stacking_model.score(X_test_imputed, y_test)
print("Accuracy of Stacking model after imputing missing values: ", stacking_score)

# 1. Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict_proba(X_test)[:, 1]
brier_rf = brier_score_loss(y_test, y_pred_rf)
print(f"Random Forest Brier Score: {brier_rf}")

# 2. Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict_proba(X_test)[:, 1]
brier_gb = brier_score_loss(y_test, y_pred_gb)
print(f"Gradient Boosting Brier Score: {brier_gb}")

# 3. XGBoost
xgb = XGBClassifier(n_estimators=100, random_state=42)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict_proba(X_test)[:, 1]
brier_xgb = brier_score_loss(y_test, y_pred_xgb)
print(f"XGBoost Brier Score: {brier_xgb}")

# 4. CatBoost
catboost = CatBoostClassifier(n_estimators=100, random_state=42, verbose=0)
catboost.fit(X_train, y_train)
y_pred_catboost = catboost.predict_proba(X_test)[:, 1]
brier_catboost = brier_score_loss(y_test, y_pred_catboost)
print(f"CatBoost Brier Score: {brier_catboost}")

# 5. Logistic Regression
lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict_proba(X_test)[:, 1]
brier_lr = brier_score_loss(y_test, y_pred_lr)
print(f"Logistic Regression Brier Score: {brier_lr}")

# 6. SVM
svm = SVC(probability=True, random_state=42)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict_proba(X_test)[:, 1]
brier_svm = brier_score_loss(y_test, y_pred_svm)
print(f"SVM Brier Score: {brier_svm}")

# 7. kNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict_proba(X_test)[:, 1]
brier_knn = brier_score_loss(y_test, y_pred_knn)
print(f"KNN Brier Score: {brier_knn}")

# 8. Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict_proba(X_test)[:, 1]
brier_nb = brier_score_loss(y_test, y_pred_nb)
print(f"Naive Bayes Brier Score: {brier_nb}")

# 9. MLP
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict_proba(X_test)[:, 1]
brier_mlp = brier_score_loss(y_test, y_pred_mlp)
print(f"MLP Brier Score: {brier_mlp}")

