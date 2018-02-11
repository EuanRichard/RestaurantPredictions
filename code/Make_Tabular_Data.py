
# coding: utf-8

# # Introduction and notes
# 
# Regression seems to work better than time series due to lots of gaps in the data. If we have time, we could try an ensemble model with time series, and a Chainer feed-forward NN (thanks Richard!).
# 
# **XGB Model:** For each date, run a model, our stores are our targets *y*. Many of our predictors *X* are based on that store's history of customers. We can also add general information based on stores in a similar location and genre, plus general information from all stores for the date of the year etc.
# 
# We then make a new model for each new day that we want to predict, e.g. we can predict tomorrow first, then 明後日,　明々後日 etc. by a new model each time. This can be done by looping over the code.
# 
# Then, we can either scan over the training data we have e.g. in 1 month steps, make an ensemble of models, and fit on the test data. Or, we should just focus on (or give a much higher weight) to the model that covered last year's Golden Week.
# 
# See `vizualization.py` in this directory for some justifications of this stuff.
# 
# ### To do
# 
# Golden week predictions need to be edited manually.
# 
# ## Okay, let's get started
# 
# Standard startup fluff.

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[2]:


# Global variables

# a list of stores that are in the train but not the test data
# can drop them for better performance
missing_stores = ["air_d63cfa6d6ab78446","air_b2d8bc9c88b85f96","air_0ead98dd07e7a82a","air_cb083b4789a8d3a2","air_cf22e368c1a71d53","air_d0a7bd3339c3d12a","air_229d7e508d9f1b5e","air_2703dcb33192b181"]

# a list of reservations that seem to be data entry errors
# not actually using these so don't need to drop them
#bad_res = ['air_2a485b92210c98b5', 'air_465bddfed3353b23',       'air_56cebcbd6906e04c', 'air_900d755ebd2f7bbd',       'air_900d755ebd2f7bbd', 'air_900d755ebd2f7bbd',       'air_a17f0778617c76e2', 'air_a17f0778617c76e2',       'air_a17f0778617c76e2', 'air_a17f0778617c76e2',       'air_a17f0778617c76e2', 'air_b439391e72899756',       'air_e7fbee4e3cfe65c5', 'air_e7fbee4e3cfe65c5',       'air_e7fbee4e3cfe65c5', 'air_e7fbee4e3cfe65c5']
#bad_res_dates = [datetime.date(2017, 1, 18), datetime.date(2017, 1, 9),       datetime.date(2017, 3, 19), datetime.date(2017, 3, 3),       datetime.date(2017, 3, 7), datetime.date(2017, 3, 24),       datetime.date(2016, 11, 10), datetime.date(2016, 11, 11),       datetime.date(2016, 12, 18), datetime.date(2017, 1, 5),       datetime.date(2017, 3, 16), datetime.date(2017, 2, 23),       datetime.date(2017, 2, 3), datetime.date(2017, 2, 4),       datetime.date(2017, 2, 5), datetime.date(2017, 2, 7)]


# # Set the Prediction Date
# 
# Set the point that we are predicting from (i.e. the first unknown day).

# In[3]:


from datetime import datetime
from datetime import timedelta
one_day = timedelta(days=1)

# The test set will be:
prediction_date = datetime(2017,4,23)
# For now we use as a train set:
#prediction_date = datetime(2017,4,23) - timedelta(days=39)

# Set how many extra days ahead of this we will predict

# SCRIPT VERSION
from sys import argv
if (len(argv) > 3) or (len(argv) < 2):
    print("Usage: days_ahead (run_calculation)")
    raise SystemExit
run_calculation = (len(argv) == 3)
days_ahead = int(argv[1])

# NOTEBOOK VERSION
#days_ahead = 1

today = prediction_date + timedelta(days=days_ahead)

print("Building tabular data for",  today)


# # Build Variables

# 
# 1. Genre of cuisine
# 1. Location
#  * todofuken
#  * ku/shi
#  * throw away more detailed info?
# 1. Reservations
#  * reserved for this day in AIR
#  * reserved for this day in HPG
#  * **note**: cutoff registrations info after prediction_date, e.g. don't use on the day reservations
#  * **note**: AIR reservations follow a weird pattern, could be related to system downtime or something. Could introduce a variable to account for this?
# 1. Customers
#  * each day for ~2 weeks before
#  * average for last week for ~4 weeks before
#  * trend for last week (average), last month, last year [taylor series!]
#  * average for this store on this day of the week (past ~6 months)
# 1. is a holiday or not
# 1. is golden week * is in Tokyo
# 1. further combinations of golden week & genre?
# 1. Average info from other stores
#   1. for average over all stores in this location:
#     * customers on this day last year 
#     * customers on this day of the week (normalized?)
#     * customers over the last week
#     * customers on this day last week 
#   1.  for average over all stores in this genre:
#     *    same as above (normalized to the average customers for this genre?) 
#   1. for average over all stores:
#     * customers on this day last year (normalized)
#  
#  ### Other ideas
#  
#  
# 1. check how far in advance reservations are 
# 1. final scores will be float, but try rounding up or down to int to account for some model bias
# 1. day of the week (currently this should be accounted for by customers model, but...)
# 
# ### Set up the initial dataset
# Remember that our targets *y* are customers for each **store**, and we will re-train for different days. So, the predictor set *X* should be information relevant to each store.

# In[5]:


# Read store data
stores = pd.read_csv("../input/air_store_info.csv",
                    index_col = ["air_store_id"])

# drop missing stores
stores.drop(missing_stores, inplace=True)

# forget about latitude and longitude (area is more important?)
stores.drop(["latitude", "longitude"], inplace=True, axis=1)


# ## Location

# In[6]:


address = stores["air_area_name"].map(str.split).values

todofuken, kushi = [], []
for store in address:
    todofuken.append(store[0])
    kushi.append(store[1])

stores.drop(["air_area_name"], axis=1, inplace=True)
stores = stores.assign(todofuken = todofuken, kushi = kushi)


# ## Reservations

# In[6]:


# Read reservations data (AIR = 6 MB, HPG = 120 MB)
reservations_air = pd.read_csv("../input/air_reserve.csv",
                              parse_dates=["visit_datetime", "reserve_datetime"])
reservations_hpg = pd.read_csv("../input/hpg_reserve.csv",
                              parse_dates=["visit_datetime", "reserve_datetime"])

# drop any reservations from ON or AFTER the prediction date,
# e.g. on-the-day reservations, because we won't have this info
# in the test set.
reservations_air = reservations_air[ reservations_air["reserve_datetime"].map(datetime.date) < prediction_date.date() ]
reservations_hpg = reservations_hpg[ reservations_hpg["reserve_datetime"].map(datetime.date) < prediction_date.date() ]

# otherwise not using the time of making the reservation, for now
reservations_air.drop(["reserve_datetime"], axis=1, inplace=True)
reservations_hpg.drop(["reserve_datetime"], axis=1, inplace=True)

# We don't care about HPG reservations for most of the HPG data.
# So we can throw most of it away, but keep the ones that are reservations
# for restaurants in the AIR system.

# Assign AIR store number to HPG reservations data
store_id = pd.read_csv("../input/store_id_relation.csv")
store_id.set_index(["hpg_store_id"], inplace=True)

def get_air_store(hpg_store):
    if hpg_store in store_id.index:
        return store_id.loc[hpg_store,"air_store_id"]
    else:
        return np.nan

air_ids = reservations_hpg["hpg_store_id"].map(get_air_store).values
reservations_hpg = reservations_hpg.assign(air_store_id = air_ids)

# prune any HPG reservations that aren't for stores in AIR
reservations_hpg.dropna(axis=0, inplace=True)
# also drop the HPG number, we don't need it anymore
reservations_hpg.drop(["hpg_store_id"], axis=1, inplace=True)


# In[7]:


### Get reservations for this day
res_air = []
res_hpg = []

#  Pick out all reservations for each store
for store in stores.index:
    res = reservations_air[reservations_air["air_store_id"] == store]

    # pick out reservations for today    
    res = res[ res["visit_datetime"].map(datetime.date) == today.date() ]
    
    # sum reservations for today
    res_air.append( res["reserve_visitors"].sum() )
    
    # same steps for HPG
    res = reservations_hpg[reservations_hpg["air_store_id"] == store]
    res = res[ res["visit_datetime"].map(datetime.date) == today.date() ]
    res_hpg.append( res["reserve_visitors"].sum() )

# Add air reservations for each store
stores = stores.assign(res_air = res_air, res_hpg = res_hpg )


# ## Customer History
# 
# ### Read in visitor data

# In[7]:


visits = pd.read_csv("../input/air_visit_data.csv",
                     parse_dates=["visit_date"],
                     index_col=["visit_date"])
visits = visits.sort_index()

# Drop the missing stores
for store in missing_stores:
    visits = visits[ visits["air_store_id"] != store ]

# print some basic info
print("Total visits:",visits.shape[0])
print("Number of unique stores after drop:",visits["air_store_id"].unique().shape[0])


# ### Visits per day, week, and month
# 
# Here we get customers:
#  * each day
#  * average for this day of the week
#  * weekly average
#  * monthly average
#  * trend for last week, last month, and last 3 months (Taylor series)
#  
# This is not the most pretty or efficient way of doing things, but at the moment calculating all of the above in one big loop using relative dates, working gradually into the past for 6 months. Note that here a month is defined as exactly 4 weeks. (The full range of data will be used later for the "average from other stores" section.)
# 
# **Important note:** Both no customer and shop closed days are missing in the dataset (visits=0). We ignore those days when taking averages, since they are also ignored in the test-set scoring.

# In[10]:


print("Scanning Customer Visits...")

# set a double index so we can look up date/store easily
visits = visits.reset_index().set_index(["visit_date", "air_store_id"]).sort_index()

# Make a new dataframe containing the visits data we will append to "stores"
preds = pd.DataFrame(index = stores.index)

# start on the day before today
day = today - one_day

# ignore division by zero errors
np.seterr(divide="ignore", invalid="ignore")

for months_before in range(6):
    print("Processing", months_before, "months before")
    
    for weeks_before in range(4):

        for days_before in range(7):

            visits_day = []
            
            # get visit data for EACH store, if today is before the prediction cutoff
            for store in stores.index:
                if ((day,store) in visits.index) and (day < prediction_date):
                    visits_day.append(visits.loc[(day,store)]["visitors"])
                else:
                    visits_day.append(0)
                    
            # also keep track of "store open / closed" for averaging purposes
            visits_day_count = [1 if i > 0 else 0 for i in visits_day]

            # If in the first 7 days, store daily data
            # (not currently using)
            # unpack a dictionary with ** to provide variable names for columns
            #if months_before == 0 and weeks_before == 0:
            #    preds = preds.assign(**{"visits_D-"+str(days_before + 1) : visits_day})
                
            # if correct day of the week, add to average for this weekday
            if days_before == 6: 
                if weeks_before == 0:
                    weekday = visits_day
                    weekday_count = visits_day_count
                else:
                    weekday = np.add(weekday, visits_day)
                    weekday_count = np.add(weekday_count, visits_day_count)
                    
            # add daily visits to weekly total
            if days_before == 0:
                visits_week = visits_day
                visits_week_count = visits_day_count
            else:
                visits_week = np.add(visits_week, visits_day)
                visits_week_count = np.add(visits_week_count, visits_day_count)
                
            # increment day
            day = day - one_day
        
        # Store weekly data (first month only)
        if months_before == 0:
            visits_week_ave = np.divide(visits_week, visits_week_count)
            preds = preds.assign(**{"visits_W-"+str(weeks_before) : visits_week_ave})
        
        # add weekly visits to monthly total
        if weeks_before == 0:
            visits_month = visits_week
            visits_month_count = visits_week_count
        else:
            visits_month = np.add(visits_month, visits_week)
            visits_month_count = np.add(visits_month_count, visits_week_count)
        
    # Store monthly data
    visits_month_ave = np.divide(visits_month, visits_month_count)
    preds = preds.assign(**{"visits_M-"+str(months_before) : visits_month_ave})
    
    # Store average visits on this day of the week for each month
    weekday_ave = np.divide(weekday, weekday_count)
    preds = preds.assign(**{"atw_M-"+str(months_before) : weekday_ave})
    
    # Get overall average for this weekday
    if months_before == 0:
        weekday_total = weekday
        weekday_total_count = weekday_count
    else:
        weekday_total = np.add(weekday_total, weekday)
        weekday_total_count = np.add(weekday_total_count, weekday_count)
        
weekday_total_ave = np.divide(weekday_total, weekday_total_count)
preds = preds.assign(atw_all = weekday_total_ave)


# ### Trends

# In[11]:


# prep quarterly variables
first_three_months = preds["visits_M-0"].add(preds["visits_M-1"]).add(preds["visits_M-2"])
next_three_months = preds["visits_M-3"].add(preds["visits_M-4"]).add(preds["visits_M-5"])

# assign
preds = preds.assign(trend_week = preds["visits_W-0"].div(preds["visits_W-1"]))
preds = preds.assign(trend_month = preds["visits_M-0"].div(preds["visits_M-1"]))
preds = preds.assign(trend_3_months = first_three_months.div(next_three_months))

#preds

# get rid of any inf values from all that division by zero
# XGB can still handle nan values
preds.replace([np.inf, -np.inf], np.nan, inplace=True)


# In[12]:


# combine all of the above with stores data
stores = pd.concat([stores, preds], axis=1)


# ## Holidays
# 
# ** There are no holidays in the test set except golden week. Forget about this. **
# 
# If prediction date is a holiday, assign the average fractional increase (or decrease) for each store and each genre during a holiday. Otherwise, fill with unity.
# 
# Actually this is not needed for the test set, but may help our model avoid mistakes by learning wrong things from holiday days in the training data.

# In[13]:


# holidays = pd.read_csv("../input/date_info.csv",
#                       parse_dates = ["calendar_date"],
#                       index_col = ["calendar_date"])

# day_of_week = holidays.loc[prediction_date, "day_of_week"]
# holiday = holidays.loc[prediction_date, "holiday_flg"]


# In[14]:


# # sum visitors for each store during non-holiday and during holiday
# # to get the "effect" of a holiday for this store

# # quick function for averages
# def average(alist):
#     return sum(alist) / max( len(alist), 1 )

# if holiday:
#     print("Calculating Holiday Info...")
    
#     stores_average = []
    
#     print("Scanning over", len(stores.index), "stores")
#     count = 0    
#     for store in stores.index:
#         normal_visitors = []
#         holiday_visitors = []
        
#         count += 1
#         if count % 100 == 0:
#             print ("Scanned", count, "stores")
        
#         # scan all days
#         day = prediction_date
#         while day > datetime(2016,1,1):
#             day -= one_day
            
#             if (day,store) in visits.index:
#                 if (holidays.loc[day, "holiday_flg"]):
#                     holiday_visitors.append(visits.loc[(day,store)][0])
#                 else:
#                     normal_visitors.append(visits.loc[(day,store)][0])

#         # calculate the average increase in visitors during a holiday
#         if len(holiday_visitors) > 1 and len(normal_visitors) > 1:
#             stores_average.append( average(holiday_visitors) / average(normal_visitors) )
#         else:
#             stores_average.append(1.)
    
#     stores = stores.assign(holiday_effect = stores_average)
    
# #else:
# #    stores = stores.assign(holiday_effect = np.ones(len(stores)))


# In[15]:


# # get average increase for genre
# if holiday:
    
#     genre_name = []
#     genre_effect = []

#     for genre in stores["air_genre_name"].unique():
#         this_genre = stores[ stores["air_genre_name"] == genre ]
#         genre_name.append(genre)

#         holiday_visitors = []
#         normal_visitors = []
        
#         # scan all stores
#         for store in this_genre.index:
#             # scan all days
#             day = prediction_date
#             while day > datetime(2016,1,1):
#                 day -= one_day
                
#                 if (day, store) in visits.index:
#                     if (holidays.loc[day, "holiday_flg"]):
#                         holiday_visitors.append(visits.loc[(day,store)][0])
#                     else:
#                         normal_visitors.append(visits.loc[(day,store)][0])

#         # calculate the average increase in visitors during a holiday
#         if len(holiday_visitors) > 1 and len(normal_visitors) > 1:
#             genre_average = average(holiday_visitors) / average(normal_visitors)
#         else:
#             genre_average = 1.
            
#         print(genre, this_genre.shape[0], "Holiday effect:", genre_average)
#         genre_effect.append(genre_average)
        
    
#     # Assign to stores
#     genre_effects = pd.DataFrame({"genre":genre_name, "effect":genre_effect}).set_index("genre")
#     stores = stores.assign(holiday_genre_effect = stores["air_genre_name"].map(lambda x: genre_effects.loc[x,"effect"]))
            
# #else:
# #    stores = stores.assign(holiday_genre_effect = np.ones(len(stores)))


# ## Average info from other stores
# 
# Get the effective customer multiplier for various situations from all our data. This gives some hint for how to treat each genre and location.
# 
# For average over:
# * Stores in this location (todofuken AND kushi)
# * Stores in this genre
# 
# ... we want the:
# * Multiplier for this month
# * Multiplier for this day of the week

# In[16]:


# Combine low-population labels
def get_low_pop(series):
    to_drop = []
    for name in series.index:
        if series[name] < 5:
            to_drop.append(name)
    return to_drop
        
genres_to_drop = get_low_pop(stores["air_genre_name"].value_counts())
kushi_to_drop = get_low_pop(stores["kushi"].value_counts())

stores["air_genre_name"] = stores["air_genre_name"].map(lambda x: x if x not in genres_to_drop else "genre_dropped")
stores["kushi"] = stores["kushi"].map(lambda x: x if x not in kushi_to_drop else "kushi_dropped")


# In[17]:


# calculate the average stuff over all stores

def normalize(li):
    """
    Simple function to normalize a list.
    """
    average = np.sum(li) / len(li)
    li = np.divide(li, average)
    return li


def calc_effects(col):
    """
    Takes in a column property of "stores", e.g. area or genre,
    and calculates the normalized effect of each month and weekday.
    """
    
    property_name = col.name
    
    # prepare dataframe to hold the effects of this property
    effects = pd.DataFrame(index = col.unique(), columns = ["months", "weekdays"])
    
    # For each label of property
    for label in effects.index:
        
        print("Calculating for", label)
        
        # get sublist of stores with this label
        store_list = stores[ stores[property_name] == label ].index
        
        # prepare temp variables for speed (DF access is slow)
        months = np.zeros(12)
        months_count = np.zeros(12)
        weekdays = np.zeros(7)
        weekdays_count = np.zeros(7)

        # Loop over all dates
        dates = pd.date_range(datetime(2016,1,1), prediction_date - one_day)
        for date in dates:

            # Count number of visitors, and number of stores open
            total_visitors = 0
            total_open = 0
            for store in store_list:
                if (date,store) in visits.index:
                    total_open += 1
                    total_visitors += visits.loc[(date,store)]["visitors"]

            # fill for averages
            months[date.month - 1] += total_visitors
            months_count[date.month - 1] += total_open
            weekdays[date.weekday()] += total_visitors
            weekdays_count[date.weekday()] += total_open

        # Calculate normalized effects
        months_ave = normalize( np.divide(months, months_count) )
        weekdays_ave = normalize( np.divide(weekdays, weekdays_count) )
        
        # Store info to DF
        effects.loc[label,"months"] = months_ave
        effects.loc[label, "weekdays"] = weekdays_ave
        
    # return effects DF
    return effects


def append_to_stores(effects, col):
    stores[str(col.name + "_monthly_effects")] = col.map(lambda x: effects.loc[x, "months"][today.month-1])
    stores[str(col.name + "_weekday_effects")] = col.map(lambda x: effects.loc[x, "weekdays"][today.weekday()])
        
        


# In[20]:


# Run calculation & save
if (run_calculation):
    kushi_effects = calc_effects(stores["kushi"])
    todofuken_effects = calc_effects(stores["todofuken"])
    genre_effects = calc_effects(stores["air_genre_name"])
    kushi_effects.to_csv("histograms/kushi_effects.csv")
    todofuken_effects.to_csv("histograms/todofuken_effects.csv")
    genre_effects.to_csv("histograms/genre_effects.csv")

# Else read from old calculation
else:
    kushi_effects = pd.read_csv("histograms/kushi_effects.csv", index_col=0)
    todofuken_effects = pd.read_csv("histograms/todofuken_effects.csv", index_col=0)
    genre_effects = pd.read_csv("histograms/genre_effects.csv", index_col=0)
    
    # Unfortunately it seems pandas can't store / read lists inside of cells properly
    # ugly hack needed...
    kushi_effects["months"] = kushi_effects["months"].map( lambda x: np.fromstring(x[2:-1], sep=" ") )
    todofuken_effects["months"] = todofuken_effects["months"].map( lambda x: np.fromstring(x[2:-1], sep=" ") )
    genre_effects["months"] = genre_effects["months"].map( lambda x: np.fromstring(x[2:-1], sep=" ") )
    kushi_effects["weekdays"] = kushi_effects["weekdays"].map( lambda x: np.fromstring(x[2:-1], sep=" ") )
    todofuken_effects["weekdays"] = todofuken_effects["weekdays"].map( lambda x: np.fromstring(x[2:-1], sep=" ") )
    genre_effects["weekdays"] = genre_effects["weekdays"].map( lambda x: np.fromstring(x[2:-1], sep=" ") )

append_to_stores(kushi_effects, stores["kushi"])
append_to_stores(todofuken_effects, stores["todofuken"])
append_to_stores(genre_effects, stores["air_genre_name"])


# In[ ]:


# Assign min, med, max for related stores (in this area & genre)
gb = stores.reset_index().groupby(["air_genre_name","kushi"])
recent_visits = visits.reset_index().copy()
recent_visits = recent_visits[ recent_visits["visit_date"] > prediction_date - timedelta(weeks=15) ]
recent_visits = recent_visits.set_index(["air_store_id"])
stores["genarea_min"] = 0
stores["genarea_med"] = 0
stores["genarea_max"] = 0

# loop over all genre-area pairs
for name, group in gb:
    related_stores = group["air_store_id"].values

    visitors = []
    for store in related_stores:
        visitors.extend( recent_visits.loc[store,"visitors"].values )

    min_ = np.percentile(visitors, 16)
    med_ = np.percentile(visitors, 50)
    max_ = np.percentile(visitors, 84)
    
    for store in related_stores:
        stores.loc[store,"genarea_min"] = min_
        stores.loc[store,"genarea_med"] = med_
        stores.loc[store,"genarea_max"] = max_


# ## Golden Week
# * Calculate average effect of golden week on that store's
#   * genre
#   * area (todofuken, kushi.
#   
# This is important as it will be in the test set. We should probably try and calculate the average effect of golden week for each day inside of golden week. Note also that this might conflict with general holidays information. **For now, placeholder function is in place.**
# 

# In[158]:


# GW is 4/29 -> 5/5
# in 2016 this is FRI ~ THU (MON is a "work day")
# in 2017 this is SAT ~ FRI (MON, TUE are "work days")

def check_golden_week(the_date):
    if (the_date >= datetime(2016,4,29)) and (the_date < datetime(2016,5,6)):
        return 1
    elif (the_date >= datetime(2017,4,29)) and (the_date < datetime(2017,5,6)):
        return 1
    else:
        return 0
    
if check_golden_week(today):
    stores["gw_effect"] = stores["todofuken"].map(lambda x: 0.8 if x == "Tōkyō-to" else 1.0)


# # One-hotting
# Note that low-populated labels have already been dropped earlier to avoid overfitting.

# In[87]:


# Split numeric and text data
X_numeric = stores.select_dtypes(exclude=['object']).copy()
X_text = stores.select_dtypes(include=['object']).copy()

# get one-hot columns from text data
X_onehot = pd.get_dummies(X_text)

# recombine
X_final = pd.concat([X_numeric, X_onehot], axis=1)

# save
X_final.to_csv("tabular_data/" + today.strftime("%Y_%m_%d") + ".csv")
print("Final predictors with shape", X_final.shape, "saved to CSV")

