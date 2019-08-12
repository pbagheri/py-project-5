# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 20:34:05 2017

@author: Payam
"""

# This file load all datasets and explores them, cleans them and prepares them
# for further use. 

#****************************
target = pd.read_sas('C:/Users/Payam/Documents/0_MetroC/0_Final_Projects/Projectj_1/target.sas7bdat', encoding = 'utf-8')
nanputter(target) # nanputter function replaces '.' with np.nan. '.' was created
                  # wherever there was missing values when SAS data was loaded in python
target.head(200)
target.tail()
target.isnull().any() # missing values in id
target.shape # (462753, 2)
len(target[target.id.isnull()]) # 187 missing id values
len(target[target.id.notnull()]) # 462566
target.groupby('gold').size()

#****************************            
customer = pd.read_csv('C:/Users/Payam/Documents/0_MetroC/0_Final_Projects/Projectj_1/csv/customers.csv')
customer.head()
nanputter(customer)
customer.isnull().any() # customer has missing in some columns
customer.income.isnull().sum()
len(customer.income) # income is very sparse, but we will use it by doing binning

customer.shape # (462680, 15)
customer.id.unique().shape # no duplicates in id column. 462680 unique id values.

[(i, customer[i].dtype) for i in customer] # looking at data types of each column

#****************************
spend = pd.read_sas('C:/Users/Payam/Documents/0_MetroC/0_Final_Projects/Projectj_1/spend.sas7bdat', encoding = 'utf-8')
spend.head()
spend.spend_type.unique()
nanputter(spend)
spend.isnull().any() # no missing
spend.shape # (13495672, 4)

len(spend.id.unique()) # 462680 unique id values in spend data

chartonum(spend,'spend_type')
spend.spend_type.unique()

spend.groupby('spend_type').count()

spend.groupby('id').count().max() # 2056
spend.groupby('id').count().min() # 1

dmax = spend.groupby('id')['spend_day'].max()
datemax = pd.DataFrame({'datemax': dmax})
datemax.reset_index(inplace=True)
datemax.head()

spend = spend.merge(datemax, on = 'id', how = 'inner')

#****************************
coupon= pd.read_sas('C:/Users/Payam/Documents/0_MetroC/0_Final_Projects/Projectj_1/coupon.sas7bdat', encoding = 'utf-8')
coupon.head()
nanputter(coupon)
coupon.isnull().any() # no missing"
coupon.shape # (163610, 3)
len(coupon.id.unique()) # only 80288 unique id values in coupon data

coupon_unq = coupon.groupby('id', as_index = False)['amount'].sum()

#****************************
service = pd.read_sas('C:/Users/Payam/Documents/0_MetroC/0_Final_Projects/Projectj_1/service.sas7bdat', encoding = 'utf-8')
service.head()
nanputter(service)
service.isnull().any() # no missing
service.shape # (468117, 3)
len(service.id.unique()) # 121072 unique id values are in service data

service_unq = service.groupby('id', as_index = False)['service_count'].sum()

#****************************
demo = pd.read_sas('C:/Users/Payam/Documents/0_MetroC/0_Final_Projects/Projectj_1/demo_real.sas7bdat', encoding = 'utf-8')
demo.head()
nanputter(demo)
demo.isnull().any()
sum(demo.isnull().any())  # no missing
demo.shape # (566109, 144)

[(i, demo[i].dtype) for i in demo] # looking at data types of each column
# all data are numerical in demo except the postcode
#****************************
mydm = pd.read_sas('C:/Users/Payam/Documents/0_MetroC/0_Final_Projects/Projectj_1/mydm.sas7bdat', encoding = 'utf-8')
mydm.head()
nanputter(mydm)
mydm.isnull().any()
mydm.shape # (517762, 144)
sum(mydm.isnull().any()) # all columns have missing values

[(i, mydm[i].dtype) for i in mydm] # looking at data types of each column
# all data are numerical in demo except the postcode

# target and mydm are merged (via the customer dataset) to be able to check
# correlation between features in mydm and target
mydmext = target.merge(customer[['id', 'postcode']], on='id').merge(mydm, on='postcode')

mydmextcols = list(mydmext.columns)

# the following loop creates a dummy feature for features in mydm that have
# null values and checks the correlation between these features and the target
highcormydm = []
for i in mydmextcols[3:]:
    if mydmext[i].isnull().any():
        dummy = mydmext[i].notnull().astype(int)
        if abs(pearsonr(dummy, mydmext['gold'])[0]) > 0.02:
            highcormydm.append(i)

highcormydm
len(highcormydm) 

# for no feature is mydm, the dummy has a correlation higher than 0.02 with 
# the target therefore we are not goimg to use mydm. Also, some other inspections
# showed that mydm and demo have a lot of info in common, so we only use demo 
                 

"""
# in the following loop, missing values in each feature is replaced with the
# the mean of that feature and the correlation with target is checked
mydmextcop = mydmext.copy() 
highcormydm = []
for i in mydmextcols[3:]: # 'id', ''postcode' and 'gold' are left aside
    if mydmextcop[i].isnull().any():
        mydmextcop[i].replace(np.nan, mydmextcop[i].mean(), inplace=True)
        if abs(pearsonr(mydmextcop[i], mydmextcop['gold'])[0]) > 0.05:
            highcormydm.append(i)
            
len(highcormydm) # several columns have a correlation higher than 0.05 with
                 # the target when null values are replaced with the mean
                 # of the target. These columns are kept in the final dataset
                 # that is used for the rest of the analysis.

highcormydm.append('postcode') # 'postcode' is added to the list of column names
len(highcormydm)

mydmgood = mydm[highcormydm]
mydmgood.head()
"""