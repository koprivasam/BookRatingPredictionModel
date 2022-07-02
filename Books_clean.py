#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Compute model Libraties
from sklearn import metrics
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_squared_error


# In[2]:


books = pd.read_csv(r"C:\Users\kopri\OneDrive\Documents\DSTI\Python_lab\Project\Hrvoje\Book_Hrv.csv", on_bad_lines = "skip")


# In[3]:


pwd


# In[4]:



print ( books.head() )


# In[5]:


# Dataset composition (first and last 4 rows)
books.fillna(4)


# In[6]:


# Number of records in the dataset is 11127
total_records = len(books['bookID'])
print( "Record cnt: {}".format( total_records ) )


# In[8]:


# Statistical analysis of dataset
books.describe()


# In[9]:


#Description of dataser:Datatypes,Count,Column
books.info() 


# In[10]:


# Check for null values
books.isna().sum()


# In[11]:


# First clean parsing errors (publishers separated with comma)
# Need to replace, but maybe leave value as array, this will help with title duplicates, they can also be stored in array
# This is fucking mess!!!!
for i in range( len( books.publisher ) ):
    if type( books.publisher[i] ) != 'str':
        # books[i].__setitem__( 'publisher', 'none' )
        print( books['publisher'][i] )
        break


# In[12]:


# Drop the columns isbn and isbn13, because it has no impact on the target variable "Average rating"
books = books.drop(["isbn","isbn13"],axis=1)


# In[13]:


# Pick the language_code as series and then convert it into a Set
# Join the Language codes: en-US, en-GB,en-CA to eng
list_of_lang = set(books["language_code"])
list_of_lang


# In[14]:


# Looking for distinct values for language_code, and when there are different codes with the same meaning we merge in one.
books.language_code.value_counts()


# In[15]:


# Form in the columm an uniformity  'language_code' by looping through the list
for i in list_of_lang:
    if i == "English":
        books.loc[books['language_code'] == i,"language_code"] = "eng"
    elif i == "en-US":
        books.loc[books['language_code'] == i,"language_code"] = "eng"
    elif i == "en-CA":
        books.loc[books['language_code'] == i,"language_code"] = "eng"  
    elif i == "en-GB":
        books.loc[books['language_code'] == i,"language_code"] = "eng"
    elif i == "French":
       books.loc[books['language_code'] == i,"language_code"] = "fre"


# In[16]:


# Explore the distinct values for language_code after merging same Language code
books.language_code.value_counts()


# In[17]:


# Look for null values in publication date
books.loc[books.publication_date.isna(),:]


# In[18]:


# Number of DataFrame rows and columns (including NA elements)
books.shape


# In[19]:


#mozda OD PRIJ DRUGI KOD UPOTRIJBIT ZA TO
# Determine records with the same title but without null publication date. To replace the null date. 
books[books['title'].str.contains('Montaillou', na=False)]


# In[20]:


# Clean the column "publication_date"
#Remove the records that have publication_date null
books = books.loc[books['publication_date'].notna(),:]


# In[21]:


# Clean incorrect date value 
books.loc[books['publication_date']=="1/1/1900 0:00","publication_date"]


# In[22]:


# Convert object/string to date type
books.loc[:,"Date"] = pd.to_datetime(books['publication_date'])


# In[23]:


# Select month from the date  
books['month'] = books['Date'].dt.month


# In[24]:


# Select year from the date
books['year'] = books['Date'].dt.year


# In[25]:


# View the data types
books.dtypes


# In[26]:


# View the dataframe after data cleaning
books.head()


# In[27]:


# Drop the columns "publication_date" and "Date", because it has been transformed to year and month 
books = books.drop(["publication_date","Date"],axis=1)


# In[28]:


# Clean the column "Authors"


# In[29]:


# Form a function to identify "/,C" in the column "Authors"
def Update_strings(name):
    if name.find("/") != -1:
      author_name = name.split("/")[0]
    elif name.find("(") != -1:
      author_name = name.split("(")[0]  
    else:
      author_name = name
    return author_name
   


# In[30]:


# Transform the "author" column information to have only one author by picking the first name before the '/' or '('
books.loc[:,"authors"] = books.loc[:,"authors"].apply(Update_strings)


# In[31]:


#Clean the column "Title"


# In[32]:


# Form function to replace special characters such as "!$%^&*" with white space and also '&' with and
def processString(txt):
  specialCharacters = "!$%^&*¡¿" 
  for specialChar in specialCharacters:
    if specialChar == '&':
        txt = txt.replace('&',' and ') 
    else:
        txt = txt.replace(specialChar, ' ')
  return txt


# In[33]:


# Update the title column data to bring uniformity to the column "title"
books.loc[:,"New_title"]= books.loc[:,"title"].apply(processString)


# In[34]:


# Replace double white spaces with single white space
books['New_title'] = books['New_title'].str.replace('  ',' ')


# In[35]:


# Form a new title column name: new_title to pick the first name of the title
books.loc[:,"New_title"]= books.loc[:,"New_title"].apply(Update_strings)


# In[36]:


# The number of counts for each title
books.title.value_counts()


# In[37]:


# Check for null values
books.isna().sum()


# In[38]:


# Look the "title" transformation
books[["title","New_title"]]


# In[39]:


# View null values of publisher,text_reviews_count,language_code 
books.loc[books.language_code.isna(),:]


# In[40]:


# Drop records for null values of "text_reviews_count"
books = books.loc[books['text_reviews_count'].notna(),:]


# In[41]:


# Drop records for null values of "language_code"
books = books.loc[books["language_code"].notna(),:]


# In[42]:


# Drop the columns publisher from the dataframe
books = books.drop(["publisher"],axis=1)


# In[43]:


# View the data types for the columns
books.dtypes


# In[44]:


# Replace the "title" column with "new_title" and drop the colum
books.loc[:,"title"]= books.loc[:,"New_title"].apply(Update_strings)
books = books.drop(["New_title"],axis=1)
books


# In[45]:


plt.subplots(figsize=(8,8))
plt.boxplot(books.average_rating)
plt.xticks([1], ["average_rating"])
plt.title("Boxplot of count of average rating")


# In[46]:


### This is the histogram for 'average_rating', It is target for the modelling.
### Very concentrated around 3.5 - 4.5

books.hist(column="average_rating",figsize=(6,7))


# In[47]:


# Distributional ratings 
sns.kdeplot(books['average_rating'], shade = True)
plt.title('Rating Distribution\n')
plt.xlabel('Rating')
plt.ylabel('Frequency')


# In[48]:


# Counts average ratings of top 9
sns.barplot(x=books['average_rating'].value_counts().head(9).index,y=books['average_rating'].value_counts().head(9))
plt.title('Number of Books Each Rating Received\n')
plt.xlabel('Average Ratings')
plt.ylabel('Frequency')
# plt.xticks(rotation=50)


# In[49]:


# Authors with top rated books
authors = books.nlargest(7, ['ratings_count']).set_index('authors')
sns.barplot(y=authors['ratings_count'], x=authors.index, ci = None, hue = authors['title'])
plt.xlabel('Total Ratings')


# In[50]:


# This barplot shows us the most reviewed books in the dataset, with the first Twilight taking the number 1 spot by a mile.

MostReviewedBooks = books.nlargest(10, ['ratings_count']).set_index('title')['ratings_count']
fig, ax = plt.subplots(figsize=(10, 7))
sns.barplot(MostReviewedBooks, MostReviewedBooks.index)


# In[51]:


# Look for a mean(average_rating) from dataframe books_df with group by "title" 
title_avg_rat_df = books.groupby(["title"])["average_rating"].mean()
# Convert the groupby result into a dataframe
title_df = title_avg_rat_df.to_frame().reset_index()
# Sort by title and average_rating
title_df.sort_values(["average_rating","title"],ascending=[False,True]).head(10)


# In[52]:


most_rated = books.sort_values(by="ratings_count", ascending = False).head(10)

most_rated_titles = pd.DataFrame(most_rated.title).join(pd.DataFrame(most_rated.ratings_count))
most_rated_titles


# In[53]:


# Top 5 languages
books['language_code'].value_counts().head(5).plot(kind = 'pie', autopct='%1.2f%%', figsize=(7, 8)).legend()


# In[54]:


# Find the relationship between Average_rating and num_pages with repect to the language code
df_lang=books.loc[(books['language_code'] == 'spa') | (books['language_code'] == 'fre') | (books['language_code'] == 'eng') | (books['language_code'] == 'ger')]

sns.lmplot(x="average_rating", y="ratings_count",hue = "language_code",data=df_lang);


# In[55]:


#We see a high correlation between the ratings_count and the text_reviews_count (~ 81%). 
#From this we can conclude that when a person writes a review he/she will most likely also rate the book itself.print(" Displaying the correlation between the features in following heatmap:")
sns.heatmap(data=books.corr(),
            linewidths=0.25, square=True,
            linecolor="black", annot=True)


# In[56]:


books.hist(column="bookID",figsize=(6,7))


# In[57]:


#3. Model Training and Evaluation
#Transform categorical data to numeric


# In[58]:


books


# In[59]:


# Remove titles that are non ASCII characters
books = books.loc[~((books['title'].str.contains(r'[^\x00-\x7F]')) & ((books['language_code'] == 'jpn') | (books['language_code'] == 'zho')))]


# In[60]:


# Select the top 4 counts for "language_code"
processed = books.loc[(books['language_code'] == 'eng') | (books['language_code'] == 'fre') | (books['language_code'] == 'spa') | (books['language_code'] == 'ger')]


# In[61]:


# Encoding the "language_code" column by transforming it to 4 columns corresponding to each of its values
processed = pd.concat([processed,pd.get_dummies(processed.language_code)],axis=1).drop(columns="language_code")


# In[62]:


# View the dataframe to be used for machine learning
# It has 10996 records and 12 columns
processed


# In[63]:


# Convert 'authors' column to numeric
processed['authors'] = pd.factorize(processed['authors'])[0]


# In[64]:


# Convert 'title' column to numeric
processed['title'] = pd.factorize(processed['title'])[0]


# In[65]:


# View the dataframe again now all columns are numeric which is good for the model to be trained.
processed


# In[66]:


# Copy the datafram to csv
processed.to_csv('book_processed_cleaned.csv', index=False)
books.to_csv('books_df_cleaned.csv', index=False)


# In[67]:


# Split the data into Train and Test


# In[68]:


# Split the data into train and test for the model
from sklearn.model_selection import train_test_split 


# In[69]:


# the size of the test is 20% and the train is 80%
df_train, df_test = train_test_split(processed,test_size = 0.2)


# In[70]:


# Total records for dataset
len(processed)


# In[71]:


# Total records for training the model
len(df_train)


# In[72]:


# Total records for testing the model
len(df_test)


# In[73]:


# Check the data is ramdomly selected
print(processed.average_rating.mean())
print(df_train.average_rating.mean())
print(df_test.average_rating.mean())


# In[74]:


# Review the features and the target to be used for the machine learning
df_train.columns


# In[75]:


#C get the values of the columns for the training data
X_train = df_train.loc[:,['title', 'authors', 'num_pages', 'ratings_count', 'text_reviews_count','year','month','eng','fre','ger','spa']].values
y_train = df_train.average_rating.values.round(1)


# In[76]:


# View the values for X_train
print(X_train)


# In[77]:


# View the values y_train
print(y_train)


# In[78]:


# get the values of the columns for the test data
X_test = df_test.loc[:,['title', 'authors', 'num_pages', 'ratings_count', 'text_reviews_count','year','month','eng','fre','ger','spa']].values
y_test = df_test.average_rating.values.round(1)


# In[79]:


# View the values for X_test
print(X_test)


# In[80]:


# View the values y_test
print(y_test)


# In[81]:


# Train the Model


# In[82]:


# importing module: Liner Regression
from sklearn.linear_model import LinearRegression


# In[83]:


# creating an object of LinearRegression class
lr_model = LinearRegression()


# In[84]:


# here we train the model on the training data
lr_model.fit(X=X_train, y=y_train)


# In[85]:


#Predict the Target Values


# In[86]:


y_test_predicted = lr_model.predict(X_test)


# In[87]:


# Approximate to 1 decimal place
y_test_predicted = y_test_predicted.round(1)


# In[88]:


# Now compare the actual output values for y_test with the predicted values.
pred = pd.DataFrame({'Actual': y_test.tolist(), 'Predicted': y_test_predicted.tolist()}).head(21)
pred.head(9)


# In[89]:


#Evalute the Performance of the Model


# In[90]:


# Check how accurate is the prediction
(y_test_predicted == y_test).sum()/len(y_test)


# In[91]:


# Compute the accuracy of the model
from sklearn import metrics
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_squared_error


# In[92]:


# Evalute the performance of the Algorithm
score=r2_score(y_test,y_test_predicted)
meanError = mean_squared_error(y_test,y_test_predicted)
meansquardError = np.sqrt(mean_squared_error(y_test,y_test_predicted))


# In[93]:


print('R-square is',score)
print('mean_squard_error is',meanError)
print('root_mean_squared error is',meansquardError)


# In[94]:


print('Intercept: \n', lr_model.intercept_)


# In[95]:


print('Coefficients: \n', lr_model.coef_)


# In[96]:


#Compare the Model


# In[97]:


# importing module: Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# In[98]:


rf_model = RandomForestRegressor()


# In[99]:


rf_model.fit(X=X_train,y=y_train)


# In[100]:


y_test_predicted_rf = rf_model.predict(X_test)


# In[101]:


# Approximate to 1 decimal place
y_test_predicted_rf = y_test_predicted_rf.round(1)


# In[102]:


# Now compare the actual output values for y_test with the predicted values.
preds = pd.DataFrame({'Actual': y_test.tolist(), 'Predicted': y_test_predicted_rf.tolist()}).head(25)
preds.head(10)


# In[103]:


# Check how accurate is the prediction
(y_test_predicted_rf == y_test).sum()/len(y_test)


# In[104]:


# Evalute the performance of the Algorithm
score=r2_score(y_test,y_test_predicted_rf)
meanError = mean_squared_error(y_test,y_test_predicted_rf)
meansquardError = np.sqrt(mean_squared_error(y_test,y_test_predicted_rf))


# In[105]:


print('R-square is',score)
print('mean_sqrd_error is',meanError)
print('root_mean_squared error is',meansquardError)


# In[106]:


#Conclusion
#The lower value of root_mean_squared error(RMSE) closer to 0 preferred the better model performance, conversely, the higher value of R-square(R2) closer to 1 shows that the regression line fits the data well and the model performance is better. Also, Mean square error (MSE) is the average of the square of the errors. The larger the number the larger the error which means there is bigger variation between the test result and the predicted result. Therefore, the Random Forest Regressor model is far better than Liner Regression model.


# In[107]:


#Compare the Model 2


# In[108]:


#Conclusion
#The lower value of root_mean_squared error(RMSE) closer to 0 preferred the better model performance,
#conversely, the higher value of R-square(R2) closer to 1 shows that the regression line fits the data well and the model performance is better.
#Also, Mean square error (MSE) is the average of the square of the errors.
#The larger the number the larger the error which means there is bigger variation between the test result and the predicted result.
#Therefore, the Random Forest Regressor model is far better than Liner Regression model.


# In[ ]:




