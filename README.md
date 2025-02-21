# Financial-Sentiment-Analysis-of-Meta-Platforms-Inc.
# Libraries #
import pandas as pd
import matplotlib.pyplot as plt from wordcloud import WordCloud import re
import numpy as np
from sec_api import ExtractorApi
from nltk.sentiment import SentimentIntensityAnalyzer import nltk
from string import digits
from nltk.tokenize import word_tokenize from wordcloud import WordCloud, STOPWORDS from nltk.corpus import stopwords nltk.download('stopwords') nltk.download('punkt') nltk.download('vader_lexicon')
#
# Extracting Text #
# Year of Filing - Input by user in terminal. Changes which filing is used from the below links and the year label in the visuals
year_10k = int(input('Input Desired Year: '))
# Meta 10K SEC File Links
url_2022 =
"https://www.sec.gov/ix?doc=/Archives/edgar/data/0001326801/00013268012300 0013/meta-20221231.htm"
url_2021 =
"https://www.sec.gov/ix?doc=/Archives/edgar/data/0001326801/00013268012200 0018/fb-20211231.htm"
url_2020 =
"https://www.sec.gov/ix?doc=/Archives/edgar/data/0001326801/00013268012100 0014/fb-20201231.htm"
url_2019 =
"https://www.sec.gov/ix?doc=/Archives/edgar/data/0001326801/00013268012000 0013/fb-12312019x10k.htm"
url_2018 =
"https://www.sec.gov/Archives/edgar/data/1326801/000132680119000009/fb-123 12018x10k.htm"
url_2017 =
"https://www.sec.gov/Archives/edgar/data/1326801/000132680118000009/fb-123 12017x10k.htm"
url_2016 =
"https://www.sec.gov/Archives/edgar/data/1326801/000132680117000007/fb-123 12016x10k.htm"
url_2015 =
"https://www.sec.gov/Archives/edgar/data/1326801/000132680116000043/fb-123 12015x10k.htm"
url_2014 =
"https://www.sec.gov/Archives/edgar/data/1326801/000132680115000006/fb-123 12014x10k.htm"
url_2013 =
"https://www.sec.gov/Archives/edgar/data/1326801/000132680114000007/fb-123 12013x10k.htm"
url_2012 =
"https://www.sec.gov/Archives/edgar/data/1326801/000132680113000003/fb-123 12012x10k.htm"
test_url =
"https://www.sec.gov/Archives/edgar/data/1318605/000156459021004599/tsla-1 0k_20201231.htm"
API Key to connect to the SEC API APIKEY = "INSERT YOUR API KEY HERE"
extractorApi = ExtractorApi(APIKEY)
# Meta 10K URL
if year_10k == 2022: url_10k = url_2022
if year_10k == 2021: url_10k = url_2021
if year_10k == 2020: url_10k = url_2020
if year_10k == 2019: url_10k = url_2019
if year_10k == 2018: url_10k = url_2018
if year_10k == 2017: url_10k = url_2017
if year_10k == 2016: url_10k = url_2016
if year_10k == 2015: url_10k = url_2015
if year_10k == 2014: url_10k = url_2014
if year_10k == 2013: url_10k = url_2013
if year_10k == 2012: url_10k = url_2012
if year_10k == 2001: url_10k = test_url
# get the standardized and cleaned text of section 7 (MD&A) text_format_10k = extractorApi.get_section(url_10k, "7", "text")
# Cleaning Data #
#Remove unwanted characters
clean_10k = re.sub('\W+',' ', text_format_10k)
# Convert data to lower case clean_10k=clean_10k.lower()
# Remove all the digits
remove_digits = str.maketrans('', '', digits) clean_10k = clean_10k.translate(remove_digits)
# Remove stop words
pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*')
clean_10k = pattern.sub('', clean_10k)
# Remove single letters
tmp = re.sub(r'\b\w\b', ' ', clean_10k) clean_10k = re.sub(r'\s{2,}', ' ', tmp).strip()
# Remove table words and roman numbers
clean_10k = clean_10k.replace('table_start', '') clean_10k = clean_10k.replace('table_end', '') clean_10k = clean_10k.replace('ii', '')
# Tokenization of filing
token_10k = word_tokenize(clean_10k)
# Checks if any data is found in the filing (bug checking) if clean_10k == "":
print("No data found in filing") else:
print("Data Found!")#clean_10k)
# Calculate most frequent words in the filing frequentWords = nltk.FreqDist(token_10k) frequentWords.tabulate(10)
# Get stopwords
stopwords = set(STOPWORDS)
# WordCloud Generation
wordcloud = WordCloud(stopwords=stopwords, background_color="black", max_words=100, ).generate(clean_10k)
# WordCloud Visualization fig=plt.figure(figsize=(15, 8)) plt.imshow(wordcloud, interpolation='bilinear') plt.axis("off")
plt.title(str(year_10k) + ' META WordCloud') plt.show()
#Sentiment Score for 10K Sec 7
sentiment = SentimentIntensityAnalyzer() sentiment.polarity_scores(clean_10k)
# Save sentiment scores in dataframe and plot it df_10k = sentiment.polarity_scores(clean_10k) df_10k = pd.DataFrame(df_10k, index=[0])
df_10k = df_10k[["neg", "pos", "neu"]] df_10k.plot.bar(align='edge', width=0.7) plt.title(str(year_10k) + ' META Sentiment Analysis') plt.xlabel('Text')
plt.ylabel('Sentiment Score') print(df_10k)
plt.show()
Python Script 2: Text Mining for All MD&As Combined
#
# Libraries #
import pandas as pd
import matplotlib.pyplot as plt from wordcloud import WordCloud import re
from sec_api import ExtractorApi
from nltk.sentiment import SentimentIntensityAnalyzer import nltk
from string import digits
from nltk.tokenize import word_tokenize from wordcloud import WordCloud, STOPWORDS from nltk.corpus import stopwords nltk.download('stopwords') nltk.download('punkt') nltk.download('vader_lexicon')
#
# Extracting Text #
# Meta 10K SEC File Links
url_2022 =
"https://www.sec.gov/ix?doc=/Archives/edgar/data/0001326801/00013268012300 0013/meta-20221231.htm"
url_2021 =
"https://www.sec.gov/ix?doc=/Archives/edgar/data/0001326801/00013268012200 0018/fb-20211231.htm"
url_2020 =
"https://www.sec.gov/ix?doc=/Archives/edgar/data/0001326801/00013268012100 0014/fb-20201231.htm"
url_2019 =
"https://www.sec.gov/ix?doc=/Archives/edgar/data/0001326801/00013268012000 0013/fb-12312019x10k.htm"
url_2018 =
"https://www.sec.gov/Archives/edgar/data/1326801/000132680119000009/fb-123 12018x10k.htm"
url_2017 =
"https://www.sec.gov/Archives/edgar/data/1326801/000132680118000009/fb-123 12017x10k.htm"
url_2016 =
"https://www.sec.gov/Archives/edgar/data/1326801/000132680117000007/fb-123 12016x10k.htm"
url_2015 =
"https://www.sec.gov/Archives/edgar/data/1326801/000132680116000043/fb-123 12015x10k.htm"
url_2014 =
"https://www.sec.gov/Archives/edgar/data/1326801/000132680115000006/fb-123 12014x10k.htm"
url_2013 =
"https://www.sec.gov/Archives/edgar/data/1326801/000132680114000007/fb-123 12013x10k.htm"
url_2012 =
"https://www.sec.gov/Archives/edgar/data/1326801/000132680113000003/fb-123 12012x10k.htm"
# API Key to connect to the SEC API APIKEY = "INSERT YOUR API KEY HERE"
extractorApi = ExtractorApi(APIKEY)
# get the standardized and cleaned text of section 7 (MD&A) text_format_10k = extractorApi.get_section(url_2012, "7", "text") + extractorApi.get_section(url_2013, "7", "text") + extractorApi.get_section(url_2014, "7", "text") + extractorApi.get_section(url_2015, "7", "text") + extractorApi.get_section(url_2016, "7", "text") +
extractorApi.get_section(url_2017, "7", "text") + extractorApi.get_section(url_2018, "7", "text") + extractorApi.get_section(url_2019, "7", "text") + extractorApi.get_section(url_2020, "7", "text") + extractorApi.get_section(url_2021, extractorApi.get_section(url_2022, "7", "7", "text") "text") + # # Cleaning Data # #Remove unwanted characters clean_10k = re.sub('\W+',' ', text_format_10k) # Convert data to lower case clean_10k=clean_10k.lower() # Remove all the digits remove_digits = str.maketrans('', '', digits) clean_10k = clean_10k.translate(remove_digits) # Remove stop words pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('english')) + r')\b\s*') clean_10k = pattern.sub('', clean_10k) # Remove single letters tmp = re.sub(r'\b\w\b', ' ', clean_10k) clean_10k = re.sub(r'\s{2,}', ' ', tmp).strip() # Remove table words and roman numbers clean_10k = clean_10k.replace('table_start', '') clean_10k = clean_10k.replace('table_end', '') clean_10k = clean_10k.replace('ii', '') # Tokenization of filing
token_10k = word_tokenize(clean_10k)
# Checks if any data is found in the filing (bug checking) if clean_10k == "":
print("No data found in filing") else:
print(clean_10k)
# Calculate most frequent words in the filing frequentWords = nltk.FreqDist(token_10k) frequentWords.tabulate(10)
# Get stopwords
stopwords = set(STOPWORDS)
# WordCloud Generation
wordcloud = WordCloud(stopwords=stopwords, background_color="black", max_words=100, ).generate(clean_10k)
# WordCloud Visualization fig=plt.figure(figsize=(15, 8)) plt.imshow(wordcloud, interpolation='bilinear') plt.axis("off")
plt.title('2012 - 2022 META Word Cloud') plt.show()
#Sentiment Score for 10K Sec 7
sentiment = SentimentIntensityAnalyzer() sentiment.polarity_scores(clean_10k)
# Save sentiment scores in dataframe and plot it df_10k = sentiment.polarity_scores(clean_10k) df_10k = pd.DataFrame(df_10k, index=[0])
df_10k = df_10k[["neg", "pos", "neu"]] df_10k.plot.bar(align='edge', width=0.7) plt.title(' 2012 - 2022 META Sentiment Analysis')
plt.xlabel('Text') plt.ylabel('Sentiment Score') print(df_10k)
plt.show()


