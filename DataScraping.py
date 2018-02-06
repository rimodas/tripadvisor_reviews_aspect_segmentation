
# coding: utf-8

# In[2]:

#from urllib.request import urlopen
import bs4 
import requests
import pandas as pd
from pandas import DataFrame
import csv
base_url = 'https://www.tripadvisor.com'
base_write_url = 'C:/Users/rimo/Desktop/Project/'


# In[2]:

def getHotelNames():
    hotelNameAndUrlInternal = dict()
    #iterating through hotels
    for i in range (15):
        URL = base_url+'/Hotels-g32655-oa'+ str(i*30) +'-Los_Angeles_California-Hotels.html'
        page = requests.get(URL) #get the URL
        parsedlink = bs4.BeautifulSoup(page.text, 'html.parser') #parse the url
        for name in parsedlink.find_all("a", {"class" : "property_title"}):
            hotelNameAndUrlInternal[name.text] = base_url+name.get('href')
    return hotelNameAndUrlInternal


# In[3]:

def getORURL(URL):
    trigger = '-Reviews-'
    beforeTrigger,afterTrigger = URL.split(trigger)
    URL = beforeTrigger+trigger+'or{}-'+afterTrigger
    return URL


# In[4]:

#method to interate,parse and store reviews
def getReviews(hotelName,URLIn):
    url = getORURL(URLIn)
    reviewlinks=[]
    for i in range(50):
        URL = url.format(i*5)
        page = requests.get(URL) #get the URL
        soup = bs4.BeautifulSoup(page.text, 'html.parser') #parse the url
        links=[l.get('href') for div in soup.find_all('div', {'class': 'quote' }) for l in div.findAll('a')]
        for j in links:
            reviewlinks.append(base_url+j)
    #reviewlinks = list(set(reviewlinks))
    group= [] # a global list which will have all the details
    reviewlinks=list(set(reviewlinks))
    #Get the review details
    for i in reviewlinks:
        #parsing the reviewlinks
        source_candidate = requests.get(i)
        rsoup = bs4.BeautifulSoup(source_candidate.text, 'html.parser')
        #store the rating
        rating = str(rsoup.select('span[class*=ui_bubble_rating]')[0])
        rating = rating[11]
        #review header
        header = [info.strip('\n') for head in rsoup.find_all('div', id='PAGEHEADING') for info in head]
        header = header[0][1:-1]
        #review details
        review=[i.text for i in rsoup.find_all('p', class_='partial_entry')][0]
        group.append([header, review, rating])
    with open(base_write_url+hotelName+'.csv', 'w',encoding='utf-8') as fp:
        csv_writer = csv.writer(fp, delimiter=',')
        csv_writer.writerows(group)
    print('written '+hotelName+' file. Please check')


# In[5]:

def getMasterData():
    hotelDictionary = getHotelNames()
    listOfHotels = []
    i = 1
    for hotelName in hotelDictionary:
        hotelNameToWrite = hotelName.replace('/','-')
        listOfHotels.append([str(i)+'_'+hotelNameToWrite,hotelDictionary[hotelName]])
        i+=1
    with open(base_write_url+'master.csv', 'w',encoding='utf-8') as fp:
        csv_writer = csv.writer(fp, delimiter=',')
        csv_writer.writerows(listOfHotels)
    print('fetching master done')
#getMasterData()


# In[12]:

#store column names
u_cols = ['Hotel Name', 'Link']

users = pd.read_csv(base_write_url+'master.csv', sep=',', names=u_cols) #read the csv file

for i in range(0,346):
    fileName = users.iloc[i,0]
    link = users.iloc[i,1]
    getReviews(fileName,link)

