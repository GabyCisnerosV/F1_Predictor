##############################################################################
#### 1_API_REQUESTS_ERGAST 
##############################################################################
# To Update data from Races, Qualifying sessions, pit stops, laps
##############################################################################
# f1 Results from https://ergast.com/mrd/ . This sourcce will stop working in 2024.
##############################################################################
# Not parallel, built in 2022

import requests
import pandas as pd

y_start,y_end=2024,2025 # range of years data is extracted
r_start,r_end=18,25 #range races in the year (Sessions) extracted

##############################################################################
### Race Results per race per driver

#1 Results in one table (all but the last race)

ResultsDF=pd.DataFrame()

for year in range(y_start,y_end): #From 2003 until 2022 (to be able to use in the close future)
    for race in range(r_start,r_end): #The maximum number of races was in 2021 with 22 races
        url="http://ergast.com/api/f1/"+str(year)+"/"+str(race)+"/results.json" #modifying the url
        Results=requests.get(url) #Request from API
        Results=Results.json() #Results in json format
        Races=Results['MRData']['RaceTable']["Races"] #subset of desired features
        
        #Each season has a different number of races per year
        if Races!=[]: #if the year and number of race exist
            print("Season ",year,"- Race ",race,": The race was held.")
            df= pd.json_normalize(Races,record_path=['Results'],
                       meta=["season","round",'raceName',
                             ['Circuit','circuitId'],
                             ['Circuit','circuitName'],
                             ['Circuit','Location','country'],
                             'date'])
            ResultsDF = pd.concat([ResultsDF, df], axis=0)
        elif Races==[]: #if this number of race was not held that year
            print("Season ",year,"- Race ",race,": The race was NOT held.")
            continue
        
#2 The last race url has a different format:
url="http://ergast.com/api/f1/current/last/results.json" #modifying the url
Results=requests.get(url) #Request from API
Results=Results.json() #Results in json format
Races=Results['MRData']['RaceTable']["Races"] #subset of desired features   
df= pd.json_normalize(Races,record_path=['Results'],
                       meta=["season","round",'raceName',
                             ['Circuit','circuitId'],
                             ['Circuit','circuitName'],
                             ['Circuit','Location','country'],
                             'date'])
ResultsDF = pd.concat([ResultsDF, df], axis=0)


ResultsDF.info()
#ResultsDF.groupby("position")["number"].count()



##############################################################################
### Qualifying Results per race, per driver

QualifyingDF=pd.DataFrame()

for year in range(y_start,y_end): #Available from 2003
    for race in range(r_start,r_end): #The maximum number of races was in 2021 with 22 races
        url="http://ergast.com/api/f1/"+str(year)+"/"+str(race)+"/qualifying.json" #modifying the url
        Results=requests.get(url) #Request from API
        Results=Results.json() #Results in json format
        Races=Results['MRData']['RaceTable']["Races"] #subset of desired features
        
        #Each season has a different number of races per year
        if Races!=[]: #if the year and number of race exist
            print("Season ",year,"- Race ",race,": The race was held.")
            df= pd.json_normalize(Races,record_path=['QualifyingResults'],
                       meta=["season","round",'raceName',
                             ['Circuit','circuitId'],
                             ['Circuit','circuitName'],
                             ['Circuit','Location','country'],
                             'date'])
            QualifyingDF = pd.concat([QualifyingDF, df], axis=0)
        elif Races==[]: #if this number of race was not held that year
            print("Season ",year,"- Race ",race,": The race was NOT held.")
            continue
        

QualifyingDF.info()



# ##############################################################################
# ### Laps times per race, per driver

LapsDF=pd.DataFrame()

for year in range(y_start,y_end): #Available from 2003
    for race in range(r_start,r_end): #The maximum number of races was in 2021 with 22 races
        for lap in range(1,100): #The maximum number of laps per race is 76
            url="http://ergast.com/api/f1/"+str(year)+"/"+str(race)+"/laps/"+str(lap)+".json" #modifying the url
            Results=requests.get(url) #Request from API
            Results=Results.json() #Results in json format
            Races=Results['MRData']['RaceTable']["Races"] #subset of desired features
            
            #Each season has a different number of races per year
            if Races!=[]: #if the year and number of race exist
                Races=Results['MRData']['RaceTable']["Races"][0] #subset of desired features
                print("Season ",year,"- Race ",race,": The race was held.")
                #General information about the race and season
                df1 = pd.json_normalize(Races)
                df1["LapNumber"]=lap
                df1=df1.drop(['Laps', 'url'], axis=1)
                
                #Specific information of the lap
                df2 = pd.json_normalize(Races["Laps"],record_path=['Timings'],
                                       meta=["number"])
                df2["number"] = pd.to_numeric(df2["number"])
                df2=df2.rename(columns={"number": "LapNumber","time":"LapDuration"})
                
                Laps=pd.merge(df1, df2,how="right",on="LapNumber")
                LapsDF = pd.concat([LapsDF, Laps], axis=0)
            elif Races==[]: #if this number of race was not held that year
                print("Season ",year,"- Race ",race,": The race was NOT held.")
                continue


LapsDF.info()
LapsDF=LapsDF.drop(["time"],axis=1)


# ##############################################################################
# ### Pits per race, per driver

PitsDF=pd.DataFrame()

for year in range(y_start,y_end): #Available from 2012
    for race in range(r_start,r_end): #The maximum number of races was in 2021 with 22 races
        url="http://ergast.com/api/f1/"+str(year)+"/"+str(race)+"/pitstops.json" #modifying the url
        Results=requests.get(url) #Request from API
        Results=Results.json() #Results in json format
        Races=Results['MRData']['RaceTable']["Races"] #subset of desired features
        
        #Each season has a different number of races per year
        if Races!=[]: #if the year and number of race exist
            Races=Results['MRData']['RaceTable']["Races"][0]
            print("Season ",year,"- Race ",race,": The race was held.")
            df= pd.json_normalize(Races,record_path=['PitStops'],
                       meta=["season","round",'raceName',
                             ['Circuit','circuitId'],
                             ['Circuit','circuitName'],
                             ['Circuit','Location','country'],
                             'date'])
            PitsDF = pd.concat([PitsDF, df], axis=0)
        elif Races==[]: #if this number of race was not held that year
            print("Season ",year,"- Race ",race,": The race was NOT held.")
            continue


PitsDF.info()





##############################################################################
### Update Existing Data

results_pre=pd.read_csv("C:/Users/gabri/Dropbox/Gaby/Proyectos/My_Portafolio/F1/Data/ResultsDF.csv")
qualifying_pre=pd.read_csv("C:/Users/gabri/Dropbox/Gaby/Proyectos/My_Portafolio/F1/Data/QualifyingDF.csv")
laps_pre=pd.read_csv("C:/Users/gabri/Dropbox/Gaby/Proyectos/My_Portafolio/F1/Data/LapsDF.csv")
pits_pre=pd.read_csv("C:/Users/gabri/Dropbox/Gaby/Proyectos/My_Portafolio/F1/Data/PitsDF.csv")

print("DFs Before:")
dfs={"Results":results_pre,"Qualifying":qualifying_pre,"Laps":laps_pre,"Pits":pits_pre}
for k,v in dfs.items():
    print(k,len(v))

results_updated=pd.concat([ResultsDF,results_pre]).drop_duplicates()
qualifying_updated=pd.concat([QualifyingDF,qualifying_pre]).drop_duplicates()
laps_updated=pd.concat([LapsDF,laps_pre]).drop_duplicates()
pits_updated=pd.concat([PitsDF,pits_pre]).drop_duplicates()

print("DFs After:")
dfs={"Results":results_pre,"Qualifying":qualifying_pre,"Laps":laps_pre,"Pits":pits_pre}
for k,v in dfs.items():
    print(k,len(v))

#Store four dataframes in csv
path = 'C:/Users/gabri/Dropbox/Gaby/Proyectos/My_Portafolio/F1/Data/'

results_updated.to_csv(path+"ResultsDF.csv")
qualifying_updated.to_csv(path+"QualifyingDF.csv")
laps_updated.to_csv(path+"LapsDF.csv")
pits_updated.to_csv(path+"PitsDF.csv")

