# Anomaly Detection in NY Taxi Data Using LRT

In this work, we propose modifying the Likelihood Ratio Test (LRT) for anomaly detection by re-defining how to sub-sample the NY Taxi data. By sub-sampling in an intelligent way, we reduce the number of possible anomalous segments we test for. We perform a spatial and a temporal analysis separately to determine both anomalous regions and anomalous times within the NYC taxi cab data set provided by Kaggle. In our spatial analysis, we conclude that anomalous regions, such as the JFK airport, are properly detected by the framework. In our temporal analysis, we detect days of the year in lower Manhattan that are anomalous. We cross-check our findings with special events in lower Manhattan to ensure the anomalies being detected are significant.

## Prerequisites 
### Environment
Main Code was built using python 2.7 , Few data exploration visualizations were done in R .

If you are an anaconda user, a conda enviroment could be created using environment preset in the repository
It could be done using the command - conda env create -f environment.yml

If not you would be required to install the packages mention individually in the Appendix. 

### Data Set 
Data used in this project is taken from a kaggle kernal. 
~~It can be downloaded from "https://www.kaggle.com/oscarleo/new-york-city-taxi-with-osrm"  [ Last accessed on 4/20/2018 , report if link is broken]~~

All the downloaded files are to be placed at "./data" folder 
"./data/cache" has been used store intermediate data dumps, you would need to clean it, everytime changes are made to the subsampling part.  

## Code organisation and Running demo  
The main code for the project is present in base_code.py, If you wish to see the demo of this project , you can run the file. 

The code is organised as follows 
* ./data          -> Data set used for the runing the code to be downloaded and placed here
* ./data_loading  -> Contains code responsible for reading different input files
* ./data_preprocessing -> Code for all pre processing steps taken including sub sampling 
* ./lrt           -> Main LRT algorithm, with switch case for 3 different modes experimented
* ./visualizations  -> Code for basic visualization plots for data exploration 

## Report 
Plese read through the report for detailed analysis and assumptions made for this experimentation

## Authors 
* Harish Pullagurla ( hpullag@ncsu.edu ) 
* Hari Krishna Majety ( hmajety@ncsu.edu ) 
* Kenneth Tran ( kvtran2@ncsu.edu )

## Appendix 
### Packages 
* For running core algorithm :- Numpy , Pandas , Scipy ,  , statsmodels.api , patsy 
* For visualizations :- Matplotlib, seaborn, bokeh 


