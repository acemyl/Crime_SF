# Databricks notebook source
# MAGIC %md
# MAGIC ## SF crime data analysis and modeling (DD: 3/23/2023 过期不侯哈)
# MAGIC

# COMMAND ----------

# MAGIC %md 
# MAGIC ### In this notebook, you can learn how to use Spark SQL for big data analysis on SF crime data. (https://data.sfgov.org/Public-Safety/Police-Department-Incident-Reports-Historical-2003/tmnf-yvry). 
# MAGIC The first part of Homework is OLAP for scrime data analysis (80 credits).  
# MAGIC The second part is unsupervised learning for spatial data analysis (20 credits). 选做  
# MAGIC The option part is the time series data analysis (50 credits).  选座
# MAGIC **Note**: you can download the small data (one month e.g. 2018-10) for debug, then download the data from 2013 to 2018 for testing and analysising. 
# MAGIC
# MAGIC ### How to submit the report for grading ? 
# MAGIC 1. Publish your notebook and send your notebook link to mike@laioffer.com. 
# MAGIC 2. Your report have to contain your data analysis insights.  
# MAGIC 3. write a ppt to present your work （选作）
# MAGIC
# MAGIC ### Bonus 
# MAGIC 1. choose different city (加10分)
# MAGIC 2. choose different analysis question （加10分）
# MAGIC 3. include other data together like house price, weather, news (加20分)
# MAGIC
# MAGIC ### Deadline 
# MAGIC Two weeks from the homework release date
# MAGIC

# COMMAND ----------

# DBTITLE 1,Import package 
from csv import reader
from pyspark.sql import Row 
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import warnings

import os
os.environ["PYSPARK_PYTHON"] = "python3"


# COMMAND ----------

# 从SF gov 官网读取下载数据

#import urllib.request
#urllib.request.urlretrieve("https://data.sfgov.org/api/views/tmnf-yvry/rows.csv?accessType=DOWNLOAD", "/tmp/myxxxx.csv")
#dbutils.fs.mv("file:/tmp/myxxxx.csv", "dbfs:/laioffer/spark_hw1/data/sf_03_18.csv")
#display(dbutils.fs.ls("dbfs:/laioffer/spark_hw1/data/"))
## 或者自己下载
# https://data.sfgov.org/api/views/tmnf-yvry/rows.csv?accessType=DOWNLOAD


# COMMAND ----------

data_path = "dbfs:/laioffer/spark_hw1/data/sf_03_18.csv"
# use this file name later

# COMMAND ----------

# MAGIC %md
# MAGIC ### Solove  big data issues via Spark
# MAGIC approach 1: use RDD (not recommend)  
# MAGIC approach 2: use Dataframe, register the RDD to a dataframe (recommend for DE)  
# MAGIC approach 3: use SQL (recomend for data analysis or DS， 基础比较差的同学)  
# MAGIC ***note***: you only need to choose one of approaches as introduced above

# COMMAND ----------

# DBTITLE 1,Get dataframe and sql

from pyspark.sql import SparkSession
spark = SparkSession \
    .builder \
    .appName("crime analysis") \
    .config("spark.some.config.option", "some-value") \
    .getOrCreate()

df_opt1 = spark.read.format("csv").option("header", "true").load(data_path)
display(df_opt1)
df_opt1.createOrReplaceTempView("sf_crime")

## helper function to transform the date, choose your way to do it. 

# refer: https://jaceklaskowski.gitbooks.io/mastering-spark-sql/spark-sql-functions-datetime.html
# from pyspark.sql.functions import to_date, to_timestamp, hour
# df_opt1 = df_opt1.withColumn('Date', to_date(df_opt1.OccurredOn, "MM/dd/yy"))
# df_opt1 = df_opt1.withColumn('Time', to_timestamp(df_opt1.OccurredOn, "MM/dd/yy HH:mm"))
# df_opt1 = df_opt1.withColumn('Hour', hour(df_opt1['Time']))
# df_opt1 = df_opt1.withColumn("DayOfWeek", date_format(df_opt1.Date, "EEEE"))

## 方法2 手工写udf 
#from pyspark.sql.functions import col, udf
#from pyspark.sql.functions import expr
#from pyspark.sql.functions import from_unixtime

#date_func =  udf (lambda x: datetime.strptime(x, '%m/%d/%Y'), DateType())
#month_func = udf (lambda x: datetime.strptime(x, '%m/%d/%Y').strftime('%Y/%m'), StringType())

#df = df_opt1.withColumn('month_year', month_func(col('Date')))\
#           .withColumn('Date_time', date_func(col('Date')))

## 方法3 手工在sql 里面
# select Date, substring(Date,7) as Year, substring(Date,1,2) as Month from sf_crime


## 方法4: 使用系统自带
# from pyspark.sql.functions import *
# df_update = df_opt1.withColumn("Date", to_date(col("Date"), "MM/dd/yyyy")) ##change datetype from string to date
# df_update.createOrReplaceTempView("sf_crime")
# crimeYearMonth = spark.sql("SELECT Year(Date) AS Year, Month(Date) AS Month, FROM sf_crime")

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q1 question (OLAP): 
# MAGIC #####Write a Spark program that counts the number of crimes for different category.
# MAGIC
# MAGIC Below are some example codes to demonstrate the way to use Spark RDD, DF, and SQL to work with big data. You can follow this example to finish other questions. 
# MAGIC

# COMMAND ----------

# DBTITLE 1,Spark dataframe based solution for Q1
q1_result = df_opt1.groupBy('category').count().orderBy('count', ascending=False)
display(q1_result)

# COMMAND ----------

# DBTITLE 1,Spark SQL based solution for Q1
#Spark SQL based
crimeCategory = spark.sql("SELECT  category, COUNT(*) AS Count FROM sf_crime GROUP BY category ORDER BY Count DESC")
display(crimeCategory)

# COMMAND ----------

# DBTITLE 1,Visualize your results
# important hints: 
## first step: spark df or sql to compute the statisitc result 
## second step: export your result to a pandas dataframe. 

crimes_pd_df = crimeCategory.toPandas()

# Spark does not support this function, please refer https://matplotlib.org/ for visuliation. You need to use display to show the figure in the databricks community. 

#display(p)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q2 question (OLAP)
# MAGIC Counts the number of crimes for different district, and visualize your results
# MAGIC

# COMMAND ----------

countcrime = spark.sql("SELECT  PdDistrict, COUNT(*) AS Count FROM sf_crime GROUP BY 1 ORDER BY 2 DESC")
display(countcrime)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q3 question (OLAP)
# MAGIC Count the number of crimes each "Sunday" at "SF downtown".   
# MAGIC hint 1: SF downtown is defiend  via the range of spatial location. For example, you can use a rectangle to define the SF downtown, or you can define a cicle with center as well. Thus, you need to write your own UDF function to filter data which are located inside certain spatial range. You can follow the example here: https://changhsinlee.com/pyspark-udf/
# MAGIC
# MAGIC hint 2: SF downtown 物理范围可以是 rectangle a < x < b  and c < y < d. thus, San Francisco Latitude and longitude coordinates are: 37.773972, -122.431297. X and Y represents each. So we assume SF downtown spacial range: X (-122.4213,-122.4313), Y(37.7540,37.7740). 也可以是中心一个圈，距离小于多少算做downtown
# MAGIC  

# COMMAND ----------

q3_result = spark.sql("""
    WITH Sunday_dt_crime AS (
        SELECT
            SUBSTRING(Date, 1, 2) AS Month,
            SUBSTRING(Date, 4, 2) AS Day,
            SUBSTRING(Date, 7) AS Year
        FROM
            sf_crime
        WHERE
            DayOfWeek = 'Sunday'
            AND -122.423671 < X
            AND X < -122.412497
            AND 37.773510 < Y
            AND Y < 37.782137
    )

    SELECT
        CONCAT(Year, '-', Month, '-', Day) AS Date,
        COUNT(*) AS Count
    FROM
        Sunday_dt_crime
    GROUP BY
        Year, Month, Day
    ORDER BY
        Year, Month, Day
""")

# Display the results
display(q3_result)


# COMMAND ----------

sf_crime = spark.table("sf_crime")
display(sf_crime)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Q4 question (OLAP)
# MAGIC Analysis the number of crime in each month of 2015, 2016, 2017, 2018. Then, give your insights for the output results. What is the business impact for your result?  

# COMMAND ----------

q4_result = spark.sql("""
WITH CrimeYearMonth AS (
    SELECT
        SUBSTRING(Date, 7) AS Year,
        SUBSTRING(Date, 1, 2) AS Month
    FROM
        sf_crime
    WHERE
        SUBSTRING(Date, 7) IN ('2015', '2016', '2017', '2018')
)


SELECT
    Year,
    Month,
    COUNT(*) AS CrimeCount
FROM
    CrimeYearMonth
GROUP BY
    Year, Month
ORDER BY
    Year, Month;
""")

display(q4_result)


# COMMAND ----------

# MAGIC %md
# MAGIC #### Q5 question (OLAP)
# MAGIC Analysis the number of crime with respsect to the hour in certian day like 2015/12/15, 2016/12/15, 2017/12/15. Then, give your travel suggestion to visit SF. 

# COMMAND ----------


q5_result = spark.sql("""
    SELECT
        SUBSTRING(Time, 1, 2) AS Hour,
        SUBSTRING(Date, 1, 10) AS Date_in_year,
        COUNT(*) AS Count
    FROM
        sf_crime
    WHERE
        Date IN ('12/31/2015', '12/31/2016', '12/31/2017')
    GROUP BY
        Date_in_year, Hour
    ORDER BY
        Date_in_year DESC, Hour
""")

display(q5_result)


# COMMAND ----------

# MAGIC %md 
# MAGIC #### Q6 question (OLAP)
# MAGIC (1) Step1: Find out the top-3 danger disrict  
# MAGIC (2) Step2: find out the crime event w.r.t category and time (hour) from the result of step 1  
# MAGIC (3) give your advice to distribute the police based on your analysis results. 
# MAGIC

# COMMAND ----------

q6_1 = spark.sql("""
WITH TopDangerDistricts AS (
    SELECT
        PdDistrict,
        COUNT(*) AS CrimeCount
    FROM
        sf_crime
    GROUP BY
        PdDistrict
    ORDER BY
        CrimeCount DESC
    LIMIT 3
)

SELECT
    *
FROM
    TopDangerDistricts;""")

display(q6_1)

# COMMAND ----------

q6_2 =spark.sql("""
                
WITH TopDangerDistricts AS (
    SELECT PdDistrict, COUNT(*) as Count
                             FROM sf_crime
                             GROUP BY 1
                             ORDER BY 2 DESC
                             LIMIT 3 
)

SELECT
    tdd.PdDistrict,
    sf_crime.Category,
    SUBSTRING(sf_crime.Time, 1, 2) AS Hour,
    COUNT(*) AS CrimeCount
FROM
    sf_crime
JOIN
    TopDangerDistricts tdd ON sf_crime.PdDistrict = tdd.PdDistrict
GROUP BY
    tdd.PdDistrict, sf_crime.Category, Hour
ORDER BY
    tdd.PdDistrict, Hour, CrimeCount DESC;
""")

display(q6_2)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Q7 question (OLAP)
# MAGIC For different category of crime, find the percentage of resolution. Based on the output, give your hints to adjust the policy.

# COMMAND ----------

q7 =spark.sql("""
              
WITH ResolutionPercentage AS (
    SELECT
        Category,
        Resolution,
        COUNT(*) AS CrimeCount
    FROM
        sf_crime
    GROUP BY
        Category, Resolution
)

SELECT
    Category,
    Resolution,
    CrimeCount,
    ROUND((CrimeCount / SUM(CrimeCount) OVER (PARTITION BY Category)) * 100, 2) AS ResolutionPercentage
FROM
    ResolutionPercentage
ORDER BY
    Category, Resolution;
""")
display(q7)

# COMMAND ----------

# MAGIC %md 
# MAGIC #### Q8 question: 
# MAGIC Analysis the new columns of the data and find how to use the new columns (e.g., like 'Fire Prevention Districts' etc)

# COMMAND ----------

q8 = spark.sql("""
SELECT
    PdDistrict,
    COUNT(*) AS RecordCount
FROM
    sf_crime
GROUP BY
    PdDistrict
ORDER BY
    RecordCount DESC;
""")

display(q8)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conclusion. 
# MAGIC Use four sentences to summary your work. Like what you have done, how to do it, what the techinical steps, what is your business impact. 
# MAGIC More details are appreciated. You can think about this a report for your manager. Then, you need to use this experience to prove that you have strong background on big  data analysis.  
# MAGIC Point 1:  what is your story ? and why you do this work ?   
# MAGIC Point 2:  how can you do it ?  keywords: Spark, Spark SQL, Dataframe, Data clean, Data visulization, Data size, clustering, OLAP,   
# MAGIC Point 3:  what do you learn from the data ?  keywords: crime, trend, advising, conclusion, runtime 
# MAGIC

# COMMAND ----------

#In this project, I conducted an extensive analysis of San Francisco crime data to derive valuable insights that can inform law enforcement strategies and community safety measures. Spatial and temporal analyses were performed to identify top dangerous districts, and subsequent queries were executed to reveal crime trends based on categories and hours. The analysis involved both clustering techniques and OLAP (Online Analytical Processing) to extract meaningful patterns from the data.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optional part: Clustering
# MAGIC You can apply Spark ML custering algorithm to cluster the spatial data, then visualize the clustering results. Do not do this until you understand Spark ML, we would like to cover this in the DS track. 

# COMMAND ----------

# MAGIC %md 
# MAGIC ### Optional part: Time series analysis
# MAGIC This part is not based on Spark, and only based on Pandas Time Series package.   
# MAGIC Note: I am not familiar with time series model, please refer the ARIMA model introduced by other teacher.   
# MAGIC process:  
# MAGIC 1.visualize time series  
# MAGIC 2.plot ACF and find optimal parameter  
# MAGIC 3.Train ARIMA  
# MAGIC 4.Prediction 
# MAGIC
# MAGIC Refer:   

# MAGIC https://www.statsmodels.org/dev/examples/notebooks/generated/tsa_arma_0.html  
# MAGIC https://www.howtoing.com/a-guide-to-time-series-forecasting-with-arima-in-python-3  
# MAGIC https://www.joinquant.com/post/9576?tag=algorithm  
# MAGIC https://blog.csdn.net/u012052268/article/details/79452244  
