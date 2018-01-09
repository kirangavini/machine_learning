/*
   Spark with Scala

             Copyright : V2 Maestros @2016
                    
Code Samples : Spark Machine Learning - Recommendation Engine

Problem Statement
*****************
The input data contains a file with user, item and ratings. 
The purpose of the exercise is to build a recommendation model
and then predict the affinity for users to various items
-----------------------------------------------------------------------------
*/
val datadir = "C:/Personal/V2Maestros/Courses/Big Data Analytics with Spark/Scala"

//Create a SQL Context from Spark context
val sqlContext = new org.apache.spark.sql.SQLContext(sc)
import sqlContext.implicits._

//Load the data file in ALS format (user, item, rating)
val ratingsData = sc.textFile(datadir + "/useritemdata.txt")
ratingsData.collect()

//Convert the strings into a proper vector
val ratingVector=ratingsData.map(x => 
        x.split(',')).map(l => (l(0).toInt, l(1).toInt, l(2).toInt))

//Build a SQL Dataframe
val ratingsDf=sqlContext.createDataFrame(ratingVector).toDF( 
            "user","item","rating")
ratingsDf.show()

//build the model based on ALS
import org.apache.spark.ml.recommendation.ALS
val als = new ALS()
als.setRank(10)
als.setMaxIter(5)
val model = als.fit(ratingsDf)

//Create a test data set of users and items you want ratings for
val testRDD = sc.parallelize(Array((1001, 9006),(1001,9004),(1001,9005)))
val testDf = sqlContext.createDataFrame(testRDD).toDF(
            "user","item")

//Predict            
val predictions=model.transform(testDf)
predictions.select("user","item","prediction").show()

