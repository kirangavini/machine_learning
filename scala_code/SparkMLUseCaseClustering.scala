/*
   Spark with Python

             Copyright : V2 Maestros @2016
                    
Code Samples : Spark Machine Learning - Clustering

The input data contains samples of cars and technical / price 
information about them. The goal of this problem is to group 
these cars into 4 clusters based on their attributes

//// Techniques Used

1. K-Means Clustering
2. Centering and Scaling

-----------------------------------------------------------------------------
*/
val datadir = "C:/Personal/V2Maestros/Courses/Big Data Analytics with Spark/Scala"

//Create a SQL Context from Spark context
val sqlContext = new org.apache.spark.sql.SQLContext(sc)
import sqlContext.implicits._

//Load the CSV file into a RDD
val autoData = sc.textFile(datadir + "/auto-data.csv")
autoData.cache()

//Remove the first line (contains headers)
val firstLine=autoData.first()
val dataLines = autoData.filter(x => x != firstLine)
dataLines.count()

//Convert the RDD into a Dense Vector. As a part of this exercise
//   1. Change labels to numeric ones
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint

//Convert to Local Vector.
def transformToNumeric( inputStr : String) : Vector = {
    val attList=inputStr.split(",")

    val doors = attList(3).contains("two") match {
        case  true => 0.0
        case  false    => 1.0
    }
    val body = attList(4).contains("sedan") match {
        case  true => 0.0
        case  false    => 1.0
    }     
    //Filter out columns not wanted at this stage
    //only use doors, body, hp, rpm, mpg-city
    val values= Vectors.dense( doors, body,
                     attList(7).toDouble, attList(8).toDouble,
                     attList(9).toDouble)
    return values
}
val autoVector = dataLines.map(transformToNumeric)
autoVector.cache()
autoVector.collect()

//Centering and scaling. To perform this every value should be subtracted
//from that column's mean and divided by its Std. Deviation.

//Perform statistical Analysis and compute mean and Std.Dev for every column
import org.apache.spark.mllib.stat.Statistics
val autoStats=Statistics.colStats(autoVector)
val colMeans=autoStats.mean
val colVariance=autoStats.variance
val colStdDev=colVariance.toArray.map( x => Math.sqrt(x))

//place the means and std.dev values in a broadcast variable
val bcMeans=sc.broadcast(colMeans)
val bcStdDev=sc.broadcast(colStdDev)

def centerAndScale(inVector : Vector ) : (Double,Vector)  = {
    val meanArray=bcMeans.value
    val stdArray=bcStdDev.value
    
    val valueArray=inVector.toArray
    var retArray=Array[Double]()
    for (i <- 0 to valueArray.size - 1)  {
        retArray = retArray :+ ( (valueArray(i) - meanArray(i)) /
                            stdArray(i) )
    }
    //use a dummy label
    return (1.0,Vectors.dense(retArray))
 }   
val csAuto = autoVector.map(centerAndScale)
csAuto.collect()

//Create a Spark Data Frame
val autoDF = sqlContext.createDataFrame(csAuto).
        toDF("dummy","features").toDF.select("features").toDF()
autoDF.select("features").show(10)

import  org.apache.spark.ml.clustering.KMeans
val kmeans = new KMeans()
kmeans.setK(3)
kmeans.setSeed(1)

val model = kmeans.fit(autoDF)
val predictions = model.transform(autoDF)
predictions.select("features","prediction").show()
predictions.groupBy("prediction").count().show()

val mergePredictions = dataLines.zip(predictions.select("prediction").rdd)

