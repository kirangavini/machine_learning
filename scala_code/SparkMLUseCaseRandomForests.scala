/*
   Spark with Scala

             Copyright : V2 Maestros @2016
                    
Code Samples : Spark Machine Learning - Random Forests

Problem Statement
*****************
The input data contains surveyed information about potential 
customers for a bank. The goal is to build a model that would 
predict if the prospect would become a customer of a bank, 
if contacted by a marketing exercise.

//// Techniques Used

1. Random Forests
2. Training and Testing
3. Confusion Matrix
4. Indicator Variables
5. Variable Reduction

-----------------------------------------------------------------------------
*/
val datadir = "C:/Personal/V2Maestros/Courses/Big Data Analytics with Spark/Scala"

//Create a SQL Context from Spark context
val sqlContext = new org.apache.spark.sql.SQLContext(sc)
import sqlContext.implicits._

//Load the CSV file into a RDD
val bankData = sc.textFile(datadir + "/bank.csv")
bankData.cache()
bankData.count()

//Remove the first line (contains headers)
val firstLine=bankData.first()
val dataLines = bankData.filter(x => x != firstLine)
dataLines.count()

//Convert the RDD into a Dense Vector. As a part of this exercise
//   1. Change labels to numeric ones
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LabeledPoint

def transformToNumeric( inputStr : String) : Vector = {
    val attList=inputStr.split(";")
    
    val age=attList(0).toFloat
    //convert outcome to float    
    val outcome = attList(16).contains("no") match {
                    case  true => 1.0
                    case  false    => 0.0
                }
  
    //create indicator variables for single/married 
    val single = attList(2).contains("single") match {
                    case  true => 1.0
                    case  false    => 0.0
                }
    val married = attList(2).contains("married") match {
                    case  true => 1.0
                    case  false    => 0.0
                } 
    val divorced = attList(2).contains("divorced") match {
                    case  true => 1.0
                    case  false    => 0.0
                }               
    
    //create indicator variables for education
    val primary = attList(3).contains("primary") match {
                    case  true => 1.0
                    case  false    => 0.0
                }           
    val secondary = attList(3).contains("secondary") match {
                    case  true => 1.0
                    case  false    => 0.0
                }           
    val tertiary = attList(3).contains("tertiary") match {
                    case  true => 1.0
                    case  false    => 0.0
                }           
   
    //convert default to float
    val default = attList(4).contains("no") match {
                    case  true => 1.0
                    case  false    => 0.0
                }
    //convert balance amount to float
    val balance = attList(5).contains("no") match {
                    case  true => 1.0
                    case  false    => 0.0
                }
    //convert loan to float
    val loan = attList(7).contains("no") match {
                    case  true => 1.0
                    case  false    => 0.0
                }
    //Filter out columns not wanted at this stage
    val values= Vectors.dense(outcome, age, single, married, 
                divorced, primary, secondary, tertiary,
                default, balance, loan )
    return values
}
//Change to a Vector
val bankVectors = dataLines.map(transformToNumeric)
bankVectors.collect()

//Perform statistical Analysis
import org.apache.spark.mllib.stat.{MultivariateStatisticalSummary, Statistics}
val bankStats=Statistics.colStats(bankVectors)
bankStats.mean
bankStats.variance
bankStats.min
bankStats.max

Statistics.corr(bankVectors)

//Transform to a Data Frame for input to Machine Learing
//Drop columns that are not required (low correlation)

def transformToLabelVectors(inStr : Vector  ) : (Float,Vector) = { 
    val values = ( inStr(0).toFloat, 
    Vectors.dense(inStr(1),inStr(2),inStr(3), 
        inStr(4),inStr(5),inStr(6),inStr(7), 
        inStr(8),inStr(9),inStr(10)))
    return values
 }
 
val bankLp = bankVectors.map(transformToLabelVectors)
bankLp.collect()
val bankDF = sqlContext.createDataFrame(bankLp).toDF("label","features")
bankDF.select("label","features").show(10)

//Perform PCA
import org.apache.spark.ml.feature.PCA
val bankPCA = new PCA()
bankPCA.setK(3)
bankPCA.setInputCol("features")
bankPCA.setOutputCol("pcaFeatures")
val pcaModel = bankPCA.fit(bankDF)
val pcaResult = pcaModel.transform(bankDF).select("label","pcaFeatures")
pcaResult.show()

//Indexing needed as pre-req for Decision Trees
import org.apache.spark.ml.feature.StringIndexer
val stringIndexer = new StringIndexer()
stringIndexer.setInputCol("label")
stringIndexer.setOutputCol("indexed")
val si_model = stringIndexer.fit(pcaResult)
val indexedBank = si_model.transform(pcaResult)
indexedBank.select("label","indexed","pcaFeatures").show()

//Split into training and testing data
val Array(trainingData, testData) = indexedBank.randomSplit(Array(0.9, 0.1))
trainingData.count()
testData.count()

import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

//Create the model
val rmClassifier = new RandomForestClassifier()
rmClassifier.setLabelCol("indexed")
rmClassifier.setFeaturesCol("pcaFeatures")
val rmModel = rmClassifier.fit(trainingData)

//Predict on the test data
val predictions = rmModel.transform(testData)
predictions.select("prediction","indexed","label","pcaFeatures").show()

val evaluator = new MulticlassClassificationEvaluator()
evaluator.setPredictionCol("prediction")
evaluator.setLabelCol("indexed")
evaluator.setMetricName("precision")
evaluator.evaluate(predictions) 

//Draw a confusion matrix
predictions.groupBy("indexed","prediction").count().show()
