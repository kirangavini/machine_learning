

val datadir = "C:/Personal/V2Maestros/Courses/Big Data Analytics with Spark/Scala"

//............................................................................
////   Building and saving the model
//............................................................................

val tweetData = sc.textFile(datadir + "/movietweets.csv")
tweetData.collect()

def convertToRDD(inStr : String) : (Double,String) = {
    val attList = inStr.split(",")
    val sentiment = attList(0).contains("positive") match {
            case  true => 0.0
            case  false    => 1.0
     }
    return (sentiment, attList(1))
}
val tweetText=tweetData.map(convertToRDD)
tweetText.collect()

val sqlContext = new org.apache.spark.sql.SQLContext(sc)
import sqlContext.implicits._
var ttDF = sqlContext.createDataFrame(tweetText).toDF("label","text")

import org.apache.spark.ml.feature.{HashingTF, Tokenizer, IDF}
import org.apache.spark.ml.classification.{NaiveBayes, NaiveBayesModel}
import org.apache.spark.ml.Pipeline

//Setup tokenizer that splits sentences to words
val tokenizer = new Tokenizer()
tokenizer.setInputCol("text")
tokenizer.setOutputCol("words")
val tokens = tokenizer.transform(ttDF)

//Setup the TF compute function
val hashingTF = new HashingTF()
hashingTF.setInputCol(tokenizer.getOutputCol)
hashingTF.setOutputCol("tempfeatures")
val hashValues = hashingTF.transform(tokens)

//Setup the IDF compute function
val idf=new IDF()
idf.setInputCol(hashingTF.getOutputCol)
idf.setOutputCol("features")
val idfModel = idf.fit(hashValues)
val idfValues = idfModel.transform(hashValues)

//Setup the Naive Bayes classifier
val nbClassifier=new NaiveBayes()

//Build the model
val nbModel=nbClassifier.fit(idfValues)

//check for accuracy
val predictions=nbModel.transform(idfValues)

//Form confusion matrix
predictions.groupBy("label","prediction").count().show()

//save the model
//Does not work on windows :(
nbModel.save(datadir+"/tweetsModel")

//............................................................................
////   Getting tweets in real time and making predictions
//............................................................................
    
import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.StreamingContext._
import org.apache.spark.streaming.Seconds

//Create streaming context with latency of 3
var streamContext = new StreamingContext(sc,Seconds(3))

val tweets = streamContext.socketTextStream("localhost", 9000)

val bc_model = sc.broadcast(nbModel)

import org.apache.spark.ml.feature.{HashingTF, Tokenizer, IDF}
//Setup tokenizer that splits sentences to words
val ptokenizer = new Tokenizer()
ptokenizer.setInputCol("text")
ptokenizer.setOutputCol("words")
val bc_tokenizer = sc.broadcast(ptokenizer)

//Setup the TF compute function
val phashingTF = new HashingTF()
phashingTF.setInputCol(ptokenizer.getOutputCol)
phashingTF.setOutputCol("tempfeatures")
val bc_hashingTF = sc.broadcast(phashingTF)

//Setup the IDF compute function
val pidf=new IDF()
pidf.setInputCol(phashingTF.getOutputCol)
pidf.setOutputCol("features")
val bc_idf = sc.broadcast(pidf)

import org.apache.spark.rdd.RDD

def predictSentiment(tweetText : RDD[String]) = {
    
    tweetText.collect()
    val tweetEnx = tweetText.map(x => (1.0, x))
    val tweetRDD = sqlContext.createDataFrame(tweetEnx).toDF("dummy","text")
    val nbModel=bc_model.value
    val ptokenizer=bc_tokenizer.value
    val phashingTF = bc_hashingTF.value
    val pidf = bc_idf.value
    
    val tokens = ptokenizer.transform(tweetRDD)
    val hashValues = phashingTF.transform(tokens)
    val idfModel = pidf.fit(hashValues)
    val idfValues = idfModel.transform(hashValues)
    val pprediction = nbModel.transform(idfValues)

    println( "Predictions for this window :" )
    pprediction.select("text","prediction").show()
}

tweets.foreachRDD( x => predictSentiment(x))

streamContext.start()
streamContext.stop()

