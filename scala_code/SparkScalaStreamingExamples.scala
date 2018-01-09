// -*- coding: utf-8 -*-
/*
-----------------------------------------------------------------------------

                   Spark with Scala

             Copyright : V2 Maestros @2016
                    
Code Samples : Spark Streaming
-----------------------------------------------------------------------------
*/

val datadir = "C:/Personal/V2Maestros/Courses/Big Data Analytics with Spark/Scala"

//............................................................................
////   Streaming with simple data
//............................................................................


import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.StreamingContext._
import org.apache.spark.streaming.Seconds

var ssc = new StreamingContext(sc,Seconds(1))

import scala.collection.mutable.Queue
import org.apache.spark.rdd.RDD

val rddQueue = new Queue[RDD[Int]]()

 // Create the QueueInputDStream and use it do some processing
val inputStream = ssc.queueStream(rddQueue)
val mappedStream = inputStream.map(x => (x % 10, 1))
val reducedStream = mappedStream.reduceByKey( (x,y) => x+y)
reducedStream.print()

ssc.start()

#simulate some data
for (i <- 1 to 10) {
    rddQueue.synchronized {
        rddQueue += ssc.sparkContext.makeRDD(1 to 1000, 10)
    }
      Thread.sleep(1000)
}
ssc.stop()

//............................................................................
////   Streaming with TCP/IP data
//............................................................................

import org.apache.spark.streaming.StreamingContext
import org.apache.spark.streaming.StreamingContext._
import org.apache.spark.streaming.Seconds
import org.apache.spark.rdd.RDD

//Create streaming context with latency of 3
var ssc = new StreamingContext(sc,Seconds(3))

val lines = ssc.socketTextStream("localhost", 9000)

//Word count within RDD    
val words = lines.flatMap(x => x.split(" "))
val pairs = words.map(x => (x, 1))
val wordCounts = pairs.reduceByKey((x,y) => x + y)
wordCounts.print()

//Count lines
def computeMetrics(rdd : RDD[String]):Long =  {
    val linesCount=rdd.count()
    rdd.collect()
    println( "Lines in RDD :" + linesCount)
    return linesCount
}
val lineCount= lines.foreachRDD(computeMetrics(_))

//Compute window metrics
def windowMetrics(rdd : RDD[String] ) = {
    println( "Window RDD size:" +  rdd.count() )
}

val windowedRDD=lines.window(Seconds(6),Seconds(3))
windowedRDD.foreachRDD(windowMetrics(_))

ssc.start()
Thread.sleep(10000)
ssc.stop()

