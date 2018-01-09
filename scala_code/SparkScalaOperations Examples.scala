
/*
-----------------------------------------------------------------------------

                   Spark with Scala

             Copyright : V2 Maestros @2016
                    
Code Samples : Spark Processing
-----------------------------------------------------------------------------
*/

val datadir = "C:/Personal/V2Maestros/Courses/Big Data Analytics with Spark/Scala"

/*............................................................................
////   Loading Data From a Collection
//............................................................................*/
val collData=sc.parallelize(Array(3,5,4,7,4))
collData.cache()
collData.count()
/*............................................................................
////   Loading Data From Files
............................................................................*/

//Load the file. Lazy initialization
val autoData = sc.textFile(datadir + "/auto-data.csv")
autoData.cache()
//Loads only now.
autoData.count()
autoData.first()
autoData.take(5)

for (x <- autoData.collect()) { println(x) }

//............................................................................
////   Transformations
//............................................................................

//Map and create a new RDD
val tsvData=autoData.map( x => x.replace(",","\t"))
tsvData.take(5)

//Filter and create a new RDD
val toyotaData=autoData.filter(x => x.contains("toyota"))
toyotaData.count()

//FlatMap
val words=toyotaData.flatMap(x => x.split(","))
words.take(20)

//Distinct
for (x <- words.distinct().collect())
    println( x )

//Set operations
val words1 = sc.parallelize(Array("hello","war","peace","world"))
val words2 = sc.parallelize(Array("war","peace","universe"))

words1.union(words2).distinct().collect()
words1.intersection(words2).collect()

/*............................................................................
////   Actions
//............................................................................*/

//reduce
collData.reduce(( x,y ) => x+y)
//find the shortest line
autoData.reduce((x,y) =>  if (x.length() < y.length )  x  else  y )

//Aggregations

//Perform the same work as reduce
collData.aggregate(0) ((x, y) =>    (x + y),    (x, y) => (x + y))

//Do addition and multiplication at the same time.
//X now becomes a tuple for sequence
collData.aggregate((0,1)) ((x, y) =>   
    (x._1 + y, x._2 * y ),    (x, y) => (x._1 + y._1, x._2 * y._2))

//............................................................................
////   Functions in Spark
//............................................................................
//cleanse and transform an RDD

def cleanseRDD( autoStr : String ):String={
    val attList = autoStr.split(",")
    if ( attList(3).matches("two")) {
        attList(3)="2"
    }
    else {
        attList(3) = "4"
    }
    attList(3)
    //convert drive to upper case
    attList(5) = attList(5).toUpperCase()
    attList.mkString(",")
}

val cleanedData=autoData.map(cleanseRDD)
cleanedData.collect()

def isAllDigits(x: String) = x.matches("^\\d*$")

//Use a function to perform reduce 
def getMPG( autoStr : String) : String= {
    if ( isAllDigits(autoStr)) {
        return autoStr
    }
    else {
        val attList = autoStr.split(",")
        if ( isAllDigits(attList(9) )) {
            return attList(9)
        }
        else {
            return "0"
        }
    }
}

//find average MPG-City for all cars    
val totMPG =  autoData.reduce((x,y) =>  (getMPG(x).toInt + getMPG(y).toInt).toString )
totMPG.toInt/(autoData.count()-1)
    
//............................................................................
////   Working with Key/Value RDDs
//............................................................................

//create a KV RDD of auto Brand and Horsepower
val cylData = autoData.map( x => ( x.split(",")(0), x.split(",")(7)))
cylData.take(5)
cylData.keys.distinct.collect()

//Remove header row
val header = cylData.first()
val cylHPData= cylData.filter(x =>  x != header)

//Add a count 1 to each record and thn reduce to find totals of HP and counts
val brandValues=cylHPData.mapValues( x => (x.toInt, 1))
brandValues.collect()

val totValues=brandValues.reduceByKey((x,y) => (x._1 + y._1 , x._2 + y._2 ))
totValues.collect()
//find average by dividing HP total by count total
totValues.mapValues(x => x._1/x._2).collect()

//............................................................................
////   Advanced Spark : Accumulators & Broadcast Variables
//............................................................................

//function that splits the line as well as counts sedans and hatchbacks
//Speed optimization
 
//Initialize accumulator
val sedanCount = sc.accumulator(0)
val hatchbackCount =sc.accumulator(0)

//Set Broadcast variable
val sedanText=sc.broadcast("sedan")
val hatchbackText=sc.broadcast("hatchback")

def splitLines(line : String) : Array[String] = {
    if (line.contains(sedanText.value)) {
        sedanCount += 1
    }
    if (line.contains(hatchbackText.value)) {
        hatchbackCount += 1
    }
    line.split(",")
}
//do the map
val splitData=autoData.map(splitLines)

//Make it execute the map (lazy execution)
splitData.count()
println( sedanCount + "  " + hatchbackCount)

//............................................................................
////   Advanced Spark : Partitions
//............................................................................
collData.getNumPartitions

//Specify no. of partitions.
val collData=sc.parallelize(Array(3,5,4,7,4),2)

collData.getNumPartitions()

//localhost:4040 shows the current spark instance