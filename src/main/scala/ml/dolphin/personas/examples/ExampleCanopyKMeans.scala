import org.apache.spark.mllib.linalg.canopy._
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Example driver code for using CanopyKMeans
 *
 * @author Abhijit Bose
 * @version 1.0 06/24/2015
 * @since 1.0 06/24/2015
 *
 */

object ExampleCanopyKMeans {

  def main(args: Array[String]): Unit = {
    // define example points
    println("Starting Application....")
    val conf = new SparkConf()
                  .setAppName("Example usage of Canopy Clustering")
                  .set("spark.akka.frameSize", "10")
                  .set("spark.akka.threads", "4")
                  .set("spark.akka.timeout", "1000")
                  .set("spark.akka.heartbeat.pauses", "6000")
                  .set("spark.akka.failure-detector.threshold", "3000")
                  .set("spark.akka.heartbeat.interval", "1000")
                  .set("spark.eventLog.enabled", "true")

                  //.set("spark.storage.memoryFraction", "") // Set RDD caching limit as a fraction of overall JVM heap (60% default)
                  //.set("spark.shuffle.memoryFraction", "") // limit the total amount of memory used in shuffle-related buffers (20% default).  Rest 20% is for user code memory
                  //.set(

                 // .set("spark.shuffle.io.retryWait", )

    val sc = new SparkContext(conf)
    //val model = CanopyKMeans.train(sc, "/Users/r551839/canopy/points.csv", 2, 30.0, 20.0)
    //val model = CanopyKMeans.train(sc, "/Users/r551839/canopy/example2.csv", 3, 7.0, 3.0)
    val model = CanopyKMeans.train(sc, "/Users/r551839/canopy/wine_attribs_only.tsv", 3, 50.0, 1.0)
    model.clusterCenters.foreach(println)
  }
}
