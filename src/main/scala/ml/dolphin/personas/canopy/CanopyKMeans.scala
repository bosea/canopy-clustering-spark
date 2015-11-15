package org.apache.spark.mllib.linalg.canopy

import ml.dolphin.personas.canopy.{EuclideanVectorSpace, PersonaCommon, XORShiftRandom}
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.mllib.linalg.BLAS.axpy
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.{Logging, SparkContext}

import scala.collection.mutable
import scala.util.Random
import scala.util.control.Breaks._

/**
 * An implementation of Canopy K-means clustering in Spark
 *
 * @author Abhijit Bose
 * @version 1.0 06/24/2015
 * @since 1.0 06/24/2015
 *
 */

class CanopyKMeans private(
                            private var k: Int,
                            private var maxIterations: Int,
                            private var epsilon: Double,
                            private var seed: Int,
                            private var t1: Double,
                            private var t2: Double) extends Serializable with Logging {

  // default constructor for class
  def this() = this(2, 10, 1.e-04, new scala.util.Random().nextInt(), 0.0, 0.0)

  /**
   * Definitions for getter and setter methods for algorithm parameters follow.
   */

  // Number of centroids
  def getK: Int = k

  def setK(k: Int): this.type = {
    this.k = k
    this
  }

  // Maximum number of iterations
  def getMaxIterations: Int = maxIterations

  def setMaxIterations(maxIterations: Int): this.type = {
    this.maxIterations = maxIterations
    this
  }

  // Delta of centroid distances between successive iterations. Used to decide convergence.
  def getEpsilon: Double = epsilon

  def setEpsilon(epsilon: Double): this.type = {
    this.epsilon = epsilon
    this
  }

  // Random seed for cluster initialization
  def getSeed: Int = seed

  def setSeed(seed: Int): this.type = {
    this.seed = seed
    this
  }

  // T1: Distance from a canopy center beyond which the points can belong to other canopies.
  def getT1: Double = t1

  def setT1(value: Double): this.type = {
    this.t1 = value
    this
  }

  // T2: Distance from a canopy center within which all points belong to the same canopy.
  // T1 > T2 must be set.
  def getT2: Double = t2

  def setT2(value: Double): this.type = {
    this.t2 = value
    this
  }

  /**
   * Algorithm to be invoked for performing canopy clustering. Other methods are private.
   *
   * @param sc  Spark Context
   * @param input Input file locator as a String
   * @return org.apache.spark.mllib.clustering.KMeansModel object containing the k-means model
   */
  def runAlgorithm(sc: SparkContext, input: String): KMeansModel = {

    // Read input CSV files generated as the result of a Hive query.
    // @todo Generalize input file options.
    val features = PersonaCommon.readHivePoints(sc, input)

    /*
     * Generate k random centers from the input points. Each point is (Vector, Int) where the
     * second element represents the hashcode of the Vector.
     */
    var costDiff = Double.PositiveInfinity
    val centers = initRandom(features)
    println("Initial Centers => " + centers.foreach(println))
    /*
     * Apply the Canopy algorithm to find the canopy centers and assign canopies to each point.
     * To reduce the amount of storage, we keep the canopy->points associations as
     * canopy->Set[hashCode(point)] where point is a Vector. To check whether a Vector belongs to
     * a canopy, first produce the hash: hashcode(point), and then check which canopy it belongs to.
     *
     * We broadcast the canopy centers to all partitions so lloyd's algorithm can be performed
     * locally.
     */
    val canopies = canopyCenters(features)
    val bcCanopies = sc.broadcast(canopies)
    //println("XXX Canopies Broadcasted")
    /*
     * Main loop for k-means iterations. Within each iteration, we do all operations using
     * mapPartitions to calculate results locally within partitions and then do a global
     * collection/operation in order to avoid excessive shuffling of data.
     */
    var iteration = 0
    while (iteration < maxIterations && costDiff >= epsilon) {
      println("Iteration Number => " + iteration)
      val bcCenters = sc.broadcast(centers).value

      var costAccumulator = sc.accumulator(0.0, "k-means cost function")
      /*
       *  For each RDD partition of data, do a mapPartition operation as follows:
       *
       *  - Within a partition p:
       *    1. Initialize Array for (a) keeping a running sum of all points closest to a centroid
       *    and (b) count of such points to the centroid, so we can calculate average distances.
       *    2. For each point x in p:
       *    2.1  Find the closest centroid c of x based on distance(centers, x)
       *    2.2  Add to running sum for c (associative): contribution of x, increment count of points
       *    2.3  Add to accumulator sum of total cost (associative): distance(c, x)
       *    2.4  Return c -> (running sum, count) as iterator of mapPartition
       *  - Perform reduceByKey over all the partitions to merge the results
       */
      val partContribs = features.mapPartitions { points => {
        // local computations within a partition. Within a partition, the following are global:
        val k = bcCenters.length
        val dims = bcCenters(0)._1.size
        val runningSums = Array.fill(k)(Vectors.zeros(dims))
        //val ones = Array.fill(dims)(1.0)
        val counts = Array.fill(k)(0L)


        // Operations for each point x in points
        points.foreach(x => {
          // check which center belongs to the same canopy as x. Return the index of that center.
          var index = 0
          val isCanopied = isWithinCanopy(bcCanopies.value, bcCenters, x)
          var distance = 0.0

          if (isCanopied >= 0) {
            distance = EuclideanVectorSpace.distance(bcCenters(index)._1, x._1)
            index = isCanopied
          } else {
            // Brute-force distance calculation over all centers and find the minimum
            val (i, d) = EuclideanVectorSpace.closest(EuclideanVectorSpace.toVector(centers),
                                                             x._1)
            index = i
            distance = d
          }
          val sum = runningSums(index)

          axpy(1.0, x._1, sum)

          counts(index) += 1
          costAccumulator += distance
        })
        val contribs = for (i <- 0 until k) yield {
          (i, (runningSums(i), counts(i)))
        }
        contribs.iterator
      }}

      // Sum up the running sum and count contributions from all partitions in costContribs
      type SumCount = (Vector, Long)
      val totalContribs = partContribs.reduceByKey((x: SumCount, y: SumCount) => {
        axpy(1.0, x._1, y._1)
        (y._1, x._2 + y._2)
      }).collectAsMap()

      // Update cluster centers
      costDiff = 0.0
      for (i <- 0 until k) {
        val (sum, count) = totalContribs(i)
        if (count != 0) {
          val newCenter = Vectors.dense(sum.toArray.map(_ / count))
          costDiff += EuclideanVectorSpace.distance(newCenter, centers(i)._1)
          centers(i) = (newCenter, newCenter.hashCode())
        }
      }
      iteration += 1
    }
    val cv = centers.map(_._1) // only need the center Vector's
    println("CVs => " + cv.foreach(println))
    new KMeansModel(cv)
  }

  /**
   * Algorithm for canopy clustering.
   * 1. Find local canopy centers from each RDD partition of input data
   * 2. Merge local canopies to generate a global set of canopy centers.
   *
   * @param data RDD of [Vector, Int] where Vector corresponds to features or attributes
   *             for a given point. Int corresponds to hash code of the Vector elements.
   * @return An array of Vector's corresponding to the canopy centers.
   */
  private def canopyCenters(data: RDD[(Vector, Int)]): mutable.Map[Vector, mutable.Set[Int]] = {

    // Find local canopies from each partition
    val c = data.mapPartitions { points => {
      // Copy points into a mutable Array so we can access and modify the elements.
      // This needs to be readdressed if it becomes a memory bottleneck.
      var ptArray = mutable.ArrayBuffer[(Vector, Int)]()
      points.foreach(x => ptArray += x)

      val canopies = findCanopies(ptArray)
      canopies.foreach { x =>
        println("canopyCenters from partitions => " + x)
      }
      canopies.iterator
    }
    }.collect.toMap

    println("XXX FINISHED CANOPIES")
    // Merge local canopies across partitions to generate global canopies
    val centers = mutable.ArrayBuffer[(Vector, Int)]()
    c.foreach(x => centers.append((x._1, x._1.hashCode())))
    // Use the same algorithm again on the local canopy centers
    println("canopyCenters: CENTERS => " + centers)

    val cpCenters = findCanopies(centers)

    // Create the final canopy centers by merging hash codes from canopy centers
    // that merged
    val canopies = mutable.Map[Vector, mutable.Set[Int]]()
    cpCenters.foreach(x => {
      val setX = c(x._1)
      println("ABOSE..." + "x._1 => " + x._1 + " " + setX)
      for (hX <- setX) {
        if (canopies.contains(x._1))
          canopies(x._1).add(hX)
        else
          canopies += (x._1 -> mutable.Set[Int](hX))
      }
      if (x._2.size > 0) {
        var h = 0
        for (h <- x._2) {
          centers.foreach(y => {
            if (y._2 == h) {
              val setY = c(y._1)
              for (hY <- setY) {
                canopies(x._1).add(hY)
              }
            }
          })
        }
      }
    })

    println("Final canopies " + canopies)
    canopies
  }

  /** *
    * Canopy finding algorithm for a given set of points. Note we send the hashcode
    * of a Vector along with the Vector as a new type: (Vector, Int).
    *
    * @param points
    * @return
    */
  private def findCanopies(points: mutable.ArrayBuffer[(Vector, Int)]):
  mutable.Map[Vector, mutable.Set[Int]] = {
    var r = points
    var canopies = mutable.Map[Vector, mutable.Set[Int]]()
    println("findCanopies: POINTS SIZE " + r.size + "POINTS VALUES " + r)

    while (r.size > 0) {
      // Choose a point as canopy center
      //val canopyIdx = scala.util.Random.nextInt(points.size)
      //val canopy = points(canopyIdx)
      val shuffled = Random.shuffle(r)
      val canopy = shuffled.head
      println("INSIDE WHILE: New canopy => " + canopy)
      if (canopies.size > 0) {
        canopies.foreach(x => canopies(x._1).remove(canopy._2))
      }
      canopies += (canopy._1 -> scala.collection.mutable.Set())

      //val r = points.filter(x => x != canopy)
      //points.remove(canopyIdx)
      r = r.filter(x => x != canopy)

      for (point <- r) {
        //val point = r(idx)
        println("INNER LOOP....POINT => " + point + ", POINTS => " + r + ", POINTS_SIZE => " + r.size)

        val distance = EuclideanVectorSpace.distance(point._1, canopy._1)
        println("INNER LOOP....POINT => " + point._1 + " Canopy => " + canopy._1 + " distance => " + distance)
        if (distance <= getT1) {
          // Check if we are inserting into canopies for the first time
          //if (canopies.contains(canopy._1)) {
          canopies(canopy._1).add(point._2)
          //} else {
          //  canopies += (canopy._1 -> scala.collection.mutable.Set(point._2))
          //}
        }
        if (distance < getT2) {
          //points.remove(idx)
          r = r.filter(x => x != point)
          println("Point removed => " + point)
        }
        println("...CANOPIES SO FAR => " + canopies)
      }
    }
    // Add self to the list of canopy edges.
    canopies.foreach(x => canopies(x._1).add(x._1.hashCode()))
    println("..Reached end of findCanopies. Canopies => " + canopies)
    canopies
  }

  /**
   * Sample data points randomly to pick k initial cluster centers
   * @param data
   * @return An Array of (Vector, hashcode()) as k sampled points
   */
  private def initRandom(data: RDD[(Vector, Int)]): Array[(Vector, Int)] = {
    val random = new XORShiftRandom(this.seed)
    data.takeSample(false, k, new XORShiftRandom(this.seed).nextInt())
  }

  /**
   * For a given point "x" and a set of cluster centers, find which cluster center and
   * the point are both within a canopy. If such a co-occurrence cannot be found, return -1.
   * @param canopies
   * @param centers
   * @param x
   * @return
   */
  private def isWithinCanopy(canopies: mutable.Map[Vector, mutable.Set[Int]],
                             centers: Array[(Vector, Int)],
                             x: (Vector, Int)): Int = {
    var index = 0
    breakable {
      for (center <- centers) {
        for ((k, v) <- canopies) {
          //println("isWithinCanopy: " + "canopy -> k " + k + " , v => " + v + ", center -> " +
          //center + " x -> " + x )
          if (v.contains(center._2) && (v.contains(x._2))) {
            //println("isWithinCanopy: " + "canopy -> k " + k + " , v => " + v + ", center -> " +
            //center + " x -> " + x + " , index -> " + index)
            break
          }
        }
        index += 1
      }
    }
    if (index == centers.size) {
      index = -1
    }
    println("isWithinCanopy: " + "canopies -> " + canopies + " centers -> " + centers +
      " x -> " + x + "index -> " + index)
    index
  }
}

/**
 * User-callable methods for running Canopy k-Means package
 */

object CanopyKMeans {

  /**
   * Builds a canopy clustering model with all parameters specified by the user
   *
   * @param sc Spark Context
   * @param input Location of file(s) with input data points
   * @param k Number of clusters
   * @param maxIterations Maximum number of iterations
   * @param epsilon Distance threshold to determine convergence
   * @param seed Seed value for randomly picking k initial centers
   * @param t1 Distance from canopy center beyond which points can belong to other canopies
   * @param t2 Distance from canopy center within which all points belong to same canopy
   * @return
   */
  def train(
             sc: SparkContext,
             input: String,
             k: Int,
             maxIterations: Int,
             epsilon: Double,
             seed: Int,
             t1: Double,
             t2: Double): KMeansModel = {
    new CanopyKMeans()
      .setK(k)
      .setMaxIterations(maxIterations)
      .setEpsilon(epsilon)
      .setSeed(seed)
      .setT1(t1)
      .setT2(t2).runAlgorithm(sc, input)
  }

  /**
   * Builds a canopy clustering model with a mix of default parameters and parameters specified
   * by the user
   *
   * @param sc Spark Context
   * @param input Location of file(s) with input data points
   * @param k Number of clusters
   * @param t1 Distance from canopy center beyond which points can belong to other canopies
   * @param t2 Distance from canopy center within which all points belong to same canopy
   * @return
   */
  def train(
             sc: SparkContext,
             input: String,
             k: Int,
             t1: Double,
             t2: Double): KMeansModel = {
    if (t1 <= t2) {
      println("Parameter T1 (" + t1 + ") must be larger than T2 (" + t2 + "). Run aborted.")
      sc.stop()
      sys.exit()
    }
    train(sc, input, k, 4, 1.e-04, new scala.util.Random().nextInt(), t1, t2)
  }
}
