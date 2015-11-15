package ml.dolphin.personas.canopy

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD

import scala.io.Source

/**
 * Common methods for canopy clustering. Still to be developed. mostly input manipulations.
 *
 * @author Abhijit Bose
 * @version 1.0 06/24/2015
 * @since 1.0 06/24/2015
 */

object PersonaCommon {

  /**
   * Returns a Attribute object given an input file containing attribute metadata in CSV format.
   *
   * @param input Input file with attributes metadata
   * @return attribute: Attribute containing Attribute object that has information about all
   *         attributes and their processing information
   */
  def readCsvMetadata(input: String): Attributes = {

    /*
   * Each line of input CSV file with attributes metadata has the following:
   *   name, flag, flatten
   *   where:
   *     name: String = name of the attribute
   *     flag: Boolean == 1 => the attribute will be used for clustering
   *     flatten: Boolean == 1 => the attribute is categorical and will be flattened.
   */

    val schema = Source.fromFile(input).getLines()
    // List of attributes in the same order they appear in the data
    val aList = schema.map(s => {
      val elem = s.split(',')
      (elem(0).trim, elem(1).toInt, elem(2).toInt)
    }).zipWithIndex.toList
    // Map of attributes
    val aMap = aList.map(s => {
      s._1._1 ->(s._1._2, s._1._3, s._2)
    }).toMap
    new Attributes(aList, aMap)
  }

  /**
   * Reads data points in Hive format. Each row is converted into a Vector along with its hashcode.
   * @param sc
   * @param input
   * @return
   */
  def readHivePoints(sc: SparkContext, input: String): RDD[(Vector, Int)] = {
    val data = sc.textFile(input)
    val rows = data.map(s => {
      val buffer = s.split('\t').toBuffer
      val features = Vectors.dense(buffer.map(_.toDouble).toArray)
      (features, features.hashCode())
    })
    rows
  }


  /* Flattening is a procedure by which each distinct value of a categorical attribute is
  *   converted into an additional Boolean attribute. For example,
  *   city = ["new york", "london", "delhi", "tokyo"] are distinct values of attribute "city"
  *   Flattened Boolean attributes created: city_new_york, city_london, city_delhi, city_tokyo
    *
    *   The attributes are named by concatenating the parent attribute with the attribute value
    *   with "_" in between. Any blank space(s) in the attribute value will be converted into
    *   "_" as shown in the above example.
    */


  /**
   *
   * @param num
   * @return
   */
  def toIntegerBucket(num: Double): Int = {
    if (num < 0.0) {
      println("ERROR: ml.dolphin.personas.PersonaCommon: toIntegerBucket(..) cannot handle negative values")
      sys.exit()
    }
    val leftover = num - num.floor
    if (leftover < 0.5)
      num.floor.toInt
    else
      num.ceil.toInt
  }


}
