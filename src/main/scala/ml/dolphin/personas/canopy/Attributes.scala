package ml.dolphin.personas.canopy

/**
 * Definitions of an attribute list and an attribute map.
 *
 * @author Abhijit Bose
 * @version 1.0 06/24/2015
 * @since 1.0 06/24/2015
 *
 */

class Attributes(xL: List[((String, Int, Int), Int)], xM: Map[String, (Int, Int, Int)]) {
  var l: List[((String, Int, Int), Int)] = xL
  var m: Map[String, (Int, Int, Int)] = xM
}
