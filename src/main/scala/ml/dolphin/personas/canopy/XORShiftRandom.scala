package ml.dolphin.personas.canopy

//package org.apache.spark.mllib.linalg.canopy

import java.util.Random

/**
 * XORShift Random Number Generator extended from Java's Random class
 * https://en.wikipedia.org/wiki/Xorshift
 *
 * @note This method is NOT thread-safe. For safe parallel execution, a parallel pseudo random
 *       generator such as SPRNG (http://www.sprng.org) should be used to generate the seeds
 *       across the different threads.
 * @author Abhijit Bose
 * @version 1.0 06/24/2015
 * @since 1.0 06/24/2015
 *
 */

class XORShiftRandom(private var seed: Int) extends Random {

  // Default constructor for class
  def this() = this(System.nanoTime().toInt)

  // override java.util.Random.next method for getting the next pseudo-random number
  override def next(nBits: Int): Int = {
    var x = this.seed
    x ^= (x << 21);
    x ^= (x >>> 35);
    x ^= (x << 4);
    this.seed = x;
    x &= ((1 << nBits) - 1);
    x;
  }

}
