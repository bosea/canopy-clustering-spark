Code Walkthrough


Reading Input Data

Take a look at PersonaCommon.readHivePoints and PersonaCommon.readCsvMetadata
methods, for reading in tab-delimited (format for Hive tables) and 
comma-delimited CSV files, and creating a RDD of Vector's out of them. This is standard boilerplate
code for Spark. What is slightly different is that I also calculate a hash of 
each row and store each row as a tuple of (Vector, Int). Canopy algorithm calls for keeping track of
which data points are within a given distance of a canopy center. Instead of storing the original 
features (i.e. columns of the dataset) of a Vector, I store just the hash to keep track of these proximities. For large datasets,
a small amount of computation to calculate and compare hashes is usually cheaper than trying to store all the data objects in memory. So, our data is now stored as RDD[(Vector, Int)].

Vector Space operations

Take a look at VectorSpace.scala and EuclideanVectorSpace.scala. 

The basic operations on a Vector space are defined in VectorSpace.scala 
as a trait:

trait  Note the generic type "A" used to define the attributes of the
points. We will later extend it to Vector since our data points are defined
as (Vector, Int).


distance(): calculate distance between a pair of points 

distance() and angle() between two points, and  
groupwise: calculate centroid() of a sequence of points, and calculate the
closest() of a group of points to a given point, as a Scala trait. Note the
generic type definition as in "trait VectorSpace[A] {...".  

I extend this to an Euclidean space of Vectors in EuclideanVectorSpace.scala:

object EuclideanVectorSpace extends VectorSpace[Vector] {
..

and define the methods  

