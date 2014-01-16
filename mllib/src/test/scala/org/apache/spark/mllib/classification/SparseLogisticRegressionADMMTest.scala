/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.mllib.classification

import org.apache.spark.SparkContext
import org.apache.spark.mllib.optimization.ADMMState
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.util.Vector
import scala.util.Random
import org.scalatest._
import org.scalacheck.Gen
import prop._
import matchers._

class SparseLogisticRegressionWithADMMCases extends FunSuite with ShouldMatchers {
  test("gradient of objective function points in correct direction") {
    val state = ADMMState(
      points = Array(LabeledPoint(label = 1.0, features = Array(1.0))),
      initialWeights = Array(0.0)
    )

    val updater = SparseLogisticRegressionADMMUpdater(rho = 0.0, lambda = 0.0)

    val stepSize = 0.05
    val weights = Vector(1.0)

    val objective = updater.objective(state)(weights)
    val grad = updater.gradient(state)(weights)

    val newWeights = stepSize * grad + weights
    val newObjective = updater.objective(state)(newWeights)

    // gradient points in the right direction
    newObjective should be > objective
    // objective function is convex.
    newObjective should be > (objective + grad(0) * stepSize)
  }
}


class SparseLogisticRegressionWADMMUpdaterSpecification
    extends PropSpec
    with ShouldMatchers
    with GeneratorDrivenPropertyChecks {
  import SparseLogisticRegressionADMMUpdater._

  val near = (v: Double) => v plusOrMinus 0.01

  property("logPhi approximation is valid") {
    forAll { (margin: Double) =>
      whenever (math.abs(margin) < 1E10) {
        logPhi(margin) should be (near(math.log(1.0 / (1.0 + math.exp(-margin)))))
      }
    }
  }

  property("phi approximation is valid") {
    forAll { (margin: Double) =>
      whenever (math.abs(margin) < 1E10) {
        phi(margin) should be (near((1.0 / (1.0 + math.exp(-margin)))))
      }
    }
  }

  val clampedTriple = for {
    left <- Gen.choose(-1000.0, 1000.0)
    right <- Gen.choose(left, left + 1000.0)
    middle <- Gen.choose(-5000.0, 5000.0)
  } yield (left, right, middle)

  property("clampToRange is well behaved") {
    forAll (clampedTriple) { (t: (Double, Double, Double)) =>
      val clamped = clampToRange(t._1, t._2)(t._3)
      clamped should be >= t._1
      clamped should be <= t._2
      if (clamped > t._1 && clamped < t._2) {
        clamped should equal (t._3)
      }
    }
  }
}

class SparseLogisticRegressionWithADMMSpecification
    extends PropSpec
    with BeforeAndAfterAll
    with ShouldMatchers
    with TableDrivenPropertyChecks
    with GeneratorDrivenPropertyChecks {
  @transient private var sc: SparkContext = _

  override def beforeAll() {
    sc = new SparkContext("local[8]", "test")
  }

  override def afterAll() {
    sc.stop()
    System.clearProperty("spark.driver.port")
  }

  // Generate input of the form Y = logistic(offset + scale*X)
  // Copypasta'd from the Spark LogisticRegressionSuite.scala
  def generateLogisticInput(
      offset: Double,
      scale: Double,
      nPoints: Int,
      seed: Int): Seq[LabeledPoint]  = {
    val rnd = new Random(seed)
    val x1 = Array.fill[Double](nPoints)(rnd.nextGaussian())

    // NOTE: if U is uniform[0, 1] then ln(u) - ln(1-u) is Logistic(0,1)
    val unifRand = new Random(45)
    val rLogis = (0 until nPoints).map { i =>
      val u = unifRand.nextDouble()
      math.log(u) - math.log(1.0-u)
    }

    // y <- A + B*x + rLogis()
    // y <- as.numeric(y > 0)
    val y: Seq[Int] = (0 until nPoints).map { i =>
      val yVal = offset + scale * x1(i) + rLogis(i)
      if (yVal > 0) 1 else 0
    }

    val testData = (0 until nPoints).map(i => LabeledPoint(y(i), Array(x1(i))))
    testData
  }

  property("recovers A + BX from generated data") {
    forAll(Table(
      ("a", "b"),
      (0.0, 0.0),
      (1.0, -1.0),
      (0.05, 1.05),
      (3.0, 3.0)
    )) { (a: Double, b: Double) =>
      val nPoints = 10000
      val numIterations = 5
      val lambda = 0.0
      val rho = 0.000

      val testData = generateLogisticInput(a, b, nPoints, 42)

      val testRDD = sc.parallelize(testData, 5)
      testRDD.cache()
      val lr = new SparseLogisticRegressionWithADMM(numIterations, lambda, rho)

      val model = lr.run(testRDD)

      val near = (v: Double) => v plusOrMinus(math.max(0.05, v * 0.05))
      model.intercept should be (near(a))
      model.weights(0) should be (near(b))
    }
  }
}
