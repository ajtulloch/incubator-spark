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

import org.apache.spark.mllib.optimization.{ADMMOptimizer, ADMMState, ADMMUpdater}
import org.apache.spark.mllib.regression.{GeneralizedLinearAlgorithm, LabeledPoint}
import org.apache.spark.mllib.util.DataValidators
import org.apache.spark.rdd.RDD
import org.apache.spark.Logging
import org.apache.spark.util.Vector
import org.jblas.DoubleMatrix

case class PrimalDual(primal: Vector, dual: Vector)

case class DualCoordinateDescent(state: ADMMState, rho: Double, cee: Double)
    extends DescentFunction[PrimalDual]
    with Logging {
  val gramMatrix = fillSquareMatrix(state.points.length, (i: Int, j: Int) => {
      val pi = state.points(i)
      val pj = state.points(j)
      pi.label * pj.label * (Vector(pi.features) dot Vector(pj.features)) + rho / (2 * cee)
  })

  val v = state.z - state.u
  val b = state.points.map{lp => 1 - lp.label * (v dot Vector(lp.features))}

  def initialPrimalDual: PrimalDual = {
    val initialDual = state.dual.getOrElse(ADMMState.zeroes(state.points.length))
    val initialPrimal = v + state.points
      .zip(initialDual.elements)
      .map{case(lp, dualValue) => (lp.label * dualValue) * Vector(lp.features)}
      .reduce(_ + _)
    logInfo("Initial Primal: %s, Initial Dual: %s".format(initialPrimal, initialDual))
    PrimalDual(initialPrimal, initialDual)
  }

  def descend(pd: PrimalDual): PrimalDual = {
    val deltas = state.points
      .zip(pd.dual.elements)
      .map{case(LabeledPoint(label, features), dualValue) => {
        val g = label * (pd.primal dot Vector(features)) - 1 + (rho / (2 * cee)) * dualValue
        val pg = if (dualValue == 0) math.min(0, g) else g
        if (pg != 0) {
          val qii = Vector(features) dot Vector(features)
          val dii = rho / (2 * cee)
          val newDualValue = math.max(dualValue - g / (qii + dii), 0)
          val primalDelta = (newDualValue - dualValue) * label * Vector(features)
          (primalDelta, newDualValue)
        } else {
          // TODO(tulloch) - just return None here and avoid the zero
          // vector allocation?
          val primalDelta = ADMMState.zeroes(pd.primal.length)
          (primalDelta, dualValue)
        }
      }}

    val newPrimal = pd.primal + deltas.map(_._1).reduce(_ + _)
    val newDual = Vector(deltas.map(_._2))
    PrimalDual(newPrimal, newDual)
  }

  def objective(pd: PrimalDual): Double = {
    val loss = for {
      i <- 0 to (pd.dual.length - 1)
      j <- 0 to (pd.dual.length - 1)
    } yield pd.dual(i) * pd.dual(j) * gramMatrix.get(i, j)

    val regular = Vector(b) dot pd.dual
    (1 / (2 * rho)) * loss.sum + regular
  }

  def fillSquareMatrix(n: Int, f: (Int, Int) => Double): DoubleMatrix = {
    val result = new DoubleMatrix(n, n)
    for (i <- 0 to n - 1; j <- 0 to n - 1) {
      result.put(i, j, f(i, j))
    }
    result
  }
}

case class SVMADMMUpdater(
  rho: Double,
  cee: Double,
  maxNumIterations: Int = 5,
  absoluteTolerance: Double = 1E-4,
  relativeTolerance: Double = 1E-2) extends ADMMUpdater with Logging {

  def xUpdate(state: ADMMState): ADMMState = {
    val optimizer = DescentMinimizer[PrimalDual](
      maxNumIterations, absoluteTolerance, relativeTolerance)
    val descender = DualCoordinateDescent(state, rho, cee)

    val minimizer = optimizer.minimize(descender, descender.initialPrimalDual)

    // returned updated state and dual variables
    logError("Minimizer: %s".format(minimizer))
    state.copy(x = minimizer.primal, dual = Some(minimizer.dual))
  }

  def zUpdate(states: RDD[ADMMState]): RDD[ADMMState] = {
    val numerator = states.map(state => state.x + state.u).reduce(_ + _)
    val denominator = states.count + (1.0 / rho)
    val newZ = numerator / denominator
    states.map(_.copy(z = newZ))
  }
}

class SVMWithADMM(
  val numIterations: Int,
  val rho: Double,
  val cee: Double)
    extends GeneralizedLinearAlgorithm[SVMModel]
    with Serializable {

  override val optimizer = new ADMMOptimizer(
    numIterations,
    new SVMADMMUpdater(rho = rho, cee = cee))

  override val validators = List(DataValidators.classificationLabels)

  override def createModel(
    weights: Array[Double],
    intercept: Double): SVMModel = new SVMModel(weights, intercept)
}

object SVMWithADMM {
  def train(
    input: RDD[LabeledPoint],
    numIterations: Int,
    rho: Double,
    cee: Double) = {
    new SVMWithADMM(numIterations, rho, cee).run(input)
  }
}
