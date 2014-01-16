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

package org.apache.spark.mllib.optimization

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
 * Alternating Direction Method of Multipliers (ADMM) optimization
 * routine for Spark.
 * @param numIterations number of iterations of ADMM to run.
 * @param updater ADMMUpdater that computes the primal/dual updates.
 */
class ADMMOptimizer(
  val numIterations: Int,
  val updater: ADMMUpdater)
    extends Optimizer with Serializable {

  override def optimize(
    data: RDD[(Double, Array[Double])],
    initialWeights: Array[Double]): Array[Double] = {
    val numPartitions = data.partitions.length

    val admmStates = data
      .map{case(zeroOnelabel, features) => {
        // The input points are 0,1 - we map to (-1, 1) for consistency
        // with the presentation in
        // http://www.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
        val scaledLabel = 2 * zeroOnelabel - 1
        LabeledPoint(scaledLabel, features)
      }}
      .groupBy{lp => {
        // map each data point to a given ADMM partition
        lp.hashCode() % numPartitions
      }}
      .map{case (_, points) => ADMMState(points, initialWeights) }

    // Run numIterations of runRound
    val finalStates = (1 to numIterations)
      .foldLeft(admmStates)((s, _) => runRound(s))

    // return average of final weight vectors across the partitions
    ADMMUpdater.average(finalStates.map(_.x)).elements
  }

  private def runRound(states: RDD[ADMMState]): RDD[ADMMState] = {
    // run the updates sequentially. Note that the xUpdate and uUpdate
    // happen in parallel, while the zUpdate collects the xUpdates
    // from the mappers.
    val xUpdate = (s: RDD[ADMMState]) => s.map(updater.xUpdate)
    val zUpdate = (s: RDD[ADMMState]) => updater.zUpdate(s)
    val uUpdate = (s: RDD[ADMMState]) => s.map(updater.uUpdate)

    (xUpdate andThen zUpdate andThen uUpdate)(states)
  }
}
