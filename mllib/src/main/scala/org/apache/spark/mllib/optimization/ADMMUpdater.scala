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

import org.apache.spark.rdd.RDD
import org.apache.spark.util.Vector


/**
 * This trait is implemented by different ADMM algorithms, and is
 * passed to the main ADMMOptimizer class, which calls the appropriate
 * update methods at the appropriate stage.
 *
 * See Chapter 8, Distributed Model Fitting
 * of http://www.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
 * for the mathematical background and introduction of these methods.
 */
trait ADMMUpdater {
  def xUpdate(state: ADMMState): ADMMState

  def zUpdate(states: RDD[ADMMState]): RDD[ADMMState]

  def uUpdate(state: ADMMState): ADMMState =
    state.copy(u = state.u + state.x - state.z)
}

object ADMMUpdater {
  /**
   * Implementation of zUpdate, shared by L1-regularized logistic
   * regression and lasso regression.
   */
  def linearZUpdate(lambda: Double, rho: Double)(states: RDD[ADMMState]): RDD[ADMMState] = {
    val numStates = states.count
    // TODO(tulloch) - remove this epsilon > 0 hack?
    val epsilon = 0.00001 // avoid division by zero for shrinkage

    // TODO(tulloch) - make sure this only sends x, u to the reducer
    // instead of the full ADMM state.
    val xBar = average(states.map(_.x))
    val uBar = average(states.map(_.u))

    val zNew = Vector((xBar + uBar)
      .elements
      .zipWithIndex
      .map{case(el, index) => {
        // Eq (3) in
        // http://intentmedia.github.io/assets/2013-10-09-presenting-at-ieee-big-data
        // /pld_js_ieee_bigdata_2013_admm.pdf
        // Don't regularize the intercept term
        if (index == 0) el else shrinkage(lambda / (rho * numStates + epsilon))(el)
      }})

    states.map(state => state.copy(z = zNew))
  }

  /**
   * Eq (4.2) in http://www.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
   */
  def shrinkage(kappa: Double)(v: Double) =
    math.max(0, v - kappa) - math.max(0, -v - kappa)

  /**
   * Component-wise average of an RDD of vectors
   */
  def average(updates: RDD[Vector]): Vector = {
    updates.reduce(_ + _) / updates.count
  }
}
