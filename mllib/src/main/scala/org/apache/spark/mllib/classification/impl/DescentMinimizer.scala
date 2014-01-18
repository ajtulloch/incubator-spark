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

import org.apache.spark.Logging
import org.apache.spark.util.Vector

trait DescentFunction[T] {
  def descend(current: T): T
  def objective(current: T): Double
}

/**
 * Class that minimizes an objective function to a specified tolerance
 * within a maximum number of iterations by iteratively desending from
 * a given point. 
 */
case class DescentMinimizer[T](
  maxNumIterations: Int,
  absoluteTolerance: Double,
  relativeTolerance: Double) extends Logging {

  // Have we made sufficient progress (according to our objective
  // function) from (currentIterate, nextIterate)?
  private def sufficientProgress(f: DescentFunction[T])(positions: Stream[T]): Boolean = {
    val objectives = positions.map(f.objective)
    val (current, next) = (objectives.head, objectives.last)
    // termination condition
    current - next > absoluteTolerance && (current - next) > current * relativeTolerance
  }

  def iterates(f: DescentFunction[T], initial: T): Stream[T] =
    Stream.from(1).scanLeft(initial)((pd, _) => f.descend(pd))

  def minimize(f: DescentFunction[T], initial: T): T = {
    val getFinalIterate = (iterates: Stream[T]) => {
      val candidates = iterates
        // do at most maxNumIterations
        .take(maxNumIterations)
        // Pairs of (current, next) iterate pairs
        .sliding(2)
        // until we stop sufficient improvement in the objective function
        .takeWhile(sufficientProgress(f))
        // take the last element of the tuple (the last improvement
        // that made sufficient progress)
        .map(_.last)
        .toStream

      candidates.lastOption
        // take the last element (if it exists, or just return the
        // initial state)
        .getOrElse(iterates.head)
    }

    val iterations = iterates(f, initial)
    getFinalIterate(iterations)
  }
}
