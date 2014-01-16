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
import org.apache.spark.mllib.optimization.ADMMUpdater
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.util.Vector
import scala.util.Random
import org.scalatest._
import org.scalacheck.Gen
import prop._
import matchers._

class ADMMSpecification extends PropSpec with GeneratorDrivenPropertyChecks {
  val positiveInts = for (n <- Gen.choose(1, 100)) yield n
  val shrinkageParams = for {
    kappa <- Gen.choose(0.0, 5.0)
    v <- Gen.choose(-100.0, 100.0)
  } yield (kappa, v)

  property("zeroes is implemented correctly") {
    forAll (positiveInts) { (n: Int) =>
      val x = ADMMState.zeroes(n)
      x.length == n && x.elements.min == 0 && x.elements.max == 0
    }
  }

  property("shrinkage is implemented correctly") {
    forAll (shrinkageParams) { (kv: (Double, Double)) =>
      val (kappa, v) = kv
      val calculated =
        if (v > kappa) v - kappa else if (math.abs(v) <= kappa) 0 else v + kappa

      ADMMUpdater.shrinkage(kappa)(v) == calculated
    }
  }
}
