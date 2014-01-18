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
import org.scalatest._
import org.scalacheck.Gen
import prop._
import matchers._

class SVMWithADMMSpecification
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
      val rho = 0.001
      val cee = 1.0

      val testData = SVMSuite.generateSVMInput(a, Array(b), nPoints, 42)

      val testRDD = sc.parallelize(testData, 5)
      testRDD.cache()
      val lr = new SVMWithADMM(numIterations, rho, cee)

      val model = lr.run(testRDD)

      val near = (v: Double) => v plusOrMinus(math.max(0.05, v * 0.05))
      model.intercept should be (near(a))
      model.weights(0) should be (near(b))
    }
  }
}
