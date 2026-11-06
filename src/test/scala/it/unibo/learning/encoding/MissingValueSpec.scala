package it.unibo.learning.encoding

import it.unibo.testing.AgentStates
import it.unibo.testing.AgentStates.*
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class MissingValueSpec extends AnyFlatSpec with Matchers:
  "A node" should "miss a value while its output is not finite" in {
    MissingValue.of(isolated) shouldBe true
    MissingValue.of(stableBackbone) shouldBe false
  }

  it should "miss a value before its first output" in {
    MissingValue.of(AgentStates.of(me = 0)) shouldBe true
  }

  "The missing-value feature" should "be one for a node without a value and zero otherwise" in {
    MissingValue.feature(isolated) shouldBe 1.0f
    MissingValue.feature(stableBackbone) shouldBe 0.0f
  }
