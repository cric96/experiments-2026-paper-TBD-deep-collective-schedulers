package it.unibo.alchemist.model.reactions

import it.unibo.alchemist.model.reactions.NetworkObserver.{sameOutput, Observation}
import it.unibo.mobility.NodeObservation
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class NetworkObserverSpec extends AnyFlatSpec with Matchers:
  "Two outputs" should "be the same only when a re-observation would record nothing new" in {
    sameOutput(Some(3.5), Some(3.5)) shouldBe true
    sameOutput(None, None) shouldBe true
    sameOutput(Some(Double.NaN), Some(Double.NaN)) shouldBe true
    sameOutput(Some(0.0), Some(-0.0)) shouldBe false
    sameOutput(Some(1.0), None) shouldBe false
    sameOutput(None, Some(1.0)) shouldBe false
  }

  "An observation" should "be reused only at the same instant, network version and population" in {
    val observation = Observation(
      10.0,
      version = 3L,
      outputs = Map(1 -> Some(2.0), 2 -> None),
      observations = Map(1 -> NodeObservation(0.0, reachable = true, cost = 0.0))
    )
    observation.sees(10.0, 3L, Set(1, 2)) shouldBe true
    observation.sees(10.25, 3L, Set(1, 2)) shouldBe false
    observation.sees(10.0, 4L, Set(1, 2)) shouldBe false
    observation.sees(10.0, 3L, Set(1)) shouldBe false
  }
end NetworkObserverSpec
