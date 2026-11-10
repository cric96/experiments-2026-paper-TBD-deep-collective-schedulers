package it.unibo.alchemist.model.reactions

import it.unibo.alchemist.model.reactions.NetworkObserver.{contextOf, eventAge, sameOutput, Observation}
import it.unibo.learning.abstractions.CriticContext
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

  "The context of a decision" should "measure the node's own loss, its charged cost and the network's mean loss" in {
    val scale = 400.0
    val observations = Map(
      1 -> NodeObservation(100.0, reachable = true, cost = 600.0),
      2 -> NodeObservation(Double.PositiveInfinity, reachable = false, cost = Double.PositiveInfinity),
      3 -> NodeObservation(0.0, reachable = true, cost = 0.0),
      4 -> NodeObservation(200.0, reachable = true, cost = 200.0)
    )
    val networkLoss = (0.25 + 1.0 + 0.0 + 0.5) / 4
    val unreached = 1.0 / 4
    val eventAge = 0.5
    contextOf(1, observations, scale, eventAge) shouldBe
      Some(CriticContext(0.25, 1.5, networkLoss, unreached, eventAge))
    contextOf(2, observations, scale, eventAge) shouldBe
      Some(CriticContext(1.0, 1.0, networkLoss, unreached, eventAge))
    contextOf(5, observations, scale, eventAge) shouldBe None
  }

  "The age of the last event" should "grow with the seconds since it, up to one after the horizon" in {
    eventAge(0.0) shouldBe 0.0
    eventAge(CriticContext.EventAgeHorizon / 2) shouldBe 0.5
    eventAge(3 * CriticContext.EventAgeHorizon) shouldBe 1.0
  }
end NetworkObserverSpec
