package it.unibo.mobility

import org.apache.commons.math3.random.Well19937c
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class MobilitySpec extends AnyFlatSpec with Matchers:
  "The mobile objective" should "ignore errors within tolerance and cap the remaining loss" in {
    MobileObjective.loss(4.0, 5.0, 500.0) shouldBe 0.0
    MobileObjective.loss(15.0, 5.0, 500.0) shouldBe 0.02
    MobileObjective.loss(1000.0, 5.0, 500.0) shouldBe 1.0
    MobileObjective.loss(Double.PositiveInfinity, 5.0, 500.0) shouldBe 1.0
  }

  "Reflected Brownian motion" should "handle multiple boundary crossings" in {
    BrownianMotion.reflect(-10.0, 500.0) shouldBe 10.0
    BrownianMotion.reflect(510.0, 500.0) shouldBe 490.0
    BrownianMotion.reflect(1510.0, 500.0) shouldBe 490.0
  }

  it should "be stationary at zero diffusion and reproduce seeded trajectories" in {
    val still = BrownianMotion(new Well19937c(42), 500.0)
    still.move(100.0, 200.0, 0.0, 0.25) shouldBe (100.0, 200.0)
    val first = BrownianMotion(new Well19937c(42), 500.0)
    val second = BrownianMotion(new Well19937c(42), 500.0)
    (1 to 20).foreach { _ =>
      first.move(100.0, 200.0, 2.0, 0.25) shouldBe second.move(100.0, 200.0, 2.0, 0.25)
    }
  }

  it should "match the diffusion variance" in {
    val motion = BrownianMotion(new Well19937c(123), 500.0)
    val squaredDistances = (1 to 10000).map { _ =>
      val (x, y) = motion.move(250.0, 250.0, 2.0, 0.25)
      math.pow(x - 250.0, 2) + math.pow(y - 250.0, 2)
    }
    squaredDistances.sum / squaredDistances.size shouldBe (2.0 +- 0.1)
  }

  "An error integral" should "charge held outputs during sleep, including the final partial interval" in {
    val integral = ErrorIntegral(Seq(0.0, 5.0), 500.0)
    integral.observe(0.0, 15.0, reachable = true)
    integral.observe(2.0, 25.0, reachable = false)
    integral.advance(5.0)
    integral.cost(0.0) shouldBe (0.21 +- 1e-12)
    integral.cost(5.0) shouldBe (0.16 +- 1e-12)
    integral.withinTime(5.0) shouldBe 0.0
    integral.reachableTime shouldBe 2.0
    integral.advance(5.5)
    integral.cost(5.0) shouldBe (0.18 +- 1e-12)
  }

  "A mobile episode" should "close terminal sleeps using their actual duration and relabel every tolerance" in {
    import _root_.it.unibo.learning.abstractions.AgentState
    import _root_.it.unibo.learning.abstractions.AgentState.NeighborInfo
    import _root_.it.unibo.learning.agents.offline.Relabelling
    import _root_.it.unibo.learning.encoding.{MobileEncoding, SpatialTemporal}
    import scala.collection.immutable.Queue
    val config = MobilityConfig(1, 0, tolerance = 30.0, trainingEpisodes = 1, episodes = 2)
    val episode = MobileEpisode(config, 0, 0.0)
    val state = AgentState(0, Queue(Map(0 -> NeighborInfo(0.0, 0.0, 1.0, 1.0))))
    episode.observe(0.0, Map(0 -> NodeObservation(15.0, reachable = true, cost = 15.0)))
    episode.observe(1.0, Map(0 -> NodeObservation(15.0, reachable = true, cost = 15.0)))
    episode.decision(state, 5.0, 1.0)
    episode.observe(3.0, Map(0 -> NodeObservation(25.0, reachable = true, cost = 25.0)))
    val trajectory = episode.finish(4.0).head
    val outcome = trajectory.head.mobileOutcome.get
    outcome.duration shouldBe 3.0
    outcome.terminal shouldBe true
    outcome.errorIntegrals(5.0) shouldBe (0.08 +- 1e-12)
    val labelled = Relabelling.relabel(
      trajectory,
      0.95,
      0.9,
      MobileEncoding.spatial(_, 0.95, 5.0),
      SpatialTemporal.Feature.Beta,
      mobile = true
    )
    labelled.foreach { step =>
      step.discount shouldBe (math.pow(0.9, 3.0).toFloat +- 1e-6f)
      step.terminalFlag shouldBe 1.0f
    }
    labelled.map(_.state(SpatialTemporal.FeatureDim)).distinct.length shouldBe config.trainingTolerances.size
    outcome.errorIntegrals.keySet should not contain 30.0
    labelled.map(_.state(SpatialTemporal.Feature.Beta)).toSet should contain(0.95f)
    MobileEncoding.graph(state, 0.95, 5.0).length shouldBe
      MobileEncoding.SelfDim + 8 * MobileEncoding.NeighDim
  }

  it should "charge decisions with the cost signal at its loss scale, and report the held error" in {
    import _root_.it.unibo.learning.abstractions.AgentState
    import _root_.it.unibo.learning.abstractions.AgentState.NeighborInfo
    import scala.collection.immutable.Queue
    val config = MobilityConfig(1, 0, trainingEpisodes = 1, episodes = 2, lossScale = 50.0)
    val episode = MobileEpisode(config, 0, 0.0)
    val state = AgentState(0, Queue(Map(0 -> NeighborInfo(0.0, 0.0, 1.0, 1.0))))
    episode.observe(0.0, Map(0 -> NodeObservation(15.0, reachable = true, cost = 10.0)))
    episode.decision(state, 2.0, 0.0)
    episode.observe(2.0, Map(0 -> NodeObservation(15.0, reachable = true, cost = 10.0)))
    val costs = episode.finish(2.0).head.head.mobileOutcome.get.errorIntegrals
    costs(0.0) shouldBe (2.0 * 10.0 / 50.0 +- 1e-12)
    costs(5.0) shouldBe (2.0 * 5.0 / 50.0 +- 1e-12)
    val normalizedError = episode.metrics(2.0)(1)
    normalizedError shouldBe (15.0 / MobileEpisode.MobileEncodingScale +- 1e-12)
  }

  it should "count the rounds and integrate the absolute error of every node" in {
    import _root_.it.unibo.learning.abstractions.AgentState
    import _root_.it.unibo.learning.abstractions.AgentState.NeighborInfo
    import scala.collection.immutable.Queue
    val episode = MobileEpisode(MobilityConfig(1, 0, trainingEpisodes = 1, episodes = 2), 0, 0.0)
    val state = AgentState(0, Queue(Map(0 -> NeighborInfo(0.0, 0.0, 1.0, 1.0))))
    episode.observe(0.0, Map(0 -> NodeObservation(15.0, reachable = true, cost = 15.0)))
    episode.decision(state, 1.0, 0.0)
    episode.decision(state, 1.0, 1.0)
    episode.observe(2.0, Map(0 -> NodeObservation(25.0, reachable = true, cost = 25.0)))
    episode.nodeTotals(0) shouldBe NodeTotals(rounds = 2, absoluteError = 2.0 * 15.0)
    episode.nodeTotals(1) shouldBe NodeTotals(rounds = 0, absoluteError = 0.0)
  }
end MobilitySpec
