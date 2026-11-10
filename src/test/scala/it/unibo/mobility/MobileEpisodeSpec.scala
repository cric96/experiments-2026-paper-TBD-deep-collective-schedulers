package it.unibo.mobility

import it.unibo.learning.abstractions.{CriticContext, MobileOutcome, Stratification}
import it.unibo.testing.AgentStates
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class MobileEpisodeSpec extends AnyFlatSpec with Matchers:
  private val config = MobilityConfig(11, 0, tolerance = 5.0, trainingEpisodes = 1, episodes = 2)

  private def observation(error: Double): NodeObservation = NodeObservation(error, reachable = true, cost = error)

  private val initial = Map(1 -> observation(40.0), 2 -> observation(3.0), 3 -> observation(80.0))
  private val updated = initial + (1 -> observation(2.0))

  "A mobile episode" should "record only the changed nodes at an instant it has already observed" in {
    val everyNode = MobileEpisode(config, 1, 0.0)
    val changesOnly = MobileEpisode(config, 1, 0.0)
    Seq(everyNode, changesOnly).foreach { episode =>
      episode.observe(0.0, initial)
      episode.observe(120.0, initial)
    }
    everyNode.observe(120.0, updated)
    changesOnly.observeChanges(120.0, updated.view.filterKeys(_ == 1).toMap, updated)
    Seq(everyNode, changesOnly).foreach(_.observe(130.0, updated))
    // Recovery is judged on every node: the changed one alone would be within tolerance.
    changesOnly.metrics(130.0).map(_.toString) shouldBe everyNode.metrics(130.0).map(_.toString)
    everyNode.metrics(130.0)(MobileEpisodeSpec.RecoveryCensored) shouldBe 1.0
    (1 to 3).map(changesOnly.nodeTotals) shouldBe (1 to 3).map(everyNode.nodeTotals)
  }

  it should "measure the training costs of a decision in units of the cost reference time" in {
    val node = AgentStates.stableBackbone
    def costs(referenceTime: Double): Map[Double, Double] =
      val episode = MobileEpisode(config.copy(costReferenceTime = referenceTime), 0, 0.0)
      episode.observe(0.0, initial)
      episode.decision(node, period = 2.0, now = 0.0)
      episode.observe(2.0, initial)
      episode.finish(2.0).flatten.flatMap(_.mobileOutcome).map(_.errorIntegrals).headOption.getOrElse(Map.empty)
    val seconds = costs(MobileObjective.ReferenceTime)
    seconds.keySet shouldBe config.trainingTolerances.toSet
    costs(MobileEpisodeSpec.QuarterSecond) shouldBe seconds.view.mapValues(_ / MobileEpisodeSpec.QuarterSecond).toMap
  }

  private val network = config.copy(costSignal = CostSignal.Network, trainingTolerances = Seq(0.0))
  private val node = AgentStates.stableBackbone

  /** The outcomes of the decisions of `node`, taken at `decisions` with `contexts`, in an episode that observes a cost
    * `rate` times the loss scale at every second until `end`.
    */
  private def outcomes(
      config: MobilityConfig,
      rate: Double => Double,
      decisions: Seq[(Double, CriticContext)],
      end: Double,
      scale: Double = 1.0
  ): Seq[MobileOutcome] =
    val episode = MobileEpisode(config, 0, 0.0, () => scale)
    def observe(time: Double): Unit =
      val error = rate(time) * config.lossScale
      episode.observe(time, Map(node.me -> NodeObservation(error, reachable = true, cost = error)))
    (0 to end.toInt).foreach { second =>
      observe(second.toDouble)
      decisions.filter(_._1 == second.toDouble).foreach((time, context) => episode.decision(node, 1.0, time, context))
    }
    episode.finish(end).flatten.flatMap(_.mobileOutcome)

  it should "discount the network cost within a sleep, from the decision" in {
    val outcome = outcomes(network, _ => 0.3, Seq(0.0 -> CriticContext.Unknown), end = 4.0).head
    val gamma = network.costDiscount
    outcome.errorIntegrals(0.0) shouldBe (0.3 * (1.0 - math.pow(gamma, 4.0)) / math.log(1.0 / gamma) +- 1e-12)
  }

  it should "charge the network cost of a single node as its own error without the discount" in {
    def costs(config: MobilityConfig) =
      outcomes(
        config,
        time => 0.1 * (1.0 + time),
        Seq(0.0 -> CriticContext.Unknown, 2.0 -> CriticContext.Unknown),
        end = 5.0
      ).map(_.errorIntegrals(0.0))
    val error = config.copy(costSignal = CostSignal.Error, trainingTolerances = Seq(0.0))
    costs(network.copy(costDiscount = 1.0)).zip(costs(error)).foreach { (networkCost, ownCost) =>
      networkCost shouldBe (ownCost +- 1e-12)
    }
  }

  it should "record the contexts of a decision and of the next one, the cost scale and the stratification" in {
    val first = CriticContext(0.1, 0.2, 0.3)
    val second = CriticContext(0.4, 0.5, 0.6)
    val recorded = outcomes(network, _ => 0.3, Seq(0.0 -> first, 2.0 -> second), end = 3.0, scale = 4.0)
    recorded.map(outcome => (outcome.context, outcome.nextContext, outcome.terminal)) shouldBe
      Seq((first, second, false), (second, second, true))
    recorded.map(_.costScale).distinct shouldBe Seq(4.0)
    recorded.map(_.stratification).distinct shouldBe Seq(Stratification.ByLocalRegime)
  }

  it should "stratify the decisions of the other signals by their cost" in {
    outcomes(config, _ => 0.3, Seq(0.0 -> CriticContext.Unknown), end = 1.0).map(_.stratification) shouldBe
      Seq(Stratification.ByCost)
  }
end MobileEpisodeSpec

object MobileEpisodeSpec:
  /** Position of the recovery-censoring flag in the metrics. */
  val RecoveryCensored: Int = 6

  /** A cost reference time of a quarter of a second, which makes the costs four times those measured in seconds. */
  val QuarterSecond: Double = 0.25
