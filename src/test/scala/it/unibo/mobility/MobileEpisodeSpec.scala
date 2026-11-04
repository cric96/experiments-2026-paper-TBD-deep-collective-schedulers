package it.unibo.mobility

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
end MobileEpisodeSpec

object MobileEpisodeSpec:
  /** Position of the recovery-censoring flag in the metrics. */
  val RecoveryCensored: Int = 6
