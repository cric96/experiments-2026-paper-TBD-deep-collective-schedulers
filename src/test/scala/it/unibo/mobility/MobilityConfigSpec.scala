package it.unibo.mobility

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class MobilityConfigSpec extends AnyFlatSpec with Matchers:
  private val brownian = MobilityConfig(1, 0)

  "The run name" should "stay the same for the Brownian scenario" in {
    brownian.runName shouldBe
      "seed1_algo0_D1.0_tol5.0_beta0.95_dt0.25_train10_length200.0_sourcetrue"
  }

  it should "name the speed and the scenario of the waypoint and warm-start variants" in {
    MobilityConfig(1, 0, model = MobilityModel.Waypoint, speed = 1.5).runName shouldBe
      "seed1_algo0_V1.5_tol5.0_beta0.95_dt0.25_train10_length200.0_sourcetrue_waypoint"
    MobilityConfig(1, 0, warmStart = true).runName shouldBe
      "seed1_algo0_D1.0_tol5.0_beta0.95_dt0.25_train10_length200.0_sourcetrue_warm"
  }

  "The configuration" should "accept the algorithms of the simulation file only" in {
    MobilityConfig.Algorithms.foreach(algorithm => MobilityConfig(1, algorithm).algorithm shouldBe algorithm)
    an[IllegalArgumentException] should be thrownBy MobilityConfig(1, MobilityConfig.Algorithms.end + 1)
  }

  "The mobility level" should "cycle the training levels of the model, then use the evaluation one" in {
    val waypoint = MobilityConfig(1, 0, model = MobilityModel.Waypoint, speed = 1.5)
    (0 until 4).map(waypoint.levelAt) shouldBe Seq(0.0, 1.0, 2.0, 0.0)
    waypoint.levelAt(waypoint.trainingEpisodes) shouldBe 1.5
    (0 until 4).map(brownian.levelAt) shouldBe Seq(0.0, 0.5, 2.0, 0.0)
    brownian.levelAt(brownian.trainingEpisodes) shouldBe 1.0
  }

  it should "train the transfer controls in a static network" in {
    val transfer = MobilityConfig(1, MobilityConfig.TransferGNN, model = MobilityModel.Waypoint)
    (0 until transfer.trainingEpisodes).map(transfer.levelAt).toSet shouldBe Set(0.0)
  }

  "The halfway event" should "switch the source by default" in {
    (0 to brownian.trainingEpisodes).map(brownian.eventAt).toSet shouldBe Set(EpisodeEvent.SourceSwitch)
  }

  it should "cycle the training events, then use the evaluation one" in {
    val events = MobilityConfig(
      1,
      0,
      event = EpisodeEvent.RegionStartsMoving,
      trainingEvents = Seq(EpisodeEvent.SourceSwitch, EpisodeEvent.RegionStartsMoving)
    )
    (0 until 3).map(events.eventAt) shouldBe
      Seq(EpisodeEvent.SourceSwitch, EpisodeEvent.RegionStartsMoving, EpisodeEvent.SourceSwitch)
    events.eventAt(events.trainingEpisodes) shouldBe EpisodeEvent.RegionStartsMoving
  }

  it should "name the evaluation of a region that starts moving" in {
    MobilityConfig(1, 0, model = MobilityModel.Waypoint, warmStart = true,
      event = EpisodeEvent.RegionStartsMoving).runName shouldBe
      "seed1_algo0_V1.0_tol5.0_beta0.95_dt0.25_train10_length200.0_sourcetrue_waypoint_warm_region"
  }

  it should "name a region that moves for a while only" in {
    val config = MobilityConfig(1, 0, model = MobilityModel.Waypoint, warmStart = true,
      event = EpisodeEvent.RegionStartsMoving, motionDuration = 50.0)
    config.runName shouldBe
      "seed1_algo0_V1.0_tol5.0_beta0.95_dt0.25_train10_length200.0_sourcetrue_waypoint_warm_region_for50.0"
    MobilityConfig(1, 0).motionDuration.isInfinite shouldBe true
  }

  it should "name the evaluation of a source switch while a region moves" in {
    MobilityConfig(1, 0, model = MobilityModel.Waypoint, warmStart = true,
      event = EpisodeEvent.SourceSwitchWhileRegionMoves).runName shouldBe
      "seed1_algo0_V1.0_tol5.0_beta0.95_dt0.25_train10_length200.0_sourcetrue_waypoint_warm_regionmoves"
  }
end MobilityConfigSpec
