package it.unibo.mobility

import it.unibo.testing.Simulations
import it.unibo.alchemist.Molecules.*
import it.unibo.alchemist.boundary.OutputMonitor
import it.unibo.alchemist.core.Simulation
import it.unibo.alchemist.model.{Environment, Position, Reaction, Time}
import it.unibo.scafi.{GroundTruth, SchedulingCoordinator, Sensors, Topology}
import scala.jdk.CollectionConverters.*
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import java.nio.file.Files
import scala.io.Source
import scala.util.Using

class MobileSimulationSpec extends AnyFlatSpec with Matchers:
  private def rows(directory: java.io.File, name: String): Vector[Vector[String]] =
    Using.resource(Source.fromFile(new java.io.File(directory, name)))(
      _.getLines().drop(1).map(_.split(",").toVector).toVector
    )

  private def run(
      algorithm: Int,
      diffusion: Double = 1.0,
      movementInterval: Double = 0.25,
      variant: Map[String, Any] = Map.empty,
      scenario: MobilityConfig => MobilityConfig = identity
  ): (MobilityConfig, Vector[Vector[String]]) =
    val config = scenario(
      MobilityConfig(
        2,
        algorithm,
        diffusion = diffusion,
        movementInterval = movementInterval,
        trainingEpisodes = 1,
        episodes = 2,
        episodeLength = 10.0,
        directory = Files.createTempDirectory("mobile-simulation").toFile
      )
    )
    val simulation = Simulations.load(
      "simulationMobility",
      Map(
        "exportDirectory" -> config.directory.toString,
        "algoId" -> algorithm,
        "diffusion" -> diffusion,
        "movementInterval" -> movementInterval,
        "trainingEpisodes" -> 1,
        "episodes" -> 2,
        "episodeLength" -> 10.0,
        "stepsPerEpisode" -> 1,
        "width" -> 100.0,
        "spacing" -> 50.0,
        "randomness" -> 25.0
      ) ++ variant
    )
    verifyGroundTruth(simulation)
    simulation.play()
    simulation.run()
    Option(simulation.getError.getOrNull()).foreach(error => throw error)
    val episodes = rows(config.directory, s"${config.runName}_episodes.csv")
    episodes.size shouldBe 2
    episodes.map(_(1)) shouldBe Vector("true", "false")
    episodes.foreach { row =>
      row(6).toDouble should be > 0.0
      row(9).toDouble should (be >= 0.0 and be <= 1.0)
    }
    val positions = rows(config.directory, s"${config.runName}_positions.csv")
    positions.foreach { row =>
      row(3).toDouble should (be >= 0.0 and be <= 100.0)
      row(4).toDouble should (be >= 0.0 and be <= 100.0)
    }
    (config, positions)
  end run

  private def verifyGroundTruth[P <: Position[P]](simulation: Simulation[Any, P]): Unit =
    simulation.addOutputMonitor(new OutputMonitor[Any, P]:
      override def stepDone(
          environment: Environment[Any, P],
          reaction: Reaction[Any],
          time: Time,
          step: Long
      ): Unit =
        val nodes = environment.getNodes.getCurrent.asScala.toSeq
        val links = nodes.map { node =>
          val position = environment.getCurrentPosition(node)
          node.getId -> environment.getNeighborhood(node).getCurrent.getNeighbors.asScala.toSeq.map { neighbour =>
            neighbour.getId -> position.distanceTo(environment.getCurrentPosition(neighbour))
          }
        }.toMap
        val sources = nodes.filter(_.valueOf[Boolean](Sensors.source).getOrElse(false)).map(_.getId).toSet
        val reference = GroundTruth.shortestPaths(Topology(nodes.map(_.getId), links), sources)
        val coordinator = environment.getReactions.getCurrent.asScala
          .collectFirst { case scheduler: SchedulingCoordinator =>
            scheduler
          }
          .getOrElse(fail("No scheduling coordinator"))
        nodes.foreach { node =>
          coordinator.groundTruth(node.getId) shouldBe reference.getOrElse(node.getId, Double.PositiveInfinity)
        })

  /** Episode, time, node, position and source of every frame: what does not depend on the schedule. */
  private def traces(positions: Vector[Vector[String]]): Vector[Vector[String]] = positions.map(_.take(6))

  "The mobile scenario" should "train and evaluate both neural architectures through episode boundaries" in {
    val (_, graph) = run(0)
    val (_, mlp) = run(1)
    traces(graph) shouldBe traces(mlp)
  }

  it should "use identical movement and source traces for different round schedules" in {
    val (_, first) = run(3)
    val (_, second) = run(7)
    traces(first) shouldBe traces(second)
    first.filter(_(0) == "1").map(row => (row(3), row(4))).distinct.size should be > 4
  }

  it should "evaluate the static-trained transfer controls" in {
    run(8)
    run(9)
  }

  it should "finish the final episode with a decimal movement interval" in
    run(3, movementInterval = 0.1)

  it should "keep nodes stationary at zero diffusion" in {
    val (_, positions) = run(3, diffusion = 0.0)
    positions.filter(_(0) == "1").groupBy(_(2)).values.foreach { node =>
      node.map(row => (row(3), row(4))).distinct.size shouldBe 1
    }
  }

  private def deploymentOf(positions: Vector[Vector[String]], episode: String): Map[String, (String, String)] =
    positions.filter(_(0) == episode).map(row => row(2) -> (row(3), row(4))).toMap

  it should "redeploy the nodes at every episode, unless it starts warm" in {
    val (_, cold) = run(3, diffusion = 0.0)
    deploymentOf(cold, "0") should not be deploymentOf(cold, "1")
    val (_, warm) = run(3, diffusion = 0.0, variant = Map("warmStart" -> 1), scenario = _.copy(warmStart = true))
    deploymentOf(warm, "0") shouldBe deploymentOf(warm, "1")
  }

  it should "train on the staleness cost with a complete replay" in {
    val variant = Map("completeReplay" -> 1, "costSignal" -> "Staleness", "lossScale" -> 50.0, "retention" -> 6.0)
    run(0, variant = variant)
    run(1, variant = variant)
  }

  it should "train the original encoding with the full-range beta on the error of the neighbourhood" in {
    val variant = Map(
      "mobileEncoding" -> 0,
      "missingValue" -> 1,
      "betaScale" -> "FullRange",
      "costSignal" -> "Neighbourhood",
      "costReferenceTime" -> 0.25,
      "trainingTolerances" -> "0.0",
      "completeReplay" -> 1
    )
    run(0, variant = variant)
    run(1, variant = variant)
  }

  it should "run Trickle and the gradual controller within the configured periods" in {
    val period = 1.0
    val tolerance = 0.15
    Seq(11, 12).foreach { algorithm =>
      val (config, _) = run(algorithm, variant = Map("baselineMinPeriod" -> period, "baselineMaxPeriod" -> period))
      rows(config.directory, s"${config.runName}_episodes.csv").foreach { episode =>
        episode(6).toDouble shouldBe (1.0 / period) +- tolerance
      }
    }
  }

  it should "move the nodes between waypoints" in {
    val waypoint = Map("mobilityModel" -> "Waypoint", "speed" -> 2.0)
    val (_, positions) = run(3, variant = waypoint, scenario = _.copy(model = MobilityModel.Waypoint, speed = 2.0))
    val tracks = positions.filter(_(0) == "1").groupBy(_(2)).values.map(_.map(row => (row(3), row(4))).distinct.size)
    tracks.count(_ > 1) should be > 0
  }

  it should "keep the network still until halfway, then move only a region and keep the source" in {
    val events = Map("episodeEvent" -> "RegionStartsMoving", "trainingEvents" -> "RegionStartsMoving")
    val (config, positions) = run(
      3,
      variant = events,
      scenario = _.copy(event = EpisodeEvent.RegionStartsMoving, trainingEvents = Seq(EpisodeEvent.RegionStartsMoving))
    )
    val evaluation = positions.filter(_(0) == "1")
    val halfway = config.episodeLength / 2.0
    def frames(before: Boolean) = evaluation.filter(row => row(1).toDouble < halfway == before)
    def moved(rows: Vector[Vector[String]]): Set[String] =
      rows
        .groupBy(_(2))
        .collect { case (node, track) if track.map(row => (row(3), row(4))).distinct.size > 1 => node }
        .toSet
    val nodes = evaluation.map(_(2)).distinct.size
    moved(frames(before = true)) shouldBe empty
    moved(evaluation).size shouldBe math.ceil(config.movingShare * nodes).toInt
    evaluation.filter(_(5) == "true").map(_(2)).distinct.size shouldBe 1
    frames(before = true).filter(_(8) == "true") shouldBe empty
    frames(before = false).filter(_(8) == "true").map(_(2)).toSet shouldBe moved(evaluation)
    evaluation.groupBy(_(2)).values.foreach { track =>
      track.map(_(6).toLong) shouldBe sorted
      track.map(_(7).toDouble) shouldBe sorted
    }
  }

  it should "stop the region after it has moved for a while" in {
    val events =
      Map("episodeEvent" -> "RegionStartsMoving", "trainingEvents" -> "RegionStartsMoving", "motionDuration" -> 1.0)
    val event = EpisodeEvent.RegionStartsMoving
    val (config, positions) = run(
      3,
      variant = events,
      scenario = _.copy(event = event, trainingEvents = Seq(event), motionDuration = 1.0)
    )
    val evaluation = positions.filter(_(0) == "1")
    def at(time: Double): Map[String, (String, String)] =
      evaluation.filter(_(1).toDouble == time).map(row => row(2) -> (row(3), row(4))).toMap
    val halfway = config.episodeLength / 2.0
    val before = at(halfway - 1.0)
    val stopped = at(halfway + 1.0)
    val moved = before.keySet.filter(node => before(node) != stopped(node))
    moved.size shouldBe math.ceil(config.movingShare * before.size).toInt
    at(halfway + 3.0) shouldBe stopped
    evaluation.filter(row => row(1).toDouble > halfway && row(8) == "true") shouldBe empty
  }

  it should "move only a region from the start, and switch the source halfway" in {
    val events =
      Map("episodeEvent" -> "SourceSwitchWhileRegionMoves", "trainingEvents" -> "SourceSwitchWhileRegionMoves")
    val event = EpisodeEvent.SourceSwitchWhileRegionMoves
    val (config, positions) = run(3, variant = events, scenario = _.copy(event = event, trainingEvents = Seq(event)))
    val evaluation = positions.filter(_(0) == "1")
    val halfway = config.episodeLength / 2.0
    val firstHalf = evaluation.filter(_(1).toDouble < halfway)
    val moved = firstHalf
      .groupBy(_(2))
      .collect { case (node, track) if track.map(row => (row(3), row(4))).distinct.size > 1 => node }
      .toSet
    moved.size shouldBe math.ceil(config.movingShare * evaluation.map(_(2)).distinct.size).toInt
    evaluation.filter(_(8) == "true").map(_(2)).toSet shouldBe moved
    evaluation.groupBy(_(1)).values.foreach(_.filter(_(8) == "true").map(_(2)).toSet shouldBe moved)
    def sources(before: Boolean) =
      evaluation.filter(row => row(1).toDouble < halfway == before && row(5) == "true").map(_(2)).toSet
    sources(before = true).size shouldBe 1
    sources(before = false) should not be sources(before = true)
  }

  it should "train with the optional local features in the warm waypoint scenario" in {
    val variant = Map(
      "completeReplay" -> 1,
      "costSignal" -> "Staleness",
      "lossScale" -> 50.0,
      "retention" -> 6.0,
      "mobilityModel" -> "Waypoint",
      "speed" -> 2.0,
      "warmStart" -> 1,
      "ownSpeed" -> 1,
      "unexplainedChange" -> 1,
      "dependants" -> 1
    )
    val scenario =
      (config: MobilityConfig) => config.copy(model = MobilityModel.Waypoint, speed = 2.0, warmStart = true)
    run(0, variant = variant, scenario = scenario)
    run(1, variant = variant, scenario = scenario)
  }
  it should "train the encoding and replay ablations through the YAML interface" in {
    val variant = Map("mobileEncoding" -> 0, "balanceScenarios" -> 1, "periodExploration" -> "LogUniform")
    run(0, variant = variant)
    run(1, variant = variant)
  }
end MobileSimulationSpec
