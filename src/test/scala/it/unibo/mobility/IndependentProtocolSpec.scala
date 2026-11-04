package it.unibo.mobility

import it.unibo.testing.Simulations
import it.unibo.scafi.BuildingBlock
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers
import java.nio.file.Files
import scala.io.Source
import scala.util.Using

class IndependentProtocolSpec extends AnyFlatSpec with Matchers:
  private def evaluationTrace(training: Int, algorithm: Int): Vector[Vector[String]] =
    val directory = Files.createTempDirectory("independent-protocol")
    val simulation = Simulations.load(
      "simulationMobility",
      Map(
        "random" -> 11,
        "evaluationSeed" -> 101,
        "evaluationNodes" -> 50,
        "algoId" -> algorithm,
        "trainingEpisodes" -> training,
        "episodes" -> (training + 1),
        "episodeLength" -> 8.0,
        "stepsPerEpisode" -> 1,
        "speed" -> 0.0,
        "mobilityModel" -> "Waypoint",
        "exportDirectory" -> directory.toString
      )
    )
    simulation.play()
    simulation.run()
    Option(simulation.getError.getOrNull()).foreach(error => throw error)
    val path = directory.toFile
      .listFiles()
      .find(_.getName.endsWith("_positions.csv"))
      .getOrElse(fail("Missing positions"))
    Using.resource(Source.fromFile(path))(
      _.getLines()
        .drop(1)
        .map(_.split(",").toVector)
        .filter(_(0).toInt == training)
        .map(_.slice(1, 6))
        .toVector
    )
  end evaluationTrace

  "Independent evaluation" should "pair deployments and sources across training budgets and schedulers" in {
    val first = evaluationTrace(1, 3)
    val second = evaluationTrace(2, 7)
    first shouldBe second
    first.map(_(1)).distinct.size shouldBe 50
  }

  it should "keep training on the gradient for every target program" in {
    for block <- BuildingBlock.values do
      val config = MobilityConfig(11, 0, evaluationSeed = Some(101), targetProgram = block)
      config.blockAt(0) shouldBe BuildingBlock.Gradient
      config.blockAt(10) shouldBe block
  }

  it should "retain the failed node's totals without accruing error after removal" in {
    val config = MobilityConfig(11, 0, trainingEpisodes = 1, episodes = 2)
    val episode = MobileEpisode(config, 1, 0.0)
    episode.observe(0.0, Map(1 -> NodeObservation(100.0, true, 100.0), 2 -> NodeObservation(0.0, true, 0.0)))
    episode.observe(10.0, Map(1 -> NodeObservation(100.0, true, 100.0), 2 -> NodeObservation(0.0, true, 0.0)))
    episode.remove(1, 10.0)
    episode.observe(20.0, Map(2 -> NodeObservation(0.0, true, 0.0)))
    episode.finish(20.0)
    episode.nodeTotals(1).absoluteError shouldBe 1000.0
    episode.metrics(20.0).last shouldBe 25.0
  }
end IndependentProtocolSpec
