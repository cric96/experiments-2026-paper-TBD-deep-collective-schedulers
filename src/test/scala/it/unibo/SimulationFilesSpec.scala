package it.unibo

import it.unibo.testing.Simulations
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import java.io.File
import java.nio.file.Files
import scala.jdk.CollectionConverters.*

/** The Scala snippets embedded in the YAML files are compiled only when a simulation starts: load every file (and every
  * algorithm of the generalization scenario) so that a broken constructor or a renamed class fails the build.
  */
class SimulationFilesSpec extends AnyFlatSpec with Matchers:
  private val simulationFiles =
    new File("src/main/yaml").listFiles().filter(_.getName.endsWith(".yml")).sortBy(_.getName)
  private val generalizationAlgorithms = 0 to 9

  /** Learners (0, 1, 8, 9), fixed periods (3–7, 10), the heuristic (2), Trickle (11) and the gradual controller (12). */
  private val mobilityAlgorithms = 0 to 12

  /** The action that sends the exports with the message timing of Alchemist 25, which every experiment uses. */
  private val PreviousExportSend = "type: SendPreviousExport"

  /** The action that runs the aggregate program of a node. */
  private val ProgramRun = "type: RunScafiProgram"

  /** The send of Alchemist 45, which delivers every export as soon as its round completes. */
  private val ImmediateSend = "program: send"

  /** Marks a YAML comment. */
  private val Comment = "#"

  private def load(file: File, variables: Map[String, Any] = Map.empty): Int =
    Simulations.load(file.getName.stripSuffix(".yml"), variables).getEnvironment.getNodeCount.getCurrent

  private def significantLines(file: File): IndexedSeq[String] =
    Files.readAllLines(file.toPath).asScala.map(_.trim).filterNot(_.startsWith(Comment)).toIndexedSeq

  private def positionsOf(lines: IndexedSeq[String], action: String): IndexedSeq[Int] =
    lines.indices.filter(lines(_).endsWith(action))

  simulationFiles.foreach { file =>
    file.getName should "load with its default variables" in {
      load(file) should be > 0
    }

    it should "send the export of the previous round before every round, as Alchemist 25 did" in {
      val lines = significantLines(file)
      val sends = positionsOf(lines, PreviousExportSend)
      val runs = positionsOf(lines, ProgramRun)
      positionsOf(lines, ImmediateSend) shouldBe empty
      runs should not be empty
      sends.size shouldBe runs.size
      sends.zip(runs).foreach { case (send, run) => send should be < run }
    }
  }

  "simulationGeneralization.yml" should "instantiate every algorithm" in {
    val file = new File("src/main/yaml/simulationGeneralization.yml")
    generalizationAlgorithms.foreach(algorithm => load(file, Map("algoId" -> algorithm)) should be > 0)
  }

  "simulationMobility.yml" should "instantiate every algorithm" in {
    val file = new File("src/main/yaml/simulationMobility.yml")
    mobilityAlgorithms.foreach(algorithm => load(file, Map("algoId" -> algorithm)) should be > 0)
  }
end SimulationFilesSpec
