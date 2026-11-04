package it.unibo.alchemist.model.actions

import it.unibo.alchemist.model.scafi.nodes.SimpleNodeManager
import it.unibo.testing.{NeighbourRound, Simulations}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import java.io.File
import scala.jdk.CollectionConverters.*

class SendPreviousExportSpec extends AnyFlatSpec with Matchers:

  /** The third round of the second node: the first one wakes up at 1, 2, 3 s, the second one at 1.5, 2.5, 3.5 s. */
  private val observationTime = 3.5

  /** The latest round of its neighbour that each node had received at its third round, by node. */
  private def seenRounds(yaml: String): Seq[Int] =
    val simulation = Simulations.loadFile(new File(s"src/test/resources/yaml/$yaml.yml"))
    val environment = Simulations.runUntil(simulation, observationTime)
    environment.getNodes.getCurrent.asScala.toList
      .sortBy(_.getId)
      .map(node => SimpleNodeManager[Any](node).get[Int](NeighbourRound.SeenRound))

  "SendPreviousExport" should "deliver every export at the next wake-up of its node, as Alchemist 25 did" in {
    // At 3 s the first node has the first round of the second one (computed at 1.5 s, sent at 2.5 s); at 3.5 s the
    // second node has the second round of the first one (computed at 2 s, sent at 3 s).
    seenRounds("previousExportTiming") shouldBe Seq(1, 2)
  }

  "The send program of Alchemist 45" should "deliver every export as soon as its round completes" in {
    // At 3 s the first node has the second round of the second one (2.5 s), at 3.5 s the second node the third round
    // of the first one (3 s).
    seenRounds("immediateSendTiming") shouldBe Seq(2, 3)
  }
