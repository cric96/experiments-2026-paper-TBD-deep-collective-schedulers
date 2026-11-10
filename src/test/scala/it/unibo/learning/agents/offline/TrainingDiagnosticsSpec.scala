package it.unibo.learning.agents.offline

import ai.djl.ndarray.NDManager
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.util.Using

class TrainingDiagnosticsSpec extends AnyFlatSpec with Matchers:
  /** A transition relabelled for `beta`, in which the critic prefers the period of index `preferred` of the grid. */
  final private case class Labelled(beta: Double, preferred: Int)

  /** A critic whose value peaks at each transition's preferred period, and an actor that always chooses 2 s. */
  private object Peaked extends DiagnosticProbe[Labelled]:
    override val grid: Seq[Double] = TrainingDiagnostics.grid(0.5, 5.0)
    override def betaOf(transition: Labelled): Option[Double] = Option.when(transition.beta >= 0.0)(transition.beta)
    override def probe(batch: Vector[Labelled], manager: NDManager): ProbedBatch =
      val values =
        grid.indices.map(period => batch.map(transition => -math.abs(period - transition.preferred).toDouble))
      ProbedBatch(criticLoss = 0.5, Seq.fill(batch.size)(2.0), values)

  "The grid of periods" should "span the range of the actions evenly" in {
    val grid = TrainingDiagnostics.grid(0.5, 5.0)
    grid should have size TrainingDiagnostics.GridSize.toLong
    grid.head shouldBe 0.5
    grid.last shouldBe 5.0
    grid.sliding(2).map(pair => pair(1) - pair(0)).foreach(_ shouldBe (0.5 +- 1e-12))
  }

  "The diagnostics" should "average the greedy periods of the critic and the actor's, by regime and trade-off" in {
    val sample = Seq(
      true -> Vector(Labelled(0.8, 0), Labelled(0.8, 2), Labelled(0.2, 9), Labelled(-1.0, 0)),
      false -> Vector(Labelled(0.2, 9))
    )
    val diagnostics = Using.resource(NDManager.newBaseManager())(TrainingDiagnostics.of(3, sample, Peaked, _))
    diagnostics shouldBe Seq(
      TrainingDiagnostic(3, 0.2, unstable = true, 1, 0.5, 2.0, 5.0),
      TrainingDiagnostic(3, 0.8, unstable = true, 2, 0.5, 2.0, 1.0),
      TrainingDiagnostic(3, 0.2, unstable = false, 1, 0.5, 2.0, 5.0)
    )
  }
end TrainingDiagnosticsSpec
