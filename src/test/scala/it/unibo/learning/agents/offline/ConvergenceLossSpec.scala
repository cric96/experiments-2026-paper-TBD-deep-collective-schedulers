package it.unibo.learning.agents.offline

import ai.djl.ndarray.NDManager
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.util.Using

class ConvergenceLossSpec extends AnyFlatSpec with Matchers:
  private val tolerance = 1e-6f

  /** Two samples of [convergence, energy] values, both errors 3 on the convergence column, 0.5 on the energy one. */
  private def losses(loss: ConvergenceLoss): (Float, Float) = Using.resource(NDManager.newBaseManager()) { manager =>
    val predicted = manager.create(Array(Array(3.0f, 0.5f), Array(-3.0f, 0.5f)))
    val targets = manager.create(Array(Array(0.0f, 0.0f), Array(0.0f, 0.0f)))
    val scalar = loss.scalar(predicted.get(":, 0:1"), targets.get(":, 0:1")).getFloat()
    (scalar, loss.multiObjective(predicted, targets).getFloat())
  }

  "The Huber convergence loss" should "be the smooth L1 loss of every value" in {
    val (scalar, multiObjective) = losses(ConvergenceLoss.Huber)
    scalar shouldBe 2.5f +- tolerance
    multiObjective shouldBe ((2.5f + 0.125f) / 2) +- tolerance
  }

  "The squared convergence loss" should "fit the convergence values quadratically, the energy ones with Huber" in {
    val (scalar, multiObjective) = losses(ConvergenceLoss.Squared)
    scalar shouldBe 4.5f +- tolerance
    multiObjective shouldBe ((4.5f + 0.125f) / 2) +- tolerance
  }
end ConvergenceLossSpec
