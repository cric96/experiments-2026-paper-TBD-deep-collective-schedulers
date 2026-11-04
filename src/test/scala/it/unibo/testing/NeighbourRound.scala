package it.unibo.testing

import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist.*
import it.unibo.testing.NeighbourRound.*

/** Test program: counts the rounds of the node and records the latest round of a neighbour that it has received. */
class NeighbourRound extends AggregateProgram with StandardSensors with ScafiAlchemistSupport:

  override def main(): Any =
    val round = rep(0)(_ + 1)
    val seen = foldhood(NoRound)(math.max)(mux(nbr(mid()) == mid())(NoRound)(nbr(round)))
    node.put(SeenRound, seen)
    round

object NeighbourRound:

  /** Molecule with the latest round of a neighbour received by the node. */
  val SeenRound: String = "seenRound"

  /** Value of `SeenRound` before any message of a neighbour arrives. */
  val NoRound: Int = -1
