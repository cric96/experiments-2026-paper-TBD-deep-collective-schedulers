package it.unibo.alchemist.model.actions

import it.unibo.alchemist.model.incarnations.ScafiIncarnationUtils
import it.unibo.alchemist.model.scafi.actions.RunScafiProgram
import it.unibo.alchemist.model.{Action, Environment, Node, NodeReaction, Position}

import scala.jdk.CollectionConverters.*

/** The message timing of the experiments run on Alchemist 25: when a node wakes up, before its program runs, it sends
  * its neighbours the export of its previous round, so every export is delivered one period after it was computed.
  *
  * Alchemist 45 observes the end of a round, so its `send` program delivers each export as soon as the round completes;
  * Alchemist 25 did not, and its `send` fired only at the next wake-up of the node. In the simulation files this action
  * replaces `- program: send` and precedes `RunScafiProgram` in the actions of the node's reaction.
  */
final class SendPreviousExport[T, P <: Position[P]](environment: Environment[T, P], reaction: NodeReaction[T])
    extends AbstractLocalAction[T](reaction):

  override protected def cloneOnNodeReaction(newReaction: NodeReaction[T]): Action[T] =
    SendPreviousExport(environment, newReaction)

  override def execute(): Unit =
    val sender = getTargetNode
    programsOf(sender).foreach { program =>
      program.getExport(sender.getId).foreach { previous =>
        neighboursOf(sender)
          .flatMap(programsOf)
          .filter(_.programNameMolecule == program.programNameMolecule)
          .foreach(_.sendExport(sender.getId, previous))
      }
    }

  private def programsOf(node: Node[T]): Seq[RunScafiProgram[T, P]] =
    ScafiIncarnationUtils.allScafiProgramsFor[T, P](node).toSeq

  private def neighboursOf(node: Node[T]): Seq[Node[T]] =
    environment.neighborhoodOf(node).getCurrent.getNeighbors.asScala.toSeq
end SendPreviousExport
