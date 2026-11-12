package it.unibo.learning.encoding

import it.unibo.learning.abstractions.AgentState

/** How much of a node's neighbourhood moves, from odometry: the share of the node and of the neighbours of its latest
  * snapshot whose speed is positive, in [0, 1]; zero outside the mobile experiment, where nothing reports a speed.
  */
object NeighbourhoodMotion:

  def share(state: AgentState): Double =
    val neighbours =
      state.latestSnapshot.fold(Seq.empty[Double])(snapshot => state.neighboursIn(snapshot).values.map(_.speed).toSeq)
    val speeds = state.speed +: neighbours
    speeds.count(_ > 0.0).toDouble / speeds.size
