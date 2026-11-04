package it.unibo.testing

import it.unibo.learning.abstractions.AgentState
import it.unibo.learning.abstractions.AgentState.NeighborInfo

import scala.collection.immutable.Queue

/** Builders of [[AgentState]] fixtures. */
object AgentStates:

  /** What a node knows about one device (itself included) in one snapshot. */
  final case class Reading(output: Double, distance: Double = 0.0, action: Double = 1.0)

  /** A state of node `me` from its oldest to its newest neighbourhood snapshot. */
  def of(me: Int, snapshots: Map[Int, Reading]*): AgentState =
    val history = Queue.from(snapshots.map(_.view.mapValues(r => NeighborInfo(r.output, r.distance, r.action)).toMap))
    AgentState(me, history)

  /** Node 0 on a line 0 - 1 - 2 (10 m apart) of a gradient from node 2, whose source moves to node 0. */
  val sourceMovingTowardsMe: AgentState = of(
    me = 0,
    Map(0 -> Reading(20.0), 1 -> Reading(10.0, distance = 10.0, action = 2.0)),
    Map(0 -> Reading(20.0), 1 -> Reading(10.0, distance = 10.0, action = 2.0)),
    Map(0 -> Reading(0.0), 1 -> Reading(10.0, distance = 10.0, action = 3.0))
  )

  /** Node 1 in the middle of a line whose neighbour 2 just lost its value (became unreachable). */
  val neighbourBecameUnreachable: AgentState = of(
    me = 1,
    Map(0 -> Reading(0.0, distance = 10.0), 1 -> Reading(10.0), 2 -> Reading(20.0, distance = 10.0)),
    Map(0 -> Reading(0.0, distance = 10.0), 1 -> Reading(10.0), 2 -> Reading(Double.PositiveInfinity, 10.0, 4.0))
  )

  /** Node 1 with a stable neighbourhood, which routes neighbour 2 (a child in the shortest-path tree). */
  val stableBackbone: AgentState = of(
    me = 1,
    Map(0 -> Reading(0.0, distance = 10.0), 1 -> Reading(10.0), 2 -> Reading(20.0, distance = 10.0)),
    Map(0 -> Reading(0.0, distance = 10.0), 1 -> Reading(10.0), 2 -> Reading(20.0, distance = 10.0))
  )

  /** A node that has seen only itself, once. */
  val isolated: AgentState = of(me = 7, Map(7 -> Reading(Double.PositiveInfinity)))

  /** Node 0 without neighbours, which computed `outputs` (oldest first) and had slept `lastPeriod` before its last
    * round.
    */
  def alone(lastPeriod: Double, outputs: Double*): AgentState =
    of(me = 0, outputs.map(output => Map(0 -> Reading(output, action = lastPeriod)))*)

  val fixtures: Seq[AgentState] = Seq(sourceMovingTowardsMe, neighbourBecameUnreachable, stableBackbone, isolated)
end AgentStates
