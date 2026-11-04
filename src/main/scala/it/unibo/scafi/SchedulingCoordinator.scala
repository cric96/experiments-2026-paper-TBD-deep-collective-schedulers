package it.unibo.scafi

import it.unibo.learning.abstractions.{AgentState, ReplayBuffer}

/** The scheduling service used by an aggregate node; mobile training hooks default to no-ops. */
trait SchedulingCoordinator:
  def policy: AgentState => Double
  def memory: ReplayBuffer
  def activeBlock(target: BuildingBlock): BuildingBlock
  def groundTruth(nodeId: Int, block: BuildingBlock = BuildingBlock.Gradient): Double
  def mobile: Boolean = false
  def tolerance: Double = 0.0
  def beforeRound(time: Double): Unit = ()
  def decision(state: AgentState, period: Double, time: Double): Unit = ()
