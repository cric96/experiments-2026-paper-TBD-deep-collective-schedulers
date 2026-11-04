package it.unibo.scafi

/** Names of the molecules the aggregate program reads (configuration) and writes (state, exports). */
object Sensors:
  val nextWakeUp = "nextWakeUp"
  val groundTruth = "groundTruth"
  val localComputation = "localComputation"
  val localComputationWindow = "localComputationWindow"
  val ticks = "ticks"
  val reward = "reward"
  val accumulatedReward = "accumulatedReward"
  val accumulatedError = "accumulatedError"
  val window = "window"
  val actions = "actions"
  val weight = "weight"
  val error = "error"
  val source = "source"
  val program = "program"
  val electedLeader = "electedLeader"
  val experiences = "experiences"
  val accumulatedUnstableTime = "accumulatedUnstableTime"
  val speed = "speed"
