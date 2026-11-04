package it.unibo.learning.agents

import it.unibo.learning.abstractions.ReplayBuffer.Experience
import it.unibo.learning.abstractions.{AgentState, DecayReference, Trajectory}

import scala.util.Random

/** A scheduling policy shared by all the nodes, possibly learning: it maps the state of a node to its next wake-up
  * period. Only `policy` is mandatory; the other hooks default to doing nothing.
  */
trait Learner:

  /** The policy for the trade-off weight the learner was configured with. */
  def policy: AgentState => Double

  /** The policy for another trade-off weight β; learners that cannot be conditioned on β ignore it. */
  def policyForWeight(customWeight: Double): AgentState => Double = policy

  /** Online update with a batch sampled from the replay buffer shared by the nodes. */
  def update(batch: Seq[Experience]): Unit = ()

  /** Called at the end of every episode with the trajectory of each node. */
  def endEpisode(trajectories: Seq[Trajectory]): Unit = ()

  /** Training-only replay stratum supplied by the experiment, never an actor input. */
  def trainingScenario(scenario: Int): Unit = ()

  /** Lets the simulation provide its seeded source of randomness. */
  def injectRandom(random: Random): Unit = ()

  /** Hyper-parameters, by name, that the simulation decays at the end of every episode. */
  def decayingParameters: Seq[(String, DecayReference[Double])] = Seq.empty
