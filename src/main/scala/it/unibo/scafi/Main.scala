package it.unibo.scafi

import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist.*
import it.unibo.learning.TradeOff
import it.unibo.learning.abstractions.AgentState.{NeighborInfo, Snapshot}
import it.unibo.learning.abstractions.{AgentState, Step}
import it.unibo.learning.encoding.InstabilityIntensity
import it.unibo.scafi.Main.*

import scala.collection.immutable.Queue
import scala.jdk.CollectionConverters.*

/** The aggregate program of every node: it runs the active building block, observes its neighbourhood, asks the shared
  * policy when to wake up next, and records what the learners and the exports need.
  */
class Main
    extends AggregateProgram
    with StandardSensors
    with ScafiAlchemistSupport
    with BlockG
    with BlockC
    with Gradients
    with FieldUtils
    with StateManagement:

  private lazy val windowSize: Int = node.getOption[Int](Sensors.window).getOrElse(DefaultWindowSize)
  private lazy val actionSpace: List[Double] =
    node.getOption[List[Double]](Sensors.actions).getOrElse(DefaultActionSpace)
  private lazy val weight: Double = node.getOption[Double](Sensors.weight).getOrElse(DefaultWeight)
  private lazy val targetBlock: BuildingBlock =
    node.getOption[Any](Sensors.program).fold[BuildingBlock](BuildingBlock.Gradient)(BuildingBlock.fromMolecule)
  private lazy val centralAgent: SchedulingCoordinator = alchemistEnvironment.getReactions.getCurrent.asScala
    .collectFirst { case agent: SchedulingCoordinator => agent }
    .getOrElse(throw new IllegalStateException("The simulation must include a scheduling coordinator"))

  override def main(): Any =
    centralAgent.beforeRound(alchemistTimestamp.toDouble)
    val output = blockOutput()
    node.put(Sensors.localComputation, output)
    decide(output)

  private def activeBlock: BuildingBlock = centralAgent.activeBlock(targetBlock)

  /** Each block runs in its own branch, so that switching block at an episode boundary starts from a clean state. */
  private def blockOutput(): Double =
    val isSource = sense[Boolean](Sensors.source)
    val block = activeBlock
    branch(block == BuildingBlock.HopGradient)(hopDistance(isSource)) {
      branch(block == BuildingBlock.Broadcast)(broadcast(isSource)) {
        branch(block == BuildingBlock.Collect)(collect(isSource)) {
          branch(block == BuildingBlock.LeaderElection)(electLeaders(isSource))(classicGradient(isSource))
        }
      }
    }

  private def electLeaders(preferred: Boolean): Double =
    val own = LeaderElection.ownClaim(mid(), preferred)
    val claim = rep(own) { previous =>
      val messages = excludingSelf.reifyField((nbr(previous), nbrRange())).values
      LeaderElection.choose(mid(), preferred, messages)
    }
    val elected = claim.isLeader(mid())
    node.put(Sensors.electedLeader, elected)
    broadcastLeaderId(elected)

  /** G broadcasts the elected ids along a gradient from all current leaders. */
  private def broadcastLeaderId(elected: Boolean): Double =
    val potential = classicGradient(elected)
    rep(mid().toDouble) { value =>
      mux(elected)(mid().toDouble) {
        val candidates = excludingSelf.reifyField((nbr(potential) + nbrRange(), nbr(value))).values
        candidates.minByOption(identity).map(_._2).getOrElse(mid().toDouble)
      }
    }

  private def broadcast(isSource: Boolean): Double =
    val potential = classicGradient(isSource)
    val payload = if isSource then BuildingBlock.broadcastPayloadOf(mid()) else 0.0
    rep(payload) { value =>
      mux(potential == 0.0)(payload) {
        excludingSelf.minHoodSelector[Double, Double](nbr(potential) + nbrRange())(nbr(value)).getOrElse(payload)
      }
    }

  private def collect(isSource: Boolean): Double =
    C[Double, Double](classicGradient(isSource), _ + _, local = 1.0, Null = 0.0)

  /** The node's speed from odometry (m/s), written by the mobile experiment; zero when nothing moves the node. */
  private def ownSpeed: Double = node.getOption[Double](Sensors.speed).getOrElse(0.0)

  /** Chooses the next wake-up period, and logs the transition from the previous decision to the current state. */
  private def decide(output: Double): Unit =
    rep(Option.empty[Decision]) { previous =>
      val state = AgentState(mid(), window(observe(output, previous)), centralAgent.tolerance, ownSpeed)
      val reward = previous.fold(0.0)(decision => rewardOf(decision.state))
      val period = centralAgent.policy(state)
      previous.filter(_ => !centralAgent.mobile).foreach { decision =>
        centralAgent.memory.put(decision.state, indexOf(decision.period), decision.period, reward, state)
      }
      if centralAgent.mobile then centralAgent.decision(state, period, alchemistTimestamp.toDouble)
      else
        recordProgress(output, reward)
        recordExperience(previous, reward)
      node.put(Sensors.localComputationWindow, state.neighborhoodOutput)
      node.put(Sensors.ticks, roundCounter())
      node.put(Sensors.nextWakeUp, period)
      Some(Decision(state, period))
    }

  /** The outputs, distances and last periods of the node and of its neighbours. */
  private def observe(output: Double, previous: Option[Decision]): Snapshot =
    includingSelf.reifyField {
      if centralAgent.mobile then
        NeighborInfo(
          nbr(output),
          nbrRange(),
          nbr(previous.map(_.period)).getOrElse(UndecidedPeriod),
          nbr(alchemistTimestamp.toDouble),
          nbr(ownSpeed)
        )
      else NeighborInfo(nbr(output), nbrRange(), nbr(previous.map(_.period)).getOrElse(UndecidedPeriod))
    }

  /** The last `windowSize` snapshots, oldest first. */
  private def window(snapshot: Snapshot): Queue[Snapshot] =
    rep(Queue.empty[Snapshot])(snapshots => (snapshots :+ snapshot).takeRight(windowSize))

  /** The reward of having slept for the last round in `state`. */
  private def rewardOf(state: AgentState): Double =
    TradeOff.onlineReward(InstabilityIntensity.of(state), elapsedTime, weight, TradeOff.FastestPeriod, actionSpace.max)

  private def recordProgress(output: Double, reward: Double): Unit =
    val block = activeBlock
    val groundTruth = centralAgent.groundTruth(mid(), block)
    node.put(Sensors.accumulatedReward, accumulated(reward))
    node.put(Sensors.accumulatedUnstableTime, accumulated(if hasChanged(output) then elapsedTime else 0.0))
    node.put(Sensors.accumulatedError, accumulated(block.normalizedError(output, groundTruth) * elapsedTime))
    node.put(Sensors.groundTruth, groundTruth)
    node.put(Sensors.error, block.absoluteError(output, groundTruth))
    node.put(Sensors.reward, reward)

  /** The node's trajectory in the current episode, for the offline learners. */
  private def recordExperience(previous: Option[Decision], reward: Double): Unit =
    node.put(
      Sensors.experiences,
      rep(Queue.empty[Step]) { steps =>
        previous.fold(steps)(decision =>
          steps :+ Step(decision.state, indexOf(decision.period), decision.period, reward)
        )
      }
    )

  /** The sum of `value` over the rounds of the episode. */
  private def accumulated(value: Double): Double = rep(0.0)(_ + value)

  /** Whether `output` differs from the one of the previous round. */
  private def hasChanged(output: Double): Boolean =
    rep((false, output)) { case (_, previousOutput) => (output != previousOutput, output) }._1

  /** The index of `period` in the discrete action space, −1 if the period is continuous. */
  private def indexOf(period: Double): Int = actionSpace.indexOf(period)

  /** Time elapsed since the previous round. */
  private def elapsedTime: Double = alchemistDeltaTime(0.0)
end Main

object Main:

  /** Number of rounds observed by the policy, when the `window` molecule is missing. */
  val DefaultWindowSize: Int = 3

  /** Wake-up periods, when the `actions` molecule is missing. */
  val DefaultActionSpace: List[Double] = List(1.0, 1.5, 2.0, 3.0)

  /** Trade-off weight, when the `weight` molecule is missing. */
  val DefaultWeight: Double = 0.9

  /** Period attributed to the neighbours that have not decided yet. */
  val UndecidedPeriod: Double = 1.0

  /** The state a decision was taken in, and the chosen wake-up period. */
  final case class Decision(state: AgentState, period: Double)
