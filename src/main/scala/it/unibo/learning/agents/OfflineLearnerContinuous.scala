package it.unibo.learning.agents

import ai.djl.ndarray.{NDArray, NDManager}
import ai.djl.nn.{Activation, SequentialBlock}
import it.unibo.learning.TradeOff
import it.unibo.learning.abstractions.{AgentState, Trajectory}
import it.unibo.learning.agents.OfflineLearnerContinuous.*
import it.unibo.learning.agents.offline.BatchTensors.{column, rows}
import it.unibo.learning.agents.offline.*
import it.unibo.learning.encoding.{MissingValue, MobileEncoding, MobileFeatures, SpatialTemporal}
import it.unibo.mobility.MobileObjective
import it.unibo.learning.network.{MinMaxActionScaler, TorchNetwork}
import it.unibo.learning.network.NDArrayOps.*

import java.io.File

/** Offline TD3 over the spatial-temporal state, conditioned on the trade-off weight β: twin critics with clipped
  * double-Q targets, delayed policy updates and target policy smoothing ([[TwinDelayed]]); the actor outputs a
  * continuous wake-up period in [`min`, `max`].
  */
final class OfflineLearnerContinuous(
    min: Double,
    max: Double,
    randomKey: Double,
    weight: Double,
    stateDim: Int = SpatialTemporal.FeatureDim,
    hiddenSize: Int = 64,
    gamma: Float = 0.9f,
    lr: Float = 1e-3f,
    batchSize: Int = 64,
    stepsPerEpisode: Int = 160,
    explorationEpisodes: Int = 10,
    finalTrainingMultiplier: Int = 3,
    inspectionDirectory: File = ActorInspection.DefaultDirectory,
    mobile: Boolean = false,
    inspectionEnabled: Boolean = true,
    replayCapacity: Int = MobileObjective.ReplayCapacityPerRegime,
    mobileFeatures: MobileFeatures = MobileFeatures.None,
    periodExploration: PeriodExploration = PeriodExploration.Uniform,
    mobileEncoding: Boolean = true,
    rewardShape: TradeOff.RewardShape = TradeOff.RewardShape(),
    missingValueFeature: Boolean = false
) extends EpisodicOfflineLearner[ContinuousTransition](
      randomKey,
      weight,
      TrainingSchedule(explorationEpisodes, stepsPerEpisode, finalTrainingMultiplier),
      batchSize,
      transitionCapacity = if mobile then replayCapacity else Int.MaxValue
    ):
  private val inputDim: Int =
    if mobile && mobileEncoding then MobileEncoding.featureDim(mobileFeatures)
    else if missingValueFeature then stateDim + 1
    else stateDim
  private val actionScaler: MinMaxActionScaler = MinMaxActionScaler(min, max)
  private val networkSeed: Int = LearnerSeeds.networks(randomKey, weight)
  private val actor: TorchNetwork[SequentialBlock] = actorNetwork("actor", seedOffset = 1)
  private val targetActor: TorchNetwork[SequentialBlock] = actorNetwork("target-actor", seedOffset = 2)
  targetActor.softUpdateFrom(actor, TargetNetworks.HardCopy)
  private val critic1: TorchNetwork[SequentialBlock] = criticNetwork("critic-1", seedOffset = 4)
  private val targetCritic1: TorchNetwork[SequentialBlock] = criticNetwork("target-critic-1", seedOffset = 5)
  targetCritic1.softUpdateFrom(critic1, TargetNetworks.HardCopy)
  private val critic2: TorchNetwork[SequentialBlock] = criticNetwork("critic-2", seedOffset = 6)
  private val targetCritic2: TorchNetwork[SequentialBlock] = criticNetwork("target-critic-2", seedOffset = 7)
  targetCritic2.softUpdateFrom(critic2, TargetNetworks.HardCopy)
  private val policyUpdates: TwinDelayed.DelayedPolicyUpdates = TwinDelayed.DelayedPolicyUpdates()
  @volatile private var fastActor: Array[Float] => Array[Float] = actor.snapshotFastMLP(useTanh = true)

  private def actorNetwork(name: String, seedOffset: Int): TorchNetwork[SequentialBlock] =
    TorchNetwork.createMLP(
      name,
      inputDim,
      Seq(hiddenSize, hiddenSize),
      outputSize = 1,
      Some(Activation.tanhBlock()),
      lr = lr * ActorLearningRateRatio,
      seed = networkSeed + seedOffset
    )

  /** A critic takes the state followed by the scaled action. */
  private def criticNetwork(name: String, seedOffset: Int): TorchNetwork[SequentialBlock] =
    TorchNetwork.createMLP(name, inputDim + 1, Seq(hiddenSize, hiddenSize), 1, lr = lr, seed = networkSeed + seedOffset)

  override def policy: AgentState => Double = policyForWeight(weight)

  override def policyForWeight(customWeight: Double): AgentState => Double = state =>
    if random.nextDouble() < schedule.epsilonAt(currentEpisode) then
      periodExploration.period(min, max, random.nextDouble())
    else actionScaler.unscale(fastActor(encodeState(state, customWeight))(0).toDouble)

  override protected def record(trajectory: Trajectory): Unit =
    Relabelling
      .relabel(trajectory, weight, gamma.toDouble, encodeForOwnWeight, SpatialTemporal.Feature.Beta, mobile,
        rewardShape.scale)
      .foreach(relabelled => transitions.add(toTransition(relabelled), relabelled.isUnstable))

  private def encodeState(state: AgentState, beta: Double): Array[Float] =
    if mobile && mobileEncoding then MobileEncoding.spatial(state, beta, max, mobileFeatures)
    else
      rewardShape.scale.conditioned(
        SpatialTemporal.encode(state, beta, max, missingValueFeature),
        SpatialTemporal.Feature.Beta,
        beta
      )

  private def encodeForOwnWeight(state: AgentState): Array[Float] = encodeState(state, weight)

  /** The intensity that shapes the reward of a step (see [[TradeOff.RewardShape]]). */
  private def rewardIntensity(relabelled: RelabelledStep): Double =
    rewardShape.intensity(relabelled.intensity, !MissingValue.of(relabelled.step.state))

  private def toTransition(relabelled: RelabelledStep): ContinuousTransition =
    val period = relabelled.step.action
    ContinuousTransition(
      state = relabelled.state,
      scaledAction = actionScaler.scale(period),
      reward = relabelled.errorIntegral
        .fold(TradeOff.offlineReward(rewardIntensity(relabelled), period, relabelled.beta, min, max, rewardShape))(
          cost => MobileObjective.reward(cost, relabelled.energyCost, relabelled.beta)
        )
        .toFloat,
      nextState = relabelled.nextState,
      terminal = relabelled.terminalFlag,
      discountFactor = relabelled.discount,
      deferredBeta = relabelled.deferredBeta
    )

  override protected def trainOn(batch: Vector[ContinuousTransition], manager: NDManager): Unit =
    val tensors = ContinuousBatch(batch, manager)
    val targets = clippedDoubleQTargets(tensors)
    val stateActions = tensors.states.concat(tensors.actions, 1)
    Seq(critic1, critic2).foreach { critic =>
      critic.trainStep(TorchNetwork.smoothL1Loss(critic.forward(stateActions, training = true), targets))
    }
    policyUpdates.afterCriticUpdate(updatePolicy(tensors.states))

  /** Trains the actor against the first critic, then moves every target network towards its online network. */
  private def updatePolicy(states: NDArray): Unit =
    critic1.withFrozenParameters(actor.trainStep(actorLoss(states)))
    targetCritic1.softUpdateFrom(critic1, TargetNetworks.UpdateRate)
    targetCritic2.softUpdateFrom(critic2, TargetNetworks.UpdateRate)
    targetActor.softUpdateFrom(actor, TargetNetworks.UpdateRate)

  /** r + γ^Δt · min(Q1_target, Q2_target)(s', π_target(s') + clipped noise). */
  private def clippedDoubleQTargets(tensors: ContinuousBatch): NDArray =
    val nextActions = TwinDelayed.smoothed(targetActor.evaluate(tensors.nextStates), random)
    val nextStateActions = tensors.nextStates.concat(nextActions, 1)
    val nextValues = targetCritic1.evaluate(nextStateActions).minimum(targetCritic2.evaluate(nextStateActions))
    (tensors.rewards + nextValues * tensors.discounts * (-tensors.terminals + 1.0f)).stopGradient()

  /** The actor maximizes the first critic's value of its own actions. */
  private def actorLoss(states: NDArray): NDArray =
    val actions = actor.forward(states, training = true)
    -critic1.forward(states.concat(actions, 1), training = true).mean()

  override protected def onTrained(): Unit =
    fastActor = actor.snapshotFastMLP(useTanh = true)
    if inspectionEnabled && !mobile && schedule.isLastTraining(currentEpisode) && math.abs(
        randomKey - InspectedRun
      ) < RunTolerance
    then
      ActorInspection(actor, critic1, actionScaler, stateDim, inspectionDirectory)
        .save(transitions.oldest(ActorInspection.SamplesPerRegime).map(_.state), weight)
end OfflineLearnerContinuous

object OfflineLearnerContinuous:

  /** The actor learns more slowly than the critics. */
  val ActorLearningRateRatio: Float = 0.5f

  /** Seed of the run whose final policy is inspected and exported. */
  val InspectedRun: Double = 2.0
  val RunTolerance: Double = 1e-6

  final case class ContinuousTransition(
      state: Array[Float],
      scaledAction: Float,
      reward: Float,
      nextState: Array[Float],
      terminal: Float,
      discountFactor: Float,
      deferredBeta: Option[DeferredBeta] = None
  ):
    def conditionedState: Array[Float] = DeferredBeta.conditioned(state, deferredBeta)
    def conditionedNextState: Array[Float] = DeferredBeta.conditioned(nextState, deferredBeta)

  /** A batch of continuous transitions as tensors; the scalar fields are columns. */
  final private case class ContinuousBatch(
      states: NDArray,
      nextStates: NDArray,
      actions: NDArray,
      rewards: NDArray,
      terminals: NDArray,
      discounts: NDArray
  )

  private object ContinuousBatch:
    def apply(batch: Vector[ContinuousTransition], manager: NDManager): ContinuousBatch =
      ContinuousBatch(
        states = rows(batch.map(_.conditionedState), manager),
        nextStates = rows(batch.map(_.conditionedNextState), manager),
        actions = column(batch.map(_.scaledAction), manager),
        rewards = column(batch.map(_.reward), manager),
        terminals = column(batch.map(_.terminal), manager),
        discounts = column(batch.map(_.discountFactor), manager)
      )
end OfflineLearnerContinuous
