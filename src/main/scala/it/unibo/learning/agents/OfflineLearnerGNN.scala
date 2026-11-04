package it.unibo.learning.agents

import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.{NDArray, NDManager}
import ai.djl.nn.Activation
import it.unibo.learning.TradeOff
import it.unibo.mobility.MobileObjective
import it.unibo.learning.abstractions.{AgentState, Trajectory}
import it.unibo.learning.agents.OfflineLearnerGNN.*
import it.unibo.learning.agents.offline.BatchTensors.{column, rows}
import it.unibo.learning.agents.offline.*
import it.unibo.learning.encoding.{MobileEncoding, MobileFeatures, StarGraph}
import it.unibo.learning.network.{EdgeGATBlock, MinMaxActionScaler, TorchNetwork}
import it.unibo.learning.network.NDArrayOps.*

/** Offline TD3 on the node's star graph: an edge-conditioned graph-attention actor (EdgeGAT) and twin multi-objective
  * critics that predict [Q_convergence, Q_energy], scalarized with the objective weights of each transition's trade-off
  * β, with delayed policy updates and target policy smoothing ([[TwinDelayed]]).
  */
final class OfflineLearnerGNN(
    min: Double,
    max: Double,
    randomKey: Double,
    weight: Double,
    edgeHidden: Int = 24,
    readoutHidden: Int = 48,
    gamma: Float = 0.9f,
    lr: Float = 1e-3f,
    batchSize: Int = 64,
    stepsPerEpisode: Int = 110,
    explorationEpisodes: Int = 10,
    finalTrainingMultiplier: Int = 2,
    mobile: Boolean = false,
    replayCapacity: Int = MobileObjective.ReplayCapacityPerRegime,
    mobileFeatures: MobileFeatures = MobileFeatures.None,
    periodExploration: PeriodExploration = PeriodExploration.Uniform,
    mobileEncoding: Boolean = true
) extends EpisodicOfflineLearner[GraphTransition](
      randomKey,
      weight,
      TrainingSchedule(explorationEpisodes, stepsPerEpisode, finalTrainingMultiplier),
      batchSize,
      transitionCapacity = if mobile then replayCapacity else Int.MaxValue
    ):
  private val selfDim: Int =
    if mobile && mobileEncoding then MobileEncoding.selfDim(mobileFeatures) else StarGraph.SelfDim
  private val neighDim: Int =
    if mobile && mobileEncoding then MobileEncoding.neighDim(mobileFeatures) else StarGraph.NeighDim
  private val actionScaler: MinMaxActionScaler = MinMaxActionScaler(min, max)
  private val networkSeed: Int = LearnerSeeds.networks(randomKey, weight)
  private val actor: TorchNetwork[EdgeGATBlock] = actorNetwork("gat-actor", seedOffset = 101)
  private val targetActor: TorchNetwork[EdgeGATBlock] = actorNetwork("gat-target-actor", seedOffset = 102)
  targetActor.softUpdateFrom(actor, TargetNetworks.HardCopy)
  private val critic1: TorchNetwork[EdgeGATBlock] = criticNetwork("gat-critic-1", seedOffset = 103)
  private val targetCritic1: TorchNetwork[EdgeGATBlock] = criticNetwork("gat-target-critic-1", seedOffset = 104)
  targetCritic1.softUpdateFrom(critic1, TargetNetworks.HardCopy)
  private val critic2: TorchNetwork[EdgeGATBlock] = criticNetwork("gat-critic-2", seedOffset = 105)
  private val targetCritic2: TorchNetwork[EdgeGATBlock] = criticNetwork("gat-target-critic-2", seedOffset = 106)
  targetCritic2.softUpdateFrom(critic2, TargetNetworks.HardCopy)
  private val policyUpdates: TwinDelayed.DelayedPolicyUpdates = TwinDelayed.DelayedPolicyUpdates()
  @volatile private var fastActor: Array[Float] => Float = actor.block.snapshotFastInference()
  @volatile private var fastEmbeddingAndAttention: Array[Float] => (Array[Float], Array[Float]) =
    actor.block.snapshotEmbeddingAndAttention()

  private def actorNetwork(name: String, seedOffset: Int): TorchNetwork[EdgeGATBlock] =
    TorchNetwork.createEdgeGAT(
      name,
      selfDim,
      StarGraph.MaxNeighbors,
      neighDim,
      edgeHidden,
      readoutHidden,
      outputSize = 1,
      outputActivation = Some(Activation.tanhBlock()),
      lr = lr * ActorLearningRateRatio,
      seed = networkSeed + seedOffset
    )

  /** A critic sees the scaled action as an extra ego feature and predicts one value per objective. */
  private def criticNetwork(name: String, seedOffset: Int): TorchNetwork[EdgeGATBlock] =
    TorchNetwork.createEdgeGAT(
      name,
      selfDim + 1,
      StarGraph.MaxNeighbors,
      neighDim,
      edgeHidden,
      readoutHidden,
      outputSize = Objectives,
      lr = lr,
      seed = networkSeed + seedOffset
    )

  /** The actor's readout embedding of `state`, its attention over the neighbours, and the neighbours' ids. */
  def inspectEmbeddingAndAttention(state: AgentState, targetWeight: Double = weight): AttentionInspection =
    val (embedding, attention) = fastEmbeddingAndAttention(encodeState(state, targetWeight))
    val neighbours = state.latestSnapshot.fold(List.empty[Int])(StarGraph.nearestNeighbours(state, _).map(_._1))
    AttentionInspection(embedding, attention, neighbours)

  override def policy: AgentState => Double = policyForWeight(weight)

  override def policyForWeight(customWeight: Double): AgentState => Double = state =>
    if random.nextDouble() < schedule.epsilonAt(currentEpisode) then
      periodExploration.period(min, max, random.nextDouble())
    else actionScaler.unscale(fastActor(encodeState(state, customWeight)).toDouble)

  override protected def record(trajectory: Trajectory): Unit =
    Relabelling
      .relabel(trajectory, weight, gamma.toDouble, encodeForOwnWeight, StarGraph.BetaFeature, mobile)
      .foreach(relabelled => transitions.add(toTransition(relabelled), relabelled.isUnstable))

  private def encodeState(state: AgentState, beta: Double): Array[Float] =
    if mobile && mobileEncoding then MobileEncoding.graph(state, beta, max, mobileFeatures)
    else StarGraph.encode(state, beta, max)

  private def encodeForOwnWeight(state: AgentState): Array[Float] = encodeState(state, weight)

  private def toTransition(relabelled: RelabelledStep): GraphTransition =
    val period = relabelled.step.action
    val u = TradeOff.normalizedPeriod(period, min, max)
    val weights = TradeOff.objectiveWeights(relabelled.intensity, relabelled.beta)
    GraphTransition(
      state = relabelled.state,
      scaledAction = actionScaler.scale(period),
      convergenceReward = relabelled.errorIntegral.fold(-(u * u))(-_).toFloat,
      energyReward = (if mobile then -relabelled.energyCost else -((1.0 - u) * (1.0 - u))).toFloat,
      convergenceWeight = (if mobile then relabelled.beta else weights.convergence).toFloat,
      energyWeight = (if mobile then 1.0 - relabelled.beta else weights.energy).toFloat,
      nextState = relabelled.nextState,
      terminal = relabelled.terminalFlag,
      discountFactor = relabelled.discount,
      deferredBeta = relabelled.deferredBeta
    )

  override protected def trainOn(batch: Vector[GraphTransition], manager: NDManager): Unit =
    val tensors = GraphBatch(batch, manager)
    val targets = clippedDoubleQTargets(tensors)
    val stateActions = withAction(tensors.states, tensors.actions)
    Seq(critic1, critic2).foreach { critic =>
      critic.trainStep(TorchNetwork.smoothL1Loss(critic.forward(stateActions, training = true), targets))
    }
    policyUpdates.afterCriticUpdate(updatePolicy(tensors))

  /** Trains the actor against the first critic, then moves every target network towards its online network. */
  private def updatePolicy(tensors: GraphBatch): Unit =
    critic1.withFrozenParameters(actor.trainStep(actorLoss(tensors)))
    targetCritic1.softUpdateFrom(critic1, TargetNetworks.UpdateRate)
    targetCritic2.softUpdateFrom(critic2, TargetNetworks.UpdateRate)
    targetActor.softUpdateFrom(actor, TargetNetworks.UpdateRate)

  /** r + γ^Δt · min(Q1_target, Q2_target)(s', π_target(s') + clipped noise), one column per objective. */
  private def clippedDoubleQTargets(tensors: GraphBatch): NDArray =
    val nextActions = TwinDelayed.smoothed(targetActor.evaluate(tensors.nextStates), random)
    val nextStateActions = withAction(tensors.nextStates, nextActions)
    val nextValues = targetCritic1.evaluate(nextStateActions).minimum(targetCritic2.evaluate(nextStateActions))
    (tensors.rewards + nextValues * tensors.discounts * (-tensors.terminals + 1.0f)).stopGradient()

  /** The actor maximizes w_convergence · Q_convergence + w_energy · Q_energy of its own actions (first critic). */
  private def actorLoss(tensors: GraphBatch): NDArray =
    val actions = actor.forward(tensors.states, training = true)
    val values = critic1.forward(withAction(tensors.states, actions), training = true)
    -(values * tensors.objectiveWeights).sum(Array(1)).mean()

  /** Inserts the (B × 1) actions right after the ego features of the (B × TotalDim) star-graph states. */
  private def withAction(states: NDArray, actions: NDArray): NDArray =
    val ego = states.get(new NDIndex(":, :{}", java.lang.Long.valueOf(selfDim.toLong)))
    val neighbours = states.get(new NDIndex(":, {}:", java.lang.Long.valueOf(selfDim.toLong)))
    ego.concat(actions, 1).concat(neighbours, 1)

  override protected def onTrained(): Unit =
    fastActor = actor.block.snapshotFastInference()
    fastEmbeddingAndAttention = actor.block.snapshotEmbeddingAndAttention()
end OfflineLearnerGNN

object OfflineLearnerGNN:

  /** Convergence and energy. */
  val Objectives: Int = 2

  /** The actor learns more slowly than the critics. */
  val ActorLearningRateRatio: Float = 0.5f

  /** What the actor attends to in a state: its readout embedding and one attention weight per neighbour slot. */
  final case class AttentionInspection(embedding: Array[Float], attention: Array[Float], neighbours: List[Int])

  final case class GraphTransition(
      state: Array[Float],
      scaledAction: Float,
      convergenceReward: Float,
      energyReward: Float,
      convergenceWeight: Float,
      energyWeight: Float,
      nextState: Array[Float],
      terminal: Float,
      discountFactor: Float,
      deferredBeta: Option[DeferredBeta] = None
  ):
    def conditionedState: Array[Float] = DeferredBeta.conditioned(state, deferredBeta)
    def conditionedNextState: Array[Float] = DeferredBeta.conditioned(nextState, deferredBeta)

  /** A batch of graph transitions as tensors: rewards and objective weights have one column per objective. */
  final private case class GraphBatch(
      states: NDArray,
      nextStates: NDArray,
      actions: NDArray,
      rewards: NDArray,
      objectiveWeights: NDArray,
      terminals: NDArray,
      discounts: NDArray
  )

  private object GraphBatch:
    def apply(batch: Vector[GraphTransition], manager: NDManager): GraphBatch =
      GraphBatch(
        states = rows(batch.map(_.conditionedState), manager),
        nextStates = rows(batch.map(_.conditionedNextState), manager),
        actions = column(batch.map(_.scaledAction), manager),
        rewards = rows(batch.map(transition => Array(transition.convergenceReward, transition.energyReward)), manager),
        objectiveWeights =
          rows(batch.map(transition => Array(transition.convergenceWeight, transition.energyWeight)), manager),
        terminals = column(batch.map(_.terminal), manager),
        discounts = column(batch.map(_.discountFactor), manager)
      )
end OfflineLearnerGNN
