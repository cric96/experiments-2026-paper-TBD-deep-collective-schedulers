package it.unibo.learning.agents

import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.types.Shape
import ai.djl.ndarray.{NDArray, NDManager}
import ai.djl.nn.Activation
import it.unibo.learning.TradeOff
import it.unibo.mobility.MobileObjective
import it.unibo.learning.abstractions.{AgentState, CriticContext, Trajectory}
import it.unibo.learning.agents.OfflineLearnerGNN.*
import it.unibo.learning.agents.offline.*
import it.unibo.learning.encoding.{MissingValue, MobileEncoding, MobileFeatures, StarGraph}
import it.unibo.learning.network.{EdgeGATBlock, MinMaxActionScaler, TorchNetwork}
import it.unibo.learning.network.NDArrayOps.*

/** Offline TD3 on the node's star graph: an edge-conditioned graph-attention actor (EdgeGAT) and twin multi-objective
  * critics that predict [Q_convergence, Q_energy], scalarized with the objective weights of each transition's trade-off
  * β, with delayed policy updates and target policy smoothing ([[TwinDelayed]]). A `privilegedCritic` also sees what
  * the mobile experiment measured of each decision on the ground truth ([[CriticContext]], all of it with
  * `richContext`); the actor never does. A `diagnosed` learner diagnoses its networks after every training session
  * ([[TrainingDiagnostics]]); the critics fit the convergence values with the `convergenceLoss`. A measured convergence
  * cost is made hybrid by the `surrogateWeight` ([[HybridConvergence]]), growing up to the `highSurrogateWeight` at the
  * highest β of the scale, if given ([[SurrogateWeights]]).
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
    mobileEncoding: Boolean = true,
    rewardShape: TradeOff.RewardShape = TradeOff.RewardShape(),
    missingValueFeature: Boolean = false,
    privilegedCritic: Boolean = false,
    diagnosed: Boolean = false,
    convergenceLoss: ConvergenceLoss = ConvergenceLoss.Huber,
    richContext: Boolean = false,
    surrogateWeight: Double = 0.0,
    highSurrogateWeight: Option[Double] = None
) extends EpisodicOfflineLearner[GraphTransition](
      randomKey,
      weight,
      TrainingSchedule(explorationEpisodes, stepsPerEpisode, finalTrainingMultiplier),
      batchSize,
      transitionCapacity = if mobile then replayCapacity else Int.MaxValue
    ):
  private val selfDim: Int =
    if mobile && mobileEncoding then MobileEncoding.selfDim(mobileFeatures)
    else if missingValueFeature then StarGraph.SelfDim + 1
    else StarGraph.SelfDim
  private val neighDim: Int =
    if mobile && mobileEncoding then MobileEncoding.neighDim(mobileFeatures) else StarGraph.NeighDim
  require(!privilegedCritic || mobile, "Only the mobile experiment measures the context of a privileged critic")
  require(!richContext || privilegedCritic, "Only a privileged critic sees a rich context")
  private val contextDim: Int =
    if !privilegedCritic then 0 else if richContext then CriticContext.RichSize else CriticContext.Size
  private val actionScaler: MinMaxActionScaler = MinMaxActionScaler(min, max)
  private val hybrid: HybridConvergence =
    HybridConvergence(
      SurrogateWeights.of(surrogateWeight, highSurrogateWeight, rewardShape.scale),
      rewardShape,
      min,
      max
    )
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

  /** A critic sees the scaled action (and the context, if privileged) as extra ego features and predicts one value per
    * objective.
    */
  private def criticNetwork(name: String, seedOffset: Int): TorchNetwork[EdgeGATBlock] =
    TorchNetwork.createEdgeGAT(
      name,
      selfDim + 1 + contextDim,
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
      .relabel(trajectory, weight, gamma.toDouble, encodeForOwnWeight, StarGraph.BetaFeature, mobile, rewardShape.scale,
        encodedBeta)
      .foreach(relabelled => transitions.add(toTransition(relabelled), relabelled.isUnstable))

  private def encodeState(state: AgentState, beta: Double): Array[Float] =
    if mobile && mobileEncoding then MobileEncoding.graph(state, beta, max, mobileFeatures)
    else
      rewardShape.scale.conditioned(
        StarGraph.encode(state, beta, max, missingValueFeature),
        StarGraph.BetaFeature,
        beta
      )

  private def encodeForOwnWeight(state: AgentState): Array[Float] = encodeState(state, weight)

  /** The β feature of [[encodeState]]: the raw β of the mobile encoding, the feature of the β scale otherwise. */
  private def encodedBeta(beta: Double): Float =
    if mobile && mobileEncoding then beta.toFloat else rewardShape.scale.feature(beta)

  private def toTransition(relabelled: RelabelledStep): GraphTransition =
    val period = relabelled.step.action
    val u = TradeOff.normalizedPeriod(period, min, max)
    val intensity = rewardShape.intensity(relabelled.intensity, !MissingValue.of(relabelled.step.state))
    val weights = TradeOff.objectiveWeights(intensity, relabelled.beta, rewardShape)
    GraphTransition(
      state = relabelled.state,
      scaledAction = actionScaler.scale(period),
      convergenceReward =
        relabelled.errorIntegral.fold(-(u * u))(cost => -hybrid.cost(relabelled.step, cost, relabelled.beta)).toFloat,
      energyReward = (if mobile then -relabelled.energyCost else -((1.0 - u) * (1.0 - u))).toFloat,
      convergenceWeight = (if mobile then relabelled.beta else weights.convergence).toFloat,
      energyWeight = (if mobile then (1.0 - relabelled.beta) / relabelled.costScale else weights.energy).toFloat,
      nextState = relabelled.nextState,
      terminal = relabelled.terminalFlag,
      discountFactor = relabelled.discount,
      deferredBeta = relabelled.deferredBeta,
      context = relabelled.context,
      nextContext = relabelled.nextContext
    )

  override protected def trainOn(batch: Vector[GraphTransition], manager: NDManager): Unit =
    val tensors = GraphBatch(batch, manager, contextDim)
    val targets = clippedDoubleQTargets(tensors)
    val stateActions = withAction(tensors.states, tensors.actions, tensors.contexts)
    Seq(critic1, critic2).foreach { critic =>
      critic.trainStep(convergenceLoss.multiObjective(critic.forward(stateActions, training = true), targets))
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
    doubleQTargets(tensors, TwinDelayed.smoothed(targetActor.evaluate(tensors.nextStates), random))

  /** r + γ^Δt · min(Q1_target, Q2_target)(s', `nextActions`), one column per objective. */
  private def doubleQTargets(tensors: GraphBatch, nextActions: NDArray): NDArray =
    val nextStateActions = withAction(tensors.nextStates, nextActions, tensors.nextContexts)
    val nextValues = targetCritic1.evaluate(nextStateActions).minimum(targetCritic2.evaluate(nextStateActions))
    (tensors.rewards + nextValues * tensors.discounts * (-tensors.terminals + 1.0f)).stopGradient()

  /** The actor maximizes w_convergence · Q_convergence + w_energy · Q_energy of its own actions (first critic). */
  private def actorLoss(tensors: GraphBatch): NDArray =
    val actions = actor.forward(tensors.states, training = true)
    val values = critic1.forward(withAction(tensors.states, actions, tensors.contexts), training = true)
    -(values * tensors.objectiveWeights).sum(Array(1)).mean()

  /** Inserts the (B × 1) actions, and the contexts of a privileged critic, right after the ego features of the (B ×
    * TotalDim) star-graph states.
    */
  private def withAction(states: NDArray, actions: NDArray, contexts: Option[NDArray]): NDArray =
    val ego = states.get(new NDIndex(":, :{}", java.lang.Long.valueOf(selfDim.toLong)))
    val neighbours = states.get(new NDIndex(":, {}:", java.lang.Long.valueOf(selfDim.toLong)))
    val egoAndAction = ego.concat(actions, 1)
    contexts.fold(egoAndAction)(egoAndAction.concat(_, 1)).concat(neighbours, 1)

  override protected def onTrained(): Unit =
    fastActor = actor.block.snapshotFastInference()
    fastEmbeddingAndAttention = actor.block.snapshotEmbeddingAndAttention()

  override protected def diagnosticProbe: Option[DiagnosticProbe[GraphTransition]] = Option.when(diagnosed)(Probe)

  /** The networks seen by the diagnostics: the first critic is scalarized with each transition's objective weights. */
  private object Probe extends DiagnosticProbe[GraphTransition]:
    private val betaOfFeature: Map[Float, Double] =
      Relabelling
        .targetWeights(weight, rewardShape.scale.counterfactualWeights)
        .map(beta => encodedBeta(beta) -> beta)
        .toMap

    override val grid: Seq[Double] = TrainingDiagnostics.grid(min, max)

    override def betaOf(transition: GraphTransition): Option[Double] =
      transition.deferredBeta.flatMap(beta => betaOfFeature.get(beta.value))

    override def probe(batch: Vector[GraphTransition], manager: NDManager): ProbedBatch =
      val tensors = GraphBatch(batch, manager, contextDim)
      def values(actions: NDArray): NDArray = critic1.evaluate(withAction(tensors.states, actions, tensors.contexts))
      def scalarized(period: Double): Seq[Double] =
        val actions = manager.full(new Shape(batch.size.toLong, 1), actionScaler.scale(period))
        (values(actions) * tensors.objectiveWeights).sum(Array(1)).toFloatArray.toSeq.map(_.toDouble)
      val targets = doubleQTargets(tensors, targetActor.evaluate(tensors.nextStates))
      ProbedBatch(
        convergenceLoss.multiObjective(values(tensors.actions), targets).getFloat().toDouble,
        actor.evaluate(tensors.states).toFloatArray.toSeq.map(action => actionScaler.unscale(action.toDouble)),
        grid.map(scalarized)
      )
  end Probe
end OfflineLearnerGNN

object OfflineLearnerGNN:

  /** Convergence and energy. */
  val Objectives: Int = 2

  /** The actor learns more slowly than the critics. */
  val ActorLearningRateRatio: Float = 0.5f

  /** What the actor attends to in a state: its readout embedding and one attention weight per neighbour slot. */
  final case class AttentionInspection(embedding: Array[Float], attention: Array[Float], neighbours: List[Int])
