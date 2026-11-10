package it.unibo.learning.agents

import ai.djl.ndarray.{NDArray, NDManager}
import it.unibo.learning.agents.offline.BatchTensors.{column, rows}
import it.unibo.learning.agents.offline.DeferredBeta

/** A transition of the graph learner ([[OfflineLearnerGNN]]): star-graph states, a reward and an objective weight per
  * objective, and the contexts of its critics, if privileged.
  */
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
    deferredBeta: Option[DeferredBeta] = None,
    context: Array[Float] = Array.emptyFloatArray,
    nextContext: Array[Float] = Array.emptyFloatArray
):
  def conditionedState: Array[Float] = DeferredBeta.conditioned(state, deferredBeta)
  def conditionedNextState: Array[Float] = DeferredBeta.conditioned(nextState, deferredBeta)

/** A batch of graph transitions as tensors: rewards and objective weights have one column per objective; the contexts
  * of the critics only when they are privileged.
  */
final private[agents] case class GraphBatch(
    states: NDArray,
    nextStates: NDArray,
    actions: NDArray,
    rewards: NDArray,
    objectiveWeights: NDArray,
    terminals: NDArray,
    discounts: NDArray,
    contexts: Option[NDArray],
    nextContexts: Option[NDArray]
)

private[agents] object GraphBatch:
  def apply(batch: Vector[GraphTransition], manager: NDManager, privileged: Boolean): GraphBatch =
    GraphBatch(
      states = rows(batch.map(_.conditionedState), manager),
      nextStates = rows(batch.map(_.conditionedNextState), manager),
      actions = column(batch.map(_.scaledAction), manager),
      rewards = rows(batch.map(transition => Array(transition.convergenceReward, transition.energyReward)), manager),
      objectiveWeights =
        rows(batch.map(transition => Array(transition.convergenceWeight, transition.energyWeight)), manager),
      terminals = column(batch.map(_.terminal), manager),
      discounts = column(batch.map(_.discountFactor), manager),
      contexts = Option.when(privileged)(rows(batch.map(_.context), manager)),
      nextContexts = Option.when(privileged)(rows(batch.map(_.nextContext), manager))
    )
