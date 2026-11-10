package it.unibo.learning.agents

import ai.djl.ndarray.{NDArray, NDManager}
import it.unibo.learning.agents.offline.BatchTensors.{column, rows}
import it.unibo.learning.agents.offline.DeferredBeta

/** A transition of the continuous learner ([[OfflineLearnerContinuous]]): states, a scalarized reward and the contexts
  * of its critics, if privileged.
  */
final case class ContinuousTransition(
    state: Array[Float],
    scaledAction: Float,
    reward: Float,
    nextState: Array[Float],
    terminal: Float,
    discountFactor: Float,
    deferredBeta: Option[DeferredBeta] = None,
    context: Array[Float] = Array.emptyFloatArray,
    nextContext: Array[Float] = Array.emptyFloatArray
):
  def conditionedState: Array[Float] = DeferredBeta.conditioned(state, deferredBeta)
  def conditionedNextState: Array[Float] = DeferredBeta.conditioned(nextState, deferredBeta)

/** A batch of continuous transitions as tensors; the scalar fields are columns; the contexts of the critics only when
  * they are privileged.
  */
final private[agents] case class ContinuousBatch(
    states: NDArray,
    nextStates: NDArray,
    actions: NDArray,
    rewards: NDArray,
    terminals: NDArray,
    discounts: NDArray,
    contexts: Option[NDArray],
    nextContexts: Option[NDArray]
)

private[agents] object ContinuousBatch:
  /** The tensors of `batch`, with the first `contextSize` features of the contexts if the critics see them. */
  def apply(batch: Vector[ContinuousTransition], manager: NDManager, contextSize: Int): ContinuousBatch =
    ContinuousBatch(
      states = rows(batch.map(_.conditionedState), manager),
      nextStates = rows(batch.map(_.conditionedNextState), manager),
      actions = column(batch.map(_.scaledAction), manager),
      rewards = column(batch.map(_.reward), manager),
      terminals = column(batch.map(_.terminal), manager),
      discounts = column(batch.map(_.discountFactor), manager),
      contexts = Option.when(contextSize > 0)(rows(batch.map(_.context.take(contextSize)), manager)),
      nextContexts = Option.when(contextSize > 0)(rows(batch.map(_.nextContext.take(contextSize)), manager))
    )
