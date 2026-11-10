package it.unibo.learning.agents.offline

import ai.djl.ndarray.NDArray
import ai.djl.ndarray.index.NDIndex
import it.unibo.learning.network.NDArrayOps.*
import it.unibo.learning.network.TorchNetwork

/** How the critics of a TD3 learner fit the values of the convergence objective (the energy ones always with the Huber
  * loss): the Huber loss is robust to the rare large costs, but it shrinks them towards the median; the squared loss
  * does not.
  */
enum ConvergenceLoss:
  case Huber
  case Squared

  /** The loss of a critic with one scalarized value per sample: the whole value is a convergence one. */
  def scalar(predicted: NDArray, targets: NDArray): NDArray = this match
    case Huber => TorchNetwork.smoothL1Loss(predicted, targets)
    case Squared => TorchNetwork.squaredLoss(predicted, targets)

  /** The loss of a multi-objective critic, whose first column holds the convergence values and the second the energy
    * ones: the mean of the losses of the two columns.
    */
  def multiObjective(predicted: NDArray, targets: NDArray): NDArray = this match
    case Huber => TorchNetwork.smoothL1Loss(predicted, targets)
    case Squared =>
      val convergence = new NDIndex(":, 0:1")
      val energy = new NDIndex(":, 1:")
      (TorchNetwork.squaredLoss(predicted.get(convergence), targets.get(convergence)) +
        TorchNetwork.smoothL1Loss(predicted.get(energy), targets.get(energy))) / 2
