package it.unibo.learning.abstractions

/** What only the critics of a privileged learner see of a decision (centralized training, decentralized execution):
  * measured on the ground truth while the experiment trains, never an input of the policy. In units of the loss scale:
  * the node's own loss, the cost it is charged for (e.g. the error of the network around it, per unit of the network's
  * effective size) and the mean loss of the network.
  */
final case class CriticContext(ownLoss: Double, chargedCost: Double, networkLoss: Double):
  /** The features of the critics, in the order of the fields. */
  def features: Array[Float] = Array(ownLoss.toFloat, chargedCost.toFloat, networkLoss.toFloat)

object CriticContext:
  /** Number of features of a context. */
  val Size: Int = 3

  /** The context of a decision whose outcome was not measured on the ground truth. */
  val Unknown: CriticContext = CriticContext(0.0, 0.0, 0.0)
