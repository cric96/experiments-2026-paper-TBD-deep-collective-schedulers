package it.unibo.learning.abstractions

/** What only the critics of a privileged learner see of a decision (centralized training, decentralized execution):
  * measured on the ground truth while the experiment trains, never an input of the policy. In units of the loss scale:
  * the node's own loss, the cost it is charged for (e.g. the error of the network around it, per unit of the network's
  * effective size) and the mean loss of the network; a rich context also has the share of the nodes with an unbounded
  * error (the field has not reached them) and the age of the last change of the network, in [0, 1].
  */
final case class CriticContext(
    ownLoss: Double,
    chargedCost: Double,
    networkLoss: Double,
    unreachedShare: Double = 0.0,
    eventAge: Double = 0.0
):
  /** The features of the critics, in the order of the fields: the first [[CriticContext.Size]] of a basic context, all
    * of them ([[CriticContext.RichSize]]) of a rich one.
    */
  def features: Array[Float] =
    Array(ownLoss.toFloat, chargedCost.toFloat, networkLoss.toFloat, unreachedShare.toFloat, eventAge.toFloat)

object CriticContext:
  /** Number of features of a basic context. */
  val Size: Int = 3

  /** Number of features of a rich context. */
  val RichSize: Int = 5

  /** Seconds after which a change of the network counts as old: its age feature is one from then on. */
  val EventAgeHorizon: Double = 50.0

  /** The context of a decision whose outcome was not measured on the ground truth. */
  val Unknown: CriticContext = CriticContext(0.0, 0.0, 0.0)
