package it.unibo.learning.abstractions

/** One round of a node: the observed state, the index of the chosen action in the discrete action space (−1 when the
  * action is continuous), the action itself (the wake-up period) and the reward that followed.
  */
final case class Step(
    state: AgentState,
    actionIndex: Int,
    action: Double,
    reward: Double,
    mobileOutcome: Option[MobileOutcome] = None
)

/** The steps of one node during one episode, oldest first. */
type Trajectory = Seq[Step]

/** Actual outcome of a mobile decision, measured throughout its holding interval, including terminal tails.
  *
  * @param context
  *   what the privileged critics see of the decision, and `nextContext` of the next one ([[CriticContext]])
  * @param costScale
  *   the factor the costs were divided by, to keep them of order one (e.g. the effective size of the network they are
  *   summed over): dividing the energy by it too keeps the objective
  * @param stratification
  *   how the replay tells the unstable decisions from the stable ones
  */
final case class MobileOutcome(
    duration: Double,
    errorIntegrals: Map[Double, Double],
    nextState: AgentState,
    terminal: Boolean,
    energy: Double = 1.0,
    context: CriticContext = CriticContext.Unknown,
    nextContext: CriticContext = CriticContext.Unknown,
    costScale: Double = 1.0,
    stratification: Stratification = Stratification.ByCost
)

/** How the replay tells the unstable mobile decisions, sampled as often as the stable ones, from the stable ones. */
enum Stratification:
  /** Unstable when the decision paid a cost: the node's own error, or that of its neighbourhood, is rarely nonzero. */
  case ByCost

  /** Unstable when the node itself sees a change or has no value yet: a cost summed over the network is rarely zero. */
  case ByLocalRegime
