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

/** Actual outcome of a mobile decision, measured throughout its holding interval, including terminal tails. */
final case class MobileOutcome(
    duration: Double,
    errorIntegrals: Map[Double, Double],
    nextState: AgentState,
    terminal: Boolean,
    energy: Double = 1.0
)
