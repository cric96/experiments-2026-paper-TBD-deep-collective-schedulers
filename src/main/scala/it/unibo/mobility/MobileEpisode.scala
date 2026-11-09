package it.unibo.mobility

import it.unibo.learning.abstractions.{AgentState, MobileOutcome, Step, Trajectory}

/** What the experiment observes of a node after a simulator event: its error w.r.t. the ground truth, whether the
  * source can reach it, and the signal that its decisions pay for ([[CostSignal]]).
  */
final case class NodeObservation(error: Double, reachable: Boolean, cost: Double)

/** What a node did since the episode start: its rounds and its integrated absolute error (m s). */
final case class NodeTotals(rounds: Long, absoluteError: Double)

/** Accounting for a single episode; each action owns its measured holding interval, including terminal tails. The
  * metrics integrate the held error at the encoding scale, the decisions pay for the cost signal at the loss scale.
  */
final class MobileEpisode(config: MobilityConfig, val index: Int, val start: Double):
  final private case class Pending(state: AgentState, period: Double, time: Double, costs: Map[Double, Double])
  private val integrals = NodeIntegrals(config.tolerances, config.metricScaleAt(index))
  private val costIntegrals = NodeIntegrals(config.tolerances, config.lossScale)
  private var pending: Map[Int, Pending] = Map.empty
  private var steps: Map[Int, Vector[Step]] = Map.empty
  private var rounds: Long = 0L
  private var nodeRounds: Map[Int, Long] = Map.empty
  private var recoveryStart: Option[Double] = None
  private var recovered: Option[Double] = None

  /** Retain completed contributions, with no accrual after physical removal. */
  def remove(id: Int, now: Double): Unit =
    pending.get(id).foreach(previous => close(id, previous.state, now, terminal = true))
    pending -= id
    integrals.retire(id)
    costIntegrals.retire(id)

  def observe(now: Double, observations: Map[Int, NodeObservation]): Unit =
    observeChanges(now, observations, observations)

  /** Like [[observe]] with `all`, when the nodes not in `changed` were already observed at `now` as in `all`: their
    * integrals would not change, so only the `changed` ones are recorded.
    */
  def observeChanges(now: Double, changed: Map[Int, NodeObservation], all: Map[Int, NodeObservation]): Unit =
    changed.foreach { case (id, observation) =>
      integrals.observe(id, now - start, observation.error, observation.reachable)
      costIntegrals.observe(id, now - start, observation.cost, observation.reachable)
    }
    updateRecovery(now, all)

  private def updateRecovery(now: Double, observations: Map[Int, NodeObservation]): Unit =
    val changeTime = start + config.episodeLength / 2.0
    if config.sourceChanges && now >= changeTime && observations.nonEmpty && recovered.isEmpty then
      recoveryStart.filter(now - _ >= config.recoveryDwell).foreach { began =>
        recovered = Some(began + config.recoveryDwell - changeTime)
      }
      val fraction = observations.values.count(_.error <= config.toleranceAt(index)).toDouble / observations.size
      if fraction >= config.recoveryFraction then recoveryStart = recoveryStart.orElse(Some(now))
      else recoveryStart = None

  def decision(state: AgentState, period: Double, now: Double): Unit =
    close(state.me, state, now, terminal = false)
    val costs = config.tolerances.map(tolerance => tolerance -> costIntegrals(state.me).cost(tolerance)).toMap
    pending += state.me -> Pending(state, period, now, costs)
    rounds += 1
    nodeRounds += state.me -> (nodeRounds.getOrElse(state.me, 0L) + 1)

  def nodeTotals(id: Int): NodeTotals = NodeTotals(nodeRounds.getOrElse(id, 0L), integrals.absoluteOf(id))

  private def close(id: Int, nextState: AgentState, now: Double, terminal: Boolean): Unit =
    pending.get(id).foreach { previous =>
      val costs = config.tolerances
        .map(tolerance => tolerance -> (costIntegrals(id).cost(tolerance) - previous.costs(tolerance)))
        .toMap
      val trainingCosts = costs.collect {
        case (tolerance, cost) if config.trainingTolerances.contains(tolerance) =>
          tolerance -> cost / config.costReferenceTime
      }
      val outcome = MobileOutcome(now - previous.time, trainingCosts, nextState, terminal, MobileObjective.RoundCost)
      val reward = MobileObjective.reward(costs(config.toleranceAt(index)), outcome.energy, config.beta)
      val step = Step(previous.state, -1, previous.period, reward, Some(outcome))
      steps += id -> (steps.getOrElse(id, Vector.empty) :+ step)
    }

  def finish(now: Double): Seq[Trajectory] =
    (integrals.activeValues ++ costIntegrals.activeValues).foreach(_.advance(now - start))
    pending.toSeq.sortBy(_._1).foreach { case (id, previous) => close(id, previous.state, now, terminal = true) }
    pending = Map.empty
    steps.toSeq.sortBy(_._1).map(_._2)

  def metrics(now: Double): Seq[Double] =
    val duration = now - start
    val population = integrals.size.toDouble
    def mean(value: ErrorIntegral => Double): Double =
      if population == 0.0 || duration == 0.0 then 0.0 else integrals.values.map(value).sum / population / duration
    val loss = mean(_.cost(config.toleranceAt(index)))
    val roundRate = if population == 0.0 || duration == 0.0 then 0.0 else rounds / population / duration
    Seq(
      roundRate,
      mean(_.rawIntegral),
      loss,
      mean(_.withinTime(config.toleranceAt(index))),
      mean(_.reachableTime),
      recovered.getOrElse(Double.NaN),
      if !config.sourceChanges then Double.NaN else if recovered.isDefined then 0.0 else 1.0,
      MobileObjective.reward(loss, roundRate, config.beta),
      mean(_.absoluteIntegral)
    )
end MobileEpisode

object MobileEpisode:
  /** The gradient's existing normalization, in metres. */
  val MobileEncodingScale: Double = 500.0

/** One error integral per node, created on the first observation of the node (an unbounded error until then). */
final private class NodeIntegrals(tolerances: Seq[Double], scale: Double):
  private var byNode: Map[Int, ErrorIntegral] = Map.empty

  def observe(id: Int, time: Double, value: Double, reachable: Boolean): Unit =
    val integral = byNode.getOrElse(
      id, {
        val created = ErrorIntegral(tolerances, scale)
        created.observe(0.0, Double.PositiveInfinity, reachable = false)
        byNode += id -> created
        created
      }
    )
    integral.observe(time, value, reachable)

  def apply(id: Int): ErrorIntegral = byNode(id)
  def absoluteOf(id: Int): Double = byNode.get(id).fold(0.0)(_.absoluteIntegral)
  private var retired: Set[Int] = Set.empty
  def retire(id: Int): Unit = retired += id
  def activeValues: Iterable[ErrorIntegral] = byNode.filterNot((id, _) => retired.contains(id)).values
  def values: Iterable[ErrorIntegral] = byNode.values
  def size: Int = byNode.size
