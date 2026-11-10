package it.unibo.alchemist.model.reactions

import it.unibo.alchemist.Molecules.*
import it.unibo.alchemist.model.reactions.MobileExperiment.*
import it.unibo.alchemist.boundary.OutputMonitor
import it.unibo.alchemist.model.*
import it.unibo.alchemist.model.scafi.nodes.SimpleNodeManager
import it.unibo.learning.{Box, LearningInfo}
import it.unibo.learning.abstractions.{AgentState, ReplayBuffer}
import it.unibo.learning.agents.Learner
import it.unibo.learning.agents.offline.TrainingDiagnostic
import it.unibo.mobility.{BrownianMotion, EpisodeEvent, MobileEpisode, MobilityConfig, MovingRegion, NodeMotion}
import it.unibo.scafi.{BuildingBlock, SchedulingCoordinator, Sensors}
import it.unibo.util.Csv
import org.apache.commons.math3.random.{RandomAdaptor, RandomGenerator, Well19937c}

import java.io.File
import scala.util.Random

/** Mobile experiment clock and event observer. Its independent streams originate from Alchemist's injected RNG. */
final class MobileExperiment[T, P <: Position[P]](
    environment: Environment[T, P],
    distribution: TimeDistribution[T],
    random: RandomGenerator,
    learner: Learner,
    box: Box,
    config: MobilityConfig
) extends AbstractGlobalReaction[T, P](environment, distribution)
    with SchedulingCoordinator:
  // Draw a fixed number of stream roots, irrespective of policy decisions or network connectivity.
  private val motionSeed: Long = random.nextLong()
  private val sourceSeed: Long = random.nextLong()
  private val deploymentSeed: Int = random.nextInt()
  private val learningRandom = new Random(RandomAdaptor.createAdaptor(new Well19937c(random.nextLong())))
  private val redeployment = Redeployment(environment, box, deploymentSeed)
  private var episode = newEpisode(0, 0.0)
  private var motionRandom: RandomGenerator = new Well19937c(motionSeed)
  private var motion = NodeMotion.forEpisode(config, 0, motionRandom, box.width)
  private var sources = new Well19937c(sourceSeed)
  private val network = NetworkView(environment, () => orderedNodes, () => config.blockAt(episode.index))
  private val observer = NetworkObserver(network, config)
  private var halfwayPassed: Boolean = false
  private var extraSwitches: Seq[Double] = config.extraSwitchesAt(0)
  private var moving: Option[Set[Int]] = None
  private var complete: Boolean = false
  private var episodeRows: Vector[Seq[String]] = Vector.empty
  private var fullRows: Vector[Seq[String]] = Vector.empty
  private var positionRows: Vector[Seq[String]] = Vector.empty
  private var ordinalById: Map[Int, Int] = Map.empty
  // The spatial and attention snapshots of the static experiments, in a folder of their own for every run.
  private val snapshots = FinalEpisodeSnapshots(
    environment,
    learner,
    LearningInfo(config.seed, 0, 0, config.episodeLength.toInt),
    id => network.groundTruths.getOrElse(id, Unknown),
    new File(config.directory, s"$SnapshotFolder/${config.runName}"),
    episodeOverride = Some(config.episodes - 1),
    inspectEverySeed = config.evaluationSeed.contains(InspectedEvaluationSeed),
    observationTolerance = () => tolerance
  )

  override val memory: ReplayBuffer = ReplayBuffer(1, learningRandom)
  learner.injectRandom(learningRandom)
  override def mobile: Boolean = true
  override def tolerance: Double = config.toleranceAt(episode.index)
  override def policy: AgentState => Double = learner.policy
  override def activeBlock(target: BuildingBlock): BuildingBlock = config.blockAt(episode.index)
  override def groundTruth(nodeId: Int, block: BuildingBlock): Double =
    network.groundTruths.getOrElse(nodeId, Double.PositiveInfinity)

  override protected def onInitializationComplete(time: Time, environment: Environment[T, ?]): Unit =
    redeployment.capturePrototypes()
    confineNodes()
    moving = movingAtStart(0)
    if config.evaluationSeed.nonEmpty then resetIndependent(0)
    chooseSource()
    observe(0.0)
    this.environment.getSimulation.addOutputMonitor(new OutputMonitor[T, P]:
      override def stepDone(environment: Environment[T, P], reaction: Reaction[T], time: Time, step: Long): Unit =
        if !complete then observe(time.toDouble))

  override def beforeRound(time: Double): Unit = if !complete then observe(time)

  /** Records the decision with its context, observed after the round that changed the node's output. */
  override def decision(state: AgentState, period: Double, time: Double): Unit =
    if !complete then
      observe(time)
      episode.decision(
        state,
        period,
        time,
        observer.contextOf(state.me, config.sinceLastChange(episode.index, time - episode.start))
      )

  override protected def react(): Unit =
    val now = environment.getSimulation.getTime.toDouble
    if !complete then
      observe(now)
      if now + ClockTolerance >= episode.start + config.episodeLength then finishEpisode(now)
      if !complete then
        if now > episode.start then moveNodes()
        if !halfwayPassed && now + ClockTolerance >= episode.start + config.episodeLength / 2.0 then
          halfwayEvent()
          halfwayPassed = true
        if halfwayPassed then followMotionWindow(now)
        switchAgain(now)
        observe(now)
        if math.abs((now - episode.start) % FrameInterval) < ClockTolerance then recordFrame(now)
        if math.abs(now - math.rint(now)) < ClockTolerance then snapshots.observe(now)

  private def finishEpisode(now: Double): Unit =
    val trajectories = episode.finish(now)
    episodeRows :+= metadata(now) ++ episode.metrics(now).map(_.toString)
    if config.balanceScenarios then learner.trainingScenario(config.eventAt(episode.index).ordinal)
    learner.endEpisode(trajectories)
    complete = episode.index + 1 >= config.episodes
    flush()
    if !complete then
      val next = episode.index + 1
      if config.evaluationSeed.isEmpty && !config.warmStart then
        redeployment.redeploy(redeployment.positionsFor(next))
        confineNodes()
      episode = newEpisode(next, now)
      observer.forget()
      motionRandom = new Well19937c(motionSeed + next)
      motion = NodeMotion.forEpisode(config, next, motionRandom, box.width)
      sources = new Well19937c(sourceSeed + next)
      if config.evaluationSeed.nonEmpty then resetIndependent(next)
      halfwayPassed = false
      extraSwitches = config.extraSwitchesAt(next)
      moving = movingAtStart(next)
      chooseSource()
      observe(now)

  private def newEpisode(index: Int, start: Double): MobileEpisode =
    MobileEpisode(config, index, start, () => observer.costScale)

  private def activeWidth: Double =
    if config.isTraining(episode.index) || config.evaluationSeed.isEmpty then box.width
    else box.width * math.sqrt(config.evaluationNodes.toDouble / NominalPopulation)

  private def resetIndependent(index: Int): Unit =
    val training = config.isTraining(index)
    val seed = if training then config.seed else config.evaluationSeed.getOrElse(config.seed)
    val ordinal = if training then index else index - config.trainingEpisodes
    val root = seed.toLong * IndependentSeedStride + ordinal
    val deployment = new Well19937c(root)
    val count = if training then NominalPopulation else config.evaluationNodes
    val columns = math.ceil(math.sqrt(count.toDouble)).toInt
    val rows = math.ceil(count.toDouble / columns).toInt
    val positions = (0 until count).map { node =>
      val x = (node % columns + 0.5) * activeWidth / columns
      val y = (node / columns + 0.5) * activeWidth / rows
      environment.makePosition(
        Double.box(BrownianMotion.reflect(x + (deployment.nextDouble() - 0.5) * box.randomness, activeWidth)),
        Double.box(BrownianMotion.reflect(y + (deployment.nextDouble() - 0.5) * box.randomness, activeWidth))
      )
    }.toList
    if index == 0 then orderedNodes.zip(positions).foreach((node, position) => environment.moveNodeTo(node, position))
    else redeployment.redeploy(positions)
    orderedNodes.foreach(node => SimpleNodeManager[T](node).put(Sensors.program, config.blockAt(index).code))
    network.topologyChanged()
    motionRandom = new Well19937c(root + MotionStreamOffset)
    motion = NodeMotion.forEpisode(config, index, motionRandom, activeWidth)
    sources = new Well19937c(root + SourceStreamOffset)
  end resetIndependent

  private def removeLeader(): Unit =
    val victim = orderedNodes
      .filter(_.valueOf[Boolean](Sensors.electedLeader).getOrElse(false))
      .minByOption(node => (if NetworkView.isSource(node) then 0 else 1, node.getId))
      .getOrElse(throw new IllegalStateException("No elected leader at the failure event"))
    episode.remove(victim.getId, environment.getSimulation.getTime.toDouble)
    environment.removeNode(victim)
    network.topologyChanged()

  private def orderedNodes: List[Node[T]] = agents.sortBy(_.getId)

  /** The legacy perturbed grid can start outside the square, even in the zero-diffusion control. */
  private def confineNodes(): Unit =
    orderedNodes.foreach { node =>
      val position = environment.getCurrentPosition(node)
      val x = BrownianMotion.reflect(position.get(0), activeWidth)
      val y = BrownianMotion.reflect(position.get(1), activeWidth)
      environment.moveNodeTo(node, environment.makePosition(Double.box(x), Double.box(y)))
    }
    network.topologyChanged()

  /** Who moves when an episode starts: every node (None), a region, or nobody until a halfway event moves them. */
  private def movingAtStart(index: Int): Option[Set[Int]] = config.eventAt(index) match
    case EpisodeEvent.SourceSwitch => None
    case EpisodeEvent.RegionStartsMoving | EpisodeEvent.SourceSwitchThenMotion => Some(Set.empty)
    case EpisodeEvent.SourceSwitchWhileRegionMoves => Some(movingRegion())

  private def halfwayEvent(): Unit = config.eventAt(episode.index) match
    case EpisodeEvent.SourceSwitch | EpisodeEvent.SourceSwitchWhileRegionMoves | EpisodeEvent.SourceSwitchThenMotion =>
      if config.sourceChanges then
        if config.blockAt(episode.index) == BuildingBlock.LeaderElection then removeLeader()
        else chooseSource()
    case EpisodeEvent.RegionStartsMoving => moving = Some(movingRegion())

  /** In the motion window of the episode ([[MobilityConfig.motionWindowAt]]) every node of a composite episode moves;
    * after it, no node moves.
    */
  private def followMotionWindow(now: Double): Unit = config.motionWindowAt(episode.index).foreach { window =>
    if now + ClockTolerance >= episode.start + window.end then moving = Some(Set.empty)
    else if config.eventAt(episode.index) == EpisodeEvent.SourceSwitchThenMotion &&
      now + ClockTolerance >= episode.start + window.start
    then moving = None
  }

  /** Moves the source to a far node at each extra switch of the episode that `now` has reached. */
  private def switchAgain(now: Double): Unit =
    val (due, later) = extraSwitches.partition(time => now + ClockTolerance >= episode.start + time)
    due.foreach(_ => chooseSource())
    extraSwitches = later

  /** The nodes nearest a centre drawn from the motion stream of the episode. */
  private def movingRegion(): Set[Int] =
    val positions = orderedNodes.map { node =>
      val position = environment.getCurrentPosition(node)
      node.getId -> (position.get(0), position.get(1))
    }.toMap
    MovingRegion.select(positions, MovingRegion.centre(motionRandom, activeWidth), config.movingShare)

  private def moveNodes(): Unit =
    val speeds = orderedNodes.map { node =>
      val speed = if moving.forall(_.contains(node.getId)) then moveNode(node) else 0.0
      SimpleNodeManager[T](node).put(Sensors.speed, speed)
      speed
    }
    if speeds.exists(_ > 0.0) then network.topologyChanged()

  /** Moves `node` by one movement interval and returns its speed. */
  private def moveNode(node: Node[T]): Double =
    val position = environment.getCurrentPosition(node)
    val (x, y) = motion.move(node.getId, position.get(0), position.get(1))
    environment.moveNodeTo(node, environment.makePosition(Double.box(x), Double.box(y)))
    math.hypot(x - position.get(0), y - position.get(1)) / config.movementInterval

  private def chooseSource(): Unit =
    val nodes = orderedNodes
    ordinalById = nodes.zipWithIndex.map((node, ordinal) => node.getId -> ordinal).toMap
    val current = nodes.find(_.valueOf[Boolean](Sensors.source).getOrElse(false))
    val far = current.toSeq.flatMap { origin =>
      nodes.filter(node =>
        environment.getCurrentPosition(origin).distanceTo(environment.getCurrentPosition(node)) > MinimumSourceDistance
      )
    }
    val candidates = if far.nonEmpty then far else nodes.filterNot(node => current.contains(node))
    val selected = candidates(sources.nextInt(candidates.size))
    nodes.foreach(node => SimpleNodeManager[T](node).put(Sensors.source, node == selected))
    network.sourcesChanged()

  private def observe(now: Double): Unit = observer.observe(now, episode, orderedNodes)

  private def metadata(now: Double): Seq[String] =
    Seq(
      episode.index.toString,
      config.isTraining(episode.index).toString,
      now.toString,
      config.levelAt(episode.index).toString,
      tolerance.toString,
      config.beta.toString
    )

  private def recordFrame(now: Double): Unit =
    fullRows :+= metadata(now) ++ episode.metrics(now).map(_.toString)
    orderedNodes.foreach { node =>
      val ordinal = ordinalById(node.getId)
      val position = environment.getCurrentPosition(node)
      val totals = episode.nodeTotals(node.getId)
      positionRows :+= Seq(
        episode.index.toString,
        (now - episode.start).toString,
        ordinal.toString,
        position.get(0).toString,
        position.get(1).toString,
        node.valueOf[Boolean](Sensors.source).getOrElse(false).toString,
        totals.rounds.toString,
        totals.absoluteError.toString,
        moving.exists(_.contains(node.getId)).toString
      )
    }

  private def flush(): Unit =
    Csv.write(config.directory, s"${config.runName}_episodes.csv", Header, episodeRows)
    Csv.write(config.directory, s"${config.runName}_full.csv", Header, fullRows)
    Csv.write(
      config.directory,
      s"${config.runName}_positions.csv",
      PositionHeader,
      positionRows
    )
    val diagnostics = learner.trainingDiagnostics
    if diagnostics.nonEmpty then
      Csv.write(config.directory, s"${config.runName}_training.csv", TrainingDiagnostic.Header, diagnostics.map(_.row))
end MobileExperiment

object MobileExperiment:
  /** Interval (s) between two rows of the full and positions exports. */
  val FrameInterval: Double = 2.0

  /** Population of the training network; evaluation arenas are scaled to keep its density. */
  val NominalPopulation: Int = 100

  /** Distance between the stream roots of two seeds: room for this many episodes per seed. */
  val IndependentSeedStride: Long = 100000L

  /** Offsets of the motion and source streams from the deployment stream of an episode. */
  val MotionStreamOffset: Long = 1000000000L
  val SourceStreamOffset: Long = 2000000000L

  /** The test topologies whose last episode is recorded in snapshots (spatial views, attention), chosen in advance. */
  val InspectedEvaluationSeed: Int = 202

  /** Sub-folder of the export directory with the snapshots of the inspected runs, one folder per run. */
  val SnapshotFolder: String = "spatial"

  /** The value of a node that has not computed one yet, and of an unreachable gradient. */
  val Unknown: Double = Double.PositiveInfinity
  val ClockTolerance: Double = 1e-8
  val MinimumSourceDistance: Double = 200.0

  /** Per node and frame: position, source flag, rounds and integrated absolute error since the episode start, and
    * whether the node belongs to a region that started moving.
    */
  val PositionHeader: Seq[String] =
    Seq("episode", "time", "node", "x", "y", "source", "rounds", "absoluteError", "region")
  val Header: Seq[String] = Seq("episode", "training", "time", "diffusion", "tolerance", "beta", "roundsPerNodeSecond",
    "normalizedError", "toleratedError", "withinTolerance", "reachableFraction", "recoverySeconds", "recoveryCensored",
    "rewardRate", "absoluteError")
end MobileExperiment
