package it.unibo.mobility

import java.io.File

/** Parameters of the separate mobile-gradient experiment. Diffusion is measured in m²/s, speed in m/s, tolerance and
  * loss scale in metres. `costSignal` and `lossScale` shape only the training costs: the exported metrics always report
  * the held error at the encoding scale. `event` happens halfway through the evaluation episodes, `trainingEvents`
  * cycle through the training ones; `movingShare` is the share of the nodes of a region that moves, for
  * `motionDuration` seconds once a region starts moving halfway (for ever by default).
  */
final case class MobilityConfig(
    seed: Int,
    algorithm: Int,
    diffusion: Double = 1.0,
    tolerance: Double = 5.0,
    beta: Double = 0.95,
    sourceChanges: Boolean = true,
    trainingEpisodes: Int = 10,
    episodes: Int = 20,
    episodeLength: Double = 200.0,
    movementInterval: Double = 0.25,
    recoveryDwell: Double = 5.0,
    recoveryFraction: Double = 0.95,
    trainingDiffusions: Seq[Double] = Seq(0.0, 0.5, 2.0),
    trainingTolerances: Seq[Double] = Seq(0.0, 5.0, 15.0),
    directory: File = new File("build/exports/mobility"),
    costSignal: CostSignal = CostSignal.Error,
    lossScale: Double = MobileEpisode.MobileEncodingScale,
    model: MobilityModel = MobilityModel.Brownian,
    speed: Double = 1.0,
    trainingSpeeds: Seq[Double] = Seq(0.0, 1.0, 2.0),
    warmStart: Boolean = false,
    event: EpisodeEvent = EpisodeEvent.SourceSwitch,
    trainingEvents: Seq[EpisodeEvent] = Seq(EpisodeEvent.SourceSwitch),
    movingShare: Double = MovingRegion.DefaultShare,
    motionDuration: Double = Double.PositiveInfinity,
    evaluationSeed: Option[Int] = None,
    evaluationNodes: Int = 100,
    targetProgram: it.unibo.scafi.BuildingBlock = it.unibo.scafi.BuildingBlock.Gradient,
    balanceScenarios: Boolean = false
):
  require(diffusion >= 0.0 && diffusion.isFinite && tolerance >= 0.0 && tolerance.isFinite)
  require(beta >= 0.0 && beta <= 1.0 && algorithm >= 0 && algorithm <= 10)
  require(trainingEpisodes >= 0 && episodes > trainingEpisodes)
  require(
    episodeLength > 0.0 && episodeLength.isFinite && movementInterval > 0.0 && movementInterval.isFinite && recoveryDwell > 0.0
  )
  require(
    math.abs(episodeLength / movementInterval - math.rint(episodeLength / movementInterval)) < 1e-8,
    "Episode length must be a multiple of the movement interval"
  )
  require(
    !sourceChanges || math.abs(
      episodeLength / (2.0 * movementInterval) - math.rint(episodeLength / (2.0 * movementInterval))
    ) < 1e-8,
    "Source change time must fall on a movement tick"
  )
  require(recoveryFraction > 0.0 && recoveryFraction <= 1.0)
  require(trainingDiffusions.nonEmpty && trainingDiffusions.forall(value => value >= 0.0 && value.isFinite))
  require(trainingTolerances.nonEmpty && trainingTolerances.forall(value => value >= 0.0 && value.isFinite))
  require(lossScale > 0.0 && lossScale.isFinite && speed >= 0.0 && speed.isFinite)
  require(trainingSpeeds.nonEmpty && trainingSpeeds.forall(value => value >= 0.0 && value.isFinite))
  require(trainingEvents.nonEmpty && movingShare > 0.0 && movingShare <= 1.0 && motionDuration > 0.0)

  require(evaluationNodes > 1)
  require(evaluationSeed.isEmpty || trainingEpisodes > 0, "Independent protocol requires a training episode")
  require(evaluationSeed.isEmpty || !warmStart, "Independent evaluation requires episode resets")

  def blockAt(episode: Int): it.unibo.scafi.BuildingBlock =
    if isTraining(episode) then it.unibo.scafi.BuildingBlock.Gradient else targetProgram

  def metricScaleAt(episode: Int): Double = blockAt(episode).errorScale

  def isTraining(episode: Int): Boolean = episode < trainingEpisodes
  def diffusionAt(episode: Int): Double = levelOf(episode, trainingDiffusions, diffusion)
  def speedAt(episode: Int): Double = levelOf(episode, trainingSpeeds, speed)

  /** How much the nodes move in `episode`: the diffusion of the Brownian model, the speed of the waypoint one. */
  def levelAt(episode: Int): Double = model match
    case MobilityModel.Brownian => diffusionAt(episode)
    case MobilityModel.Waypoint => speedAt(episode)

  /** The transfer controls train in a static network; the other learners cycle the training levels. */
  private def levelOf(episode: Int, training: Seq[Double], evaluation: Double): Double =
    if isTraining(episode) then
      if algorithm == MobilityConfig.TransferMLP || algorithm == MobilityConfig.TransferGNN then 0.0
      else training(episode % training.size)
    else evaluation

  /** What happens halfway through `episode`: the training events cycle like the training levels. */
  def eventAt(episode: Int): EpisodeEvent =
    if isTraining(episode) then trainingEvents(episode % trainingEvents.size) else event
  def toleranceAt(episode: Int): Double =
    if isTraining(episode) then trainingTolerances(episode % trainingTolerances.size) else tolerance
  def tolerances: Seq[Double] = (trainingTolerances :+ tolerance).distinct
  def runName: String =
    val level = model match
      case MobilityModel.Brownian => s"D$diffusion"
      case MobilityModel.Waypoint => s"V$speed"
    val scenario = Seq(
      (model == MobilityModel.Waypoint) -> "_waypoint",
      warmStart -> "_warm",
      (event == EpisodeEvent.RegionStartsMoving) -> "_region",
      (event == EpisodeEvent.SourceSwitchWhileRegionMoves) -> "_regionmoves",
      motionDuration.isFinite -> s"_for$motionDuration"
    ).collect { case (true, suffix) => suffix }
    val protocol =
      evaluationSeed.fold("")(value => s"_eval${value}_nodes${evaluationNodes}_program${targetProgram.code}")
    s"seed${seed}_algo${algorithm}_${level}_tol${tolerance}_beta${beta}_dt${movementInterval}_train${trainingEpisodes}_length${episodeLength}_source$sourceChanges${scenario.mkString}$protocol"
end MobilityConfig

object MobilityConfig:
  val TransferMLP: Int = 8
  val TransferGNN: Int = 9
  val FastestBaseline: Int = 10
