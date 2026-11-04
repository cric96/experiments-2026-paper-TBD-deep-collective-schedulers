package it.unibo.alchemist.model.reactions

import it.unibo.alchemist.KotlinInterop.command
import it.unibo.alchemist.Molecules.*
import it.unibo.alchemist.model.*
import it.unibo.learning.abstractions.{AgentState, ReplayBuffer, Step}
import it.unibo.learning.agents.Learner
import it.unibo.learning.{Box, LearningInfo}
import it.unibo.scafi.{BuildingBlock, EvaluationProtocol, GroundTruths, SchedulingCoordinator, Sensors, Topology}

import scala.collection.immutable.Queue
import scala.jdk.CollectionConverters.*
import scala.util.Random

/** The global reaction that drives the experiment: it trains the shared learner (online at every step, offline at the
  * end of each episode), starts each episode on a new deployment, provides the ground truths used to measure the nodes'
  * error, and records the final episode of the inspected run.
  */
final class CentralAgent[T, P <: Position[P]](
    environment: Environment[T, P],
    distribution: TimeDistribution[T],
    learningInfo: LearningInfo,
    learner: Learner,
    environmentBox: Box
) extends AbstractGlobalReaction[T, P](environment, distribution)
    with SchedulingCoordinator:
  private val randomness = new Random(learningInfo.randomSeed)
  private val redeployment = Redeployment(environment, environmentBox, learningInfo.randomSeed)
  private val snapshots = FinalEpisodeSnapshots(environment, learner, learningInfo, snapshotGroundTruth(_))
  private def snapshotGroundTruth(nodeId: Int): Double =
    val target = agents.headOption
      .flatMap(_.valueOf[Any](Sensors.program))
      .map(BuildingBlock.fromMolecule)
      .getOrElse(BuildingBlock.Gradient)
    groundTruth(nodeId, activeBlock(target))

  private var steps: Int = 0
  private var episode: Int = 0

  /** The ground truths of the current episode, recomputed when the sources or node set change. */
  private var groundTruthsCache: Option[(Int, Set[Int], GroundTruths)] = None

  /** The experience of all the nodes, for the learners that train online. */
  val memory: ReplayBuffer = ReplayBuffer(learningInfo.bufferSize, randomness)
  learner.injectRandom(randomness)

  override protected def onInitializationComplete(time: Time, environment: Environment[T, ?]): Unit =
    redeployment.capturePrototypes()

  override protected def react(): Unit =
    val now = environment.getSimulation.getTime
    if steps > 0 then // the first step happens at the very beginning, before any round
      trainOnline()
      if steps % learningInfo.episodeSize == 0 then endEpisode()
    steps += 1
    snapshots.observe(now.toDouble)

  private def trainOnline(): Unit =
    if memory.totalSize >= learningInfo.batchSize then learner.update(memory.sample(learningInfo.batchSize))

  private def endEpisode(): Unit =
    learner.endEpisode(agents.map(_.valueOf[Queue[Step]](Sensors.experiences).getOrElse(Queue.empty)))
    val positions = redeployment.positionsFor(episode)
    environment.getSimulation.schedule(command {
      redeployment.redeploy(positions)
      episode += 1
    })
    learner.decayingParameters.foreach { case (_, parameter) => parameter.update() }

  /** The policy of the learner; in the generalization scenario, conditioned on the weight under evaluation. */
  def policy: AgentState => Double =
    EvaluationProtocol
      .evaluationWeightAt(episode)
      .filter(_ => isGeneralizationScenario)
      .fold(learner.policy)(learner.policyForWeight)

  private def isGeneralizationScenario: Boolean = agents.headOption.exists(_.valueOf[Any](Sensors.program).nonEmpty)

  /** The current value of the decaying hyper-parameter `name` of the learner (0 if it has none). */
  def decayVariableOf(name: String): Double =
    learner.decayingParameters.collectFirst { case (`name`, parameter) => parameter.value }.getOrElse(0.0)

  /** The building block the nodes run in the current episode, given the one they are configured for. */
  def activeBlock(target: BuildingBlock): BuildingBlock = EvaluationProtocol.blockAt(episode, target)

  /** The value `block` should converge to in node `nodeId`, given the current sources. */
  def groundTruth(nodeId: Int, block: BuildingBlock = BuildingBlock.Gradient): Double =
    currentGroundTruths.of(nodeId, block)

  private def currentGroundTruths: GroundTruths =
    val sources = currentSources
    val nodes = agents.map(_.getId).toSet
    groundTruthsCache
      .collect {
        case (cachedEpisode, cachedNodes, truths)
            if cachedEpisode == episode && cachedNodes == nodes && truths.sources == sources =>
          truths
      }
      .getOrElse {
        val truths = GroundTruths.of(topology, sources)
        groundTruthsCache = Some((episode, nodes, truths))
        truths
      }

  private def currentSources: Set[Int] =
    agents.filter(_.valueOf[Boolean](Sensors.source).getOrElse(false)).map(_.getId).toSet

  private def topology: Topology =
    def linksOf(node: Node[T]): Seq[(Int, Double)] =
      val position = environment.getCurrentPosition(node)
      environment.getNeighborhood(node).getCurrent.getNeighbors.asScala.toSeq.map { neighbour =>
        neighbour.getId -> position.distanceTo(environment.getCurrentPosition(neighbour))
      }
    Topology(agents.map(_.getId), agents.map(node => node.getId -> linksOf(node)).toMap)
end CentralAgent
