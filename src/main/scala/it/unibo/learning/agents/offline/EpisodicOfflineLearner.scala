package it.unibo.learning.agents.offline

import ai.djl.ndarray.NDManager
import it.unibo.learning.abstractions.Trajectory
import it.unibo.learning.agents.Learner
import it.unibo.learning.network.TorchNetwork

import scala.util.Random

/** A learner that stores the trajectories of every episode as transitions of type `T` and trains offline at the end of
  * the episode, as prescribed by its [[TrainingSchedule]].
  */
abstract class EpisodicOfflineLearner[T](
    runSeed: Double,
    weight: Double,
    protected val schedule: TrainingSchedule,
    batchSize: Int,
    transitionCapacity: Int = Int.MaxValue
) extends Learner:
  private var randomness: Random = LearnerSeeds.random(runSeed, weight)
  private val rootManager: NDManager = NDManager.newBaseManager()
  private var episode: Int = 0

  protected val transitions: StratifiedTransitions[T] = StratifiedTransitions[T](transitionCapacity)

  /** The episode in progress (0-based); during [[onTrained]] it is the episode that has just ended. */
  protected def currentEpisode: Int = episode

  protected def random: Random = randomness

  override def trainingScenario(scenario: Int): Unit = transitions.scenario(scenario)

  override def injectRandom(random: Random): Unit = randomness = random

  /** Stores the trajectories and trains, as long as the schedule prescribes training: afterwards the transitions would
    * never be used.
    */
  override def endEpisode(trajectories: Seq[Trajectory]): Unit =
    schedule.trainingStepsAfter(episode).foreach { steps =>
      trajectories.foreach(record)
      if !transitions.isEmpty then
        train(steps)
        onTrained()
    }
    episode += 1

  /** Converts a trajectory into transitions and stores them in [[transitions]]. */
  protected def record(trajectory: Trajectory): Unit

  /** Performs one gradient step on `batch`, allocating its tensors in `manager`. */
  protected def trainOn(batch: Vector[T], manager: NDManager): Unit

  /** Called after every training session, e.g. to refresh the snapshots used for fast inference. */
  protected def onTrained(): Unit

  /** Runs `body` with a manager whose tensors are released afterwards. */
  protected def withScratchManager[A](body: NDManager => A): A = TorchNetwork.withSubManager(rootManager)(body)

  private def train(steps: Int): Unit =
    val effectiveBatchSize = math.min(batchSize, transitions.size)
    (0 until steps).foreach { _ =>
      val batch = transitions.sample(effectiveBatchSize, random)
      withScratchManager(manager => trainOn(batch, manager))
    }
end EpisodicOfflineLearner
