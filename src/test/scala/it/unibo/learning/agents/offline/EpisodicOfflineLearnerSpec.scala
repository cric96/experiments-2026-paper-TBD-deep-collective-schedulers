package it.unibo.learning.agents.offline

import ai.djl.ndarray.NDManager
import it.unibo.learning.abstractions.{AgentState, Step, Trajectory}
import it.unibo.testing.AgentStates
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class EpisodicOfflineLearnerSpec extends AnyFlatSpec with Matchers:
  private val schedule = TrainingSchedule(explorationEpisodes = 2, stepsPerEpisode = 3, finalTrainingMultiplier = 2)
  private val trajectory: Trajectory = Seq(Step(AgentStates.isolated, 0, 1.0, 0.0))

  /** Answers the diagnostics as if the critic preferred the longest period and the actor chose the shortest one. */
  private object LazyCritic extends DiagnosticProbe[Int]:
    override val grid: Seq[Double] = Seq(0.5, 5.0)
    override def betaOf(transition: Int): Option[Double] = Some(transition.toDouble / 10)
    override def probe(batch: Vector[Int], manager: NDManager): ProbedBatch =
      ProbedBatch(
        criticLoss = 0.25,
        Seq.fill(batch.size)(0.5),
        Seq(Seq.fill(batch.size)(-1.0), Seq.fill(batch.size)(0.0))
      )

  /** Records one transition per step and counts the gradient steps and the training sessions. */
  final private class CountingLearner(diagnosed: Boolean = false)
      extends EpisodicOfflineLearner[Int](2.0, 0.95, schedule, batchSize = 4):
    var gradientSteps: Int = 0
    var sessions: Int = 0
    def recorded: Int = transitions.size
    override def policy: AgentState => Double = _ => 1.0
    override protected def record(trajectory: Trajectory): Unit = trajectory.foreach(_ => transitions.add(1, true))
    override protected def trainOn(batch: Vector[Int], manager: NDManager): Unit = gradientSteps += 1
    override protected def onTrained(): Unit = sessions += 1
    override protected def diagnosticProbe: Option[DiagnosticProbe[Int]] = Option.when(diagnosed)(LazyCritic)

  "An episodic offline learner" should "train after each exploration episode, longer after the last one" in {
    val learner = CountingLearner()
    (1 to 5).foreach(_ => learner.endEpisode(Seq(trajectory)))
    learner.sessions shouldBe 2
    learner.gradientSteps shouldBe 3 + 3 * 2
  }

  it should "stop storing transitions once it no longer trains" in {
    val learner = CountingLearner()
    (1 to 5).foreach(_ => learner.endEpisode(Seq(trajectory, trajectory)))
    learner.recorded shouldBe 2 * 2
  }

  it should "not train without transitions" in {
    val learner = CountingLearner()
    learner.endEpisode(Seq.empty)
    learner.sessions shouldBe 0
  }

  it should "diagnose its networks after every training session, by regime and trade-off" in {
    val learner = CountingLearner(diagnosed = true)
    (1 to 3).foreach(_ => learner.endEpisode(Seq(trajectory)))
    learner.trainingDiagnostics shouldBe Seq(0, 1).map { episode =>
      TrainingDiagnostic(episode, beta = 0.1, unstable = true, TrainingDiagnostics.SamplesPerRegime, 0.25, 0.5, 5.0)
    }
    CountingLearner().trainingDiagnostics shouldBe empty
  }
end EpisodicOfflineLearnerSpec
