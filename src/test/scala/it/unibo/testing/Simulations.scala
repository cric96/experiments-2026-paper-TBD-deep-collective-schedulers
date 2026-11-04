package it.unibo.testing

import it.unibo.alchemist.boundary.{LoadAlchemist, OutputMonitor}
import it.unibo.alchemist.core.Simulation
import it.unibo.alchemist.model.scafi.nodes.SimpleNodeManager
import it.unibo.alchemist.model.terminators.AfterTime
import it.unibo.alchemist.model.times.DoubleTime
import it.unibo.alchemist.model.{Environment, Position, Reaction, Time}
import it.unibo.scafi.Sensors

import java.io.File
import scala.collection.mutable.ListBuffer
import scala.jdk.CollectionConverters.*

/** Runs the YAML simulations headless and without exporters, to pin their observable behaviour. */
object Simulations:

  /** Aggregated, order-independent view of the state of all the nodes at a given time. */
  final case class Fingerprint(
      time: Double,
      nodes: Int,
      totalTicks: Long,
      wakeUpSum: Double,
      rewardSum: Double,
      errorSum: Double,
      outputSum: Double
  )

  /** The simulation described by `src/main/yaml/<yaml>.yml` with `variables`, without its exporters. */
  def load(yaml: String, variables: Map[String, Any]): Simulation[Any, Nothing] =
    loadFile(new File(s"src/main/yaml/$yaml.yml"), variables)

  /** The simulation described by `file` with `variables`, without its exporters. */
  def loadFile(file: File, variables: Map[String, Any] = Map.empty): Simulation[Any, Nothing] =
    val simulation = LoadAlchemist.from(file).getWith[Any, Nothing](variables.asJava)
    simulation.getOutputMonitors.asScala.toList.foreach(simulation.removeOutputMonitor)
    simulation

  /** Runs `simulation` until `time` and returns its environment. */
  def runUntil[P <: Position[P]](simulation: Simulation[Any, P], time: Double): Environment[Any, P] =
    simulation.getEnvironment.addTerminator(new AfterTime[Any, P](new DoubleTime(time)))
    simulation.play()
    simulation.run()
    Option(simulation.getError.getOrNull()).foreach(error => throw error)
    simulation.getEnvironment

  /** Runs `src/main/yaml/<yaml>.yml` and returns a fingerprint of the nodes as soon as each checkpoint is passed. */
  def run(yaml: String, variables: Map[String, Any], checkpoints: Seq[Double]): Seq[Fingerprint] =
    simulate[Nothing](load(yaml, variables), checkpoints.sorted)

  private def simulate[P <: Position[P]](simulation: Simulation[Any, P], checkpoints: Seq[Double]): Seq[Fingerprint] =
    val fingerprints = ListBuffer.empty[Fingerprint]
    simulation.addOutputMonitor(new OutputMonitor[Any, P]:
      override def stepDone(environment: Environment[Any, P], reaction: Reaction[Any], time: Time, step: Long): Unit =
        checkpoints.drop(fingerprints.size).headOption.filter(_ <= time.toDouble).foreach { checkpoint =>
          fingerprints += fingerprintOf(environment, checkpoint)
        })
    runUntil(simulation, checkpoints.max)
    fingerprints.toList

  private def fingerprintOf(environment: Environment[Any, ?], time: Double): Fingerprint =
    val managers = environment.getNodes.getCurrent.asScala.toList.map(SimpleNodeManager[Any](_))
    def sumOf(molecule: String): Double =
      managers.filter(_.has(molecule)).map(_.get[Double](molecule)).filter(_.isFinite).sum
    Fingerprint(
      time = time,
      nodes = managers.size,
      totalTicks = managers.filter(_.has(Sensors.ticks)).map(_.get[Number](Sensors.ticks).longValue()).sum,
      wakeUpSum = sumOf(Sensors.nextWakeUp),
      rewardSum = sumOf(Sensors.accumulatedReward),
      errorSum = sumOf(Sensors.accumulatedError),
      outputSum = sumOf(Sensors.localComputation)
    )
end Simulations
