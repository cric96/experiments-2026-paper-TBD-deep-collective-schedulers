package it.unibo.learning.network

import ai.djl.Model
import ai.djl.engine.Engine
import ai.djl.ndarray.types.Shape
import ai.djl.ndarray.{NDArray, NDList, NDManager}
import ai.djl.nn.core.Linear
import ai.djl.nn.{Activation, Block, Parameter, SequentialBlock}
import ai.djl.training.initializer.XavierInitializer
import ai.djl.training.loss.Loss
import ai.djl.training.optimizer.Optimizer
import ai.djl.training.tracker.Tracker
import ai.djl.training.{DefaultTrainingConfig, ParameterStore, Trainer}
import it.unibo.learning.network.TorchNetwork.serialized

import scala.jdk.CollectionConverters.*
import scala.util.Using

/** A trainable DJL (PyTorch) network made of `block`, optimized with Adam and gradient clipping. All the DJL calls go
  * through a single lock.
  */
final class TorchNetwork[+B <: Block](
    val name: String,
    val inputDim: Int,
    val block: B,
    val lr: Float = TorchNetwork.DefaultLearningRate,
    val clipGrad: Float = TorchNetwork.DefaultGradientClip,
    val seed: Int = TorchNetwork.DefaultSeed
) extends AutoCloseable:

  val model: Model = serialized {
    val model = Model.newInstance(name)
    model.setBlock(block)
    model
  }

  /** Created right after seeding the engine, so that the Xavier initialization of the parameters is reproducible. */
  val trainer: Trainer = serialized {
    Engine.getInstance().setRandomSeed(seed)
    val trainer = model.newTrainer(trainingConfiguration)
    trainer.initialize(new Shape(1, inputDim.toLong))
    trainer
  }

  private def trainingConfiguration: DefaultTrainingConfig =
    val optimizer = Optimizer.adam().optLearningRateTracker(Tracker.fixed(lr)).optClipGrad(clipGrad).build()
    new DefaultTrainingConfig(Loss.l2Loss())
      .optOptimizer(optimizer)
      .optInitializer(new XavierInitializer(), Parameter.Type.WEIGHT)

  def manager: NDManager = trainer.getManager

  def forward(input: NDArray, training: Boolean = false): NDArray = serialized {
    if training then trainer.forward(new NDList(input)).singletonOrThrow()
    else trainer.evaluate(new NDList(input)).singletonOrThrow()
  }

  def evaluate(input: NDArray): NDArray = serialized(trainer.evaluate(new NDList(input)).singletonOrThrow())

  /** A lock-free evaluation of this network, which must be a sequence of linear layers and ReLUs (see
    * [[TorchNetwork.createMLP]]), on a snapshot of its current parameters.
    */
  def snapshotFastMLP(useTanh: Boolean = false): Array[Float] => Array[Float] =
    serialized(FastInference.mlp(FastInference.denseLayersOf(block), useTanh))

  /** The activation of the last hidden layer, i.e. the input of the output head, of a sequential network. */
  def extractHiddenEmbedding(input: NDArray): NDArray = serialized {
    val layers = block.getChildren.values().asScala.toSeq
    val hiddenLayers = layers.take(layers.lastIndexWhere { case _: Linear => true; case _ => false })
    val parameterStore = new ParameterStore(manager, false)
    hiddenLayers
      .foldLeft(new NDList(input))((activation, layer) => layer.forward(parameterStore, activation, false))
      .singletonOrThrow()
  }

  /** One optimization step on the loss computed by `loss`. */
  def trainStep(loss: => NDArray): Unit = serialized {
    Using.resource(trainer.newGradientCollector())(collector => collector.backward(loss))
    trainer.step()
  }

  /** Runs `body` (typically another network's training step) without updating this network's parameters. */
  def withFrozenParameters[A](body: => A): A = serialized {
    block.freezeParameters(true)
    try body
    finally block.freezeParameters(false)
  }

  /** Polyak averaging: θ ← τ · θ_source + (1 − τ) · θ. */
  def softUpdateFrom(source: TorchNetwork[Block], tau: Float): Unit = serialized {
    Using.resource(manager.newSubManager()) { scratch =>
      val parameters = block.getParameters.values().asScala
      val sourceParameters = source.block.getParameters.values().asScala
      parameters.zip(sourceParameters).foreach { case (parameter, sourceParameter) =>
        val current = parameter.getArray
        val updated = sourceParameter.getArray.toDevice(current.getDevice, false).mul(tau).add(current.mul(1.0f - tau))
        updated.copyTo(current)
        updated.attach(scratch)
      }
    }
  }

  override def close(): Unit =
    trainer.close()
    model.close()
end TorchNetwork

object TorchNetwork:
  val DefaultLearningRate: Float = 1e-3f
  val DefaultGradientClip: Float = 10.0f
  val DefaultSeed: Int = 42

  /** Errors below this value are penalized quadratically by the smooth L1 loss, larger ones linearly. */
  val HuberDelta: Float = 1.0f

  private val lock = new AnyRef

  /** Runs `body` holding the lock that serializes every DJL call. */
  private[network] def serialized[A](body: => A): A = lock.synchronized(body)

  /** Runs `body` with a child manager of `parent`, whose arrays are released afterwards. */
  def withSubManager[A](parent: NDManager)(body: NDManager => A): A =
    serialized(Using.resource(parent.newSubManager())(body))

  /** Linear layers with ReLU for every hidden size, then a linear output layer and the optional output activation. */
  def mlpBlock(hiddenSizes: Seq[Int], outputSize: Int, outputActivation: Option[Block] = None): SequentialBlock =
    def linear(units: Int): Block = Linear.builder().setUnits(units.toLong).build()
    val layers = hiddenSizes.flatMap(units => Seq(linear(units), Activation.reluBlock())) ++
      Seq(linear(outputSize)) ++ outputActivation
    layers.foldLeft(new SequentialBlock())(_.add(_))

  def createMLP(
      name: String,
      inputDim: Int,
      hiddenSizes: Seq[Int],
      outputSize: Int,
      outputActivation: Option[Block] = None,
      lr: Float = DefaultLearningRate,
      seed: Int = DefaultSeed
  ): TorchNetwork[SequentialBlock] =
    TorchNetwork(name, inputDim, mlpBlock(hiddenSizes, outputSize, outputActivation), lr, seed = seed)

  def createEdgeGAT(
      name: String,
      selfDim: Int,
      maxNeighbors: Int,
      neighDim: Int,
      edgeHidden: Int,
      readoutHidden: Int,
      outputSize: Int = 1,
      outputActivation: Option[Block] = None,
      lr: Float = DefaultLearningRate,
      seed: Int = DefaultSeed
  ): TorchNetwork[EdgeGATBlock] =
    val block =
      EdgeGATBlock(selfDim, maxNeighbors, neighDim, edgeHidden, readoutHidden, outputSize, outputActivation)
    TorchNetwork(name, selfDim + maxNeighbors * neighDim, block, lr, seed = seed)

  /** Squared loss: 0.5 · x², averaged; it agrees with the smooth L1 loss on the errors below δ. */
  def squaredLoss(predicted: NDArray, target: NDArray): NDArray = predicted.sub(target).pow(2).mul(0.5f).mean()

  /** Smooth L1 (Huber) loss: 0.5 · x² if |x| < δ, |x| − 0.5 otherwise, averaged. */
  def smoothL1Loss(predicted: NDArray, target: NDArray): NDArray =
    val error = predicted.sub(target).abs()
    val quadratic = error.minimum(HuberDelta)
    val linear = error.sub(quadratic)
    quadratic.pow(2).mul(0.5f).add(linear).mean()
end TorchNetwork
