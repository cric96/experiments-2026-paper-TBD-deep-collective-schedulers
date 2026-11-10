package it.unibo.learning.network

import ai.djl.ndarray.NDManager
import ai.djl.nn.{Activation, Block}
import it.unibo.learning.encoding.StarGraph
import it.unibo.testing.PyTorchNumerics
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.util.{Random, Using}

class TorchNetworkSpec extends AnyFlatSpec with Matchers:
  private val tolerance = 1e-5f
  private val inputs = 5

  private def randomInputs(size: Int, count: Int = 4, seed: Int = 3): Seq[Array[Float]] =
    val random = new Random(seed)
    Seq.fill(count)(Array.fill(size)(random.nextFloat() * 2 - 1))

  private def evaluated(network: TorchNetwork[Block], input: Array[Float]): Array[Float] =
    Using.resource(NDManager.newBaseManager())(manager => network.evaluate(manager.create(Array(input))).toFloatArray)

  private def mlp(withTanh: Boolean): TorchNetwork[Block] =
    TorchNetwork.createMLP("mlp", inputs, Seq(8, 8), 3, Option.when(withTanh)(Activation.tanhBlock()), seed = 5)

  private def edgeGat(): TorchNetwork[EdgeGATBlock] =
    TorchNetwork.createEdgeGAT(
      "gat", StarGraph.SelfDim, StarGraph.MaxNeighbors, StarGraph.NeighDim, edgeHidden = 6, readoutHidden = 7,
      outputActivation = Some(Activation.tanhBlock()), seed = 9
    )

  "The fast MLP inference" should "agree with the DJL evaluation, with and without the output tanh" in
    Seq(false, true).foreach { withTanh =>
      Using.resource(mlp(withTanh)) { network =>
        val fast = network.snapshotFastMLP(useTanh = withTanh)
        randomInputs(inputs).foreach { input =>
          fast(input).zip(evaluated(network, input)).foreach { case (actual, expected) =>
            actual shouldBe expected +- tolerance
          }
        }
      }
    }

  it should "keep its exact output (reproducibility of the experiments)" taggedAs PyTorchNumerics in {
    val fast = Using.resource(mlp(withTanh = true))(_.snapshotFastMLP(useTanh = true))
    randomInputs(inputs).map(fast(_).toSeq) shouldBe Seq(
      Seq(0.069943115f, -0.4245219f, -0.5224342f),
      Seq(-0.28166482f, -0.92035604f, -0.41430202f),
      Seq(2.3045391e-4f, -0.66744125f, -0.17583466f),
      Seq(-0.46670738f, -0.9289665f, -0.92314935f)
    )
  }

  "The fast EdgeGAT inference" should "agree with the DJL evaluation" in
    Using.resource(edgeGat()) { network =>
      val fast = network.block.snapshotFastInference()
      randomInputs(StarGraph.TotalDim).foreach { input =>
        fast(input) shouldBe evaluated(network, input).head +- tolerance
      }
    }

  it should "keep its exact output (reproducibility of the experiments)" taggedAs PyTorchNumerics in {
    val fast = Using.resource(edgeGat())(_.block.snapshotFastInference())
    randomInputs(StarGraph.TotalDim).map(fast) shouldBe Seq(-0.6781583f, -0.15681955f, -0.8604447f, -0.33066946f)
  }

  "The EdgeGAT inspection" should "expose the readout embedding and a distribution of attention over the slots" in {
    val inspect = Using.resource(edgeGat())(_.block.snapshotEmbeddingAndAttention())
    randomInputs(StarGraph.TotalDim).foreach { input =>
      val (embedding, attention) = inspect(input)
      embedding should have size 7
      attention should have size StarGraph.MaxNeighbors.toLong
      attention.sum shouldBe 1.0f +- tolerance
    }
  }

  it should "keep its exact output (reproducibility of the experiments)" taggedAs PyTorchNumerics in {
    val inspect = Using.resource(edgeGat())(_.block.snapshotEmbeddingAndAttention())
    randomInputs(StarGraph.TotalDim).map(inspect).map { case (embedding, attention) =>
      (embedding.toSeq, attention.toSeq)
    } shouldBe Seq(
      (
        Seq(0.40049446f, 1.5598043f, 0.0f, 0.74864763f, 0.9247464f, 0.0f, 0.0f),
        Seq(0.11662553f, 0.11897848f, 0.12989838f, 0.12617283f, 0.13449723f, 0.14164001f, 0.036078185f, 0.19610935f)
      ),
      (
        Seq(0.0f, 0.9738458f, 0.0f, 0.42628846f, 0.8336824f, 0.0f, 0.0f),
        Seq(0.11670277f, 0.14406587f, 0.13306713f, 0.14775245f, 0.16513373f, 0.08524253f, 0.104862176f, 0.1031734f)
      ),
      (
        Seq(0.0f, 0.9255263f, 0.0f, 0.76856154f, 0.0f, 0.0f, 0.0f),
        Seq(0.1345941f, 0.11035934f, 0.2124181f, 0.11965413f, 0.067885034f, 0.107664995f, 0.053481445f, 0.19394286f)
      ),
      (
        Seq(0.0f, 0.9574197f, 0.50799847f, 0.0f, 0.0f, 0.13777144f, 0.0f),
        Seq(0.16736926f, 0.11832797f, 0.09149124f, 0.17672804f, 0.1560559f, 0.08812095f, 0.10037574f, 0.10153086f)
      )
    )
  }

  "The min-max action scaler" should "map [min, max] to [-1, 1] and back, clamping outside" in {
    val scaler = MinMaxActionScaler(0.5, 5.0)
    scaler.scale(0.5) shouldBe -1.0f
    scaler.scale(5.0) shouldBe 1.0f
    scaler.scale(9.0) shouldBe 1.0f
    scaler.unscale(0.0) shouldBe 2.75
    scaler.unscale(-3.0) shouldBe 0.5
    scaler.unscale(scaler.scale(3.2).toDouble) shouldBe 3.2 +- 1e-6
  }

  "The smooth L1 loss" should "be quadratic for small errors and linear for large ones" in
    Using.resource(NDManager.newBaseManager()) { manager =>
      val predicted = manager.create(Array(0.5f, 3.0f))
      val target = manager.create(Array(0.0f, 0.0f))
      TorchNetwork.smoothL1Loss(predicted, target).getFloat() shouldBe ((0.125f + 2.5f) / 2) +- tolerance
    }

  "The squared loss" should "be quadratic for every error" in
    Using.resource(NDManager.newBaseManager()) { manager =>
      val predicted = manager.create(Array(0.5f, 3.0f))
      val target = manager.create(Array(0.0f, 0.0f))
      TorchNetwork.squaredLoss(predicted, target).getFloat() shouldBe ((0.125f + 4.5f) / 2) +- tolerance
    }
end TorchNetworkSpec
