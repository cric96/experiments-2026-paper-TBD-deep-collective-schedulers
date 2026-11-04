package it.unibo.learning.encoding

import it.unibo.learning.abstractions.AgentState
import it.unibo.learning.abstractions.AgentState.{NeighborInfo, Snapshot}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.collection.immutable.Queue

class MobileEncodingSpec extends AnyFlatSpec with Matchers:
  private val MaxPeriod = 5.0
  private val Beta = 0.95
  private val All = MobileFeatures(ownSpeed = true, unexplainedChange = true, dependants = true)

  /** Node 0 at 100 m from the source, observed 2 s apart; neighbour 1 is its parent, neighbour 2 its child. */
  private def state(
      ownBefore: Double,
      ownNow: Double,
      parentDistanceNow: Double,
      speed: Double = 0.0,
      parentSpeed: Double = 0.0
  ): AgentState =
    val previous: Snapshot = Map(
      0 -> NeighborInfo(ownBefore, 0.0, 1.0, 10.0),
      1 -> NeighborInfo(60.0, 40.0, 1.0, 10.0),
      2 -> NeighborInfo(130.0, 30.0, 1.0, 10.0)
    )
    val current: Snapshot = Map(
      0 -> NeighborInfo(ownNow, 0.0, 1.0, 12.0),
      1 -> NeighborInfo(60.0, parentDistanceNow, 1.0, 12.0, parentSpeed),
      2 -> NeighborInfo(ownNow + 30.0, 30.0, 1.0, 12.0)
    )
    AgentState(0, Queue(previous, current), speed = speed)

  private def optional(encoded: Array[Float], features: MobileFeatures): Seq[Float] =
    encoded.drop(MobileEncoding.featureDim(MobileFeatures.None)).take(features.count).toSeq

  "The mobile encoding" should "keep its size and values without optional features" in {
    val still = state(100.0, 100.0, 40.0)
    MobileEncoding.spatial(still, Beta, MaxPeriod).length shouldBe MobileEncoding.FeatureDim
    MobileEncoding.spatial(still, Beta, MaxPeriod, MobileFeatures.None).toSeq shouldBe
      MobileEncoding.spatial(still, Beta, MaxPeriod).toSeq
    MobileEncoding.graph(still, Beta, MaxPeriod, All).length shouldBe
      MobileEncoding.selfDim(All) + StarGraph.MaxNeighbors * MobileEncoding.NeighDim
  }

  it should "append the own speed, the change not explained by the motion and the dependants, in order" in {
    val features = optional(MobileEncoding.spatial(state(100.0, 100.0, 40.0, speed = 1.5), Beta, MaxPeriod, All), All)
    features.head shouldBe math.tanh(1.5 / MobileEncoding.SpeedScale).toFloat
    features(1) shouldBe 0.0f
    features(2) shouldBe (1.0f / StarGraph.MaxNeighbors)
  }

  it should "explain a change of the own value by a change of the distances" in {
    def unexplained(parentDistanceNow: Double): Float =
      optional(MobileEncoding.spatial(state(100.0, 110.0, parentDistanceNow), Beta, MaxPeriod, All), All)(1)
    unexplained(parentDistanceNow = 50.0) shouldBe 0.0f
    unexplained(parentDistanceNow = 40.0) should be > 0.0f
  }

  it should "select the optional features independently" in {
    val speedOnly = MobileFeatures(ownSpeed = true)
    val encoded = MobileEncoding.graph(state(100.0, 100.0, 40.0, speed = 1.5), Beta, MaxPeriod, speedOnly)
    encoded.length shouldBe MobileEncoding.selfDim(speedOnly) + StarGraph.MaxNeighbors * MobileEncoding.NeighDim
    encoded(MobileEncoding.SelfDim) shouldBe math.tanh(1.5 / MobileEncoding.SpeedScale).toFloat
  }
  it should "describe how fast the neighbours move, on average and neighbour by neighbour" in {
    val features = MobileFeatures(neighbourSpeed = true)
    val moving = state(100.0, 100.0, 40.0, parentSpeed = 2.0)
    val averaged = math.tanh(2.0 / MobileEncoding.SpeedScale).toFloat / 2.0f
    optional(MobileEncoding.spatial(moving, Beta, MaxPeriod, features), features) shouldBe Seq(averaged)
    val graph = MobileEncoding.graph(moving, Beta, MaxPeriod, features)
    val neighDim = MobileEncoding.neighDim(features)
    graph.length shouldBe MobileEncoding.selfDim(features) + StarGraph.MaxNeighbors * neighDim
    graph(MobileEncoding.SelfDim) shouldBe averaged
    def slotSpeed(slot: Int): Float = graph(MobileEncoding.selfDim(features) + (slot + 1) * neighDim - 1)
    slotSpeed(0) shouldBe 0.0f // the child, nearest
    slotSpeed(1) shouldBe math.tanh(2.0 / MobileEncoding.SpeedScale).toFloat // the parent
    slotSpeed(2) shouldBe 0.0f // padding
  }
  it should "blank only the features that assume the gradient's semantics when program-agnostic" in {
    val moving = state(100.0, 110.0, 50.0)
    val agnostic = MobileFeatures(programAgnostic = true)
    agnostic.count shouldBe 0
    val spatial = MobileEncoding.spatial(moving, Beta, MaxPeriod)
    val blankedSpatial = MobileEncoding.spatial(moving, Beta, MaxPeriod, agnostic)
    val spatialBlanks = Set(SpatialTemporal.Feature.Residual, SpatialTemporal.Feature.ChildrenRatio)
    spatial(SpatialTemporal.Feature.ChildrenRatio) should not be 0.0f
    spatialBlanks.foreach(index => blankedSpatial(index) shouldBe 0.0f)
    spatial.indices.filterNot(spatialBlanks).foreach(index => blankedSpatial(index) shouldBe spatial(index))
    val graph = MobileEncoding.graph(moving, Beta, MaxPeriod)
    val blankedGraph = MobileEncoding.graph(moving, Beta, MaxPeriod, agnostic)
    val graphBlanks = StarGraph.ChildrenFeature +: (0 until StarGraph.MaxNeighbors).map(slot =>
      MobileEncoding.SelfDim + slot * MobileEncoding.NeighDim + StarGraph.RelativeOrderFeature
    )
    graph(StarGraph.ChildrenFeature) should not be 0.0f
    graphBlanks.foreach(index => blankedGraph(index) shouldBe 0.0f)
    graph.indices.filterNot(graphBlanks.toSet).foreach(index => blankedGraph(index) shouldBe graph(index))
  }
end MobileEncodingSpec
