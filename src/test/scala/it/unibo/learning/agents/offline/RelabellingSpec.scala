package it.unibo.learning.agents.offline

import it.unibo.learning.TradeOff
import it.unibo.learning.abstractions.{AgentState, MobileOutcome, Step}
import it.unibo.learning.encoding.InstabilityIntensity
import it.unibo.testing.AgentStates.*
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class RelabellingSpec extends AnyFlatSpec with Matchers:
  private val gamma = 0.9
  private val ownWeight = 0.95
  private val betaIndex = 1
  private val trajectory = Seq(
    Step(sourceMovingTowardsMe, actionIndex = 0, action = 1.0, reward = -0.1),
    Step(stableBackbone, actionIndex = 2, action = 3.0, reward = -0.2),
    Step(isolated, actionIndex = 1, action = 0.2, reward = -0.3)
  )

  /** Encodes a state as (node id, β feature), so that the tests can tell states apart. */
  private def encode(state: AgentState): Array[Float] = Array(state.me.toFloat, TradeOff.betaFeature(ownWeight))

  private val relabelled = Relabelling.relabel(trajectory, ownWeight, gamma, encode, betaIndex)
  private val weightsPerStep = Relabelling.targetWeights(ownWeight).size

  "The target weights" should "repeat the learner's own weight before the counterfactual ones" in {
    Relabelling.targetWeights(ownWeight) shouldBe Seq.fill(Relabelling.OwnWeightRepetitions)(ownWeight) ++
      Relabelling.CounterfactualWeights
  }

  "Relabelling" should "produce one transition per step and target weight, in order" in {
    relabelled should have size (trajectory.size * weightsPerStep).toLong
    relabelled.map(_.step) shouldBe trajectory.flatMap(Seq.fill(weightsPerStep)(_))
    relabelled.take(weightsPerStep).map(_.beta) shouldBe Relabelling.targetWeights(ownWeight)
  }

  it should "condition both states on the target weight" in
    relabelled.foreach { transition =>
      transition.state(betaIndex) shouldBe TradeOff.betaFeature(transition.beta)
      transition.nextState(betaIndex) shouldBe TradeOff.betaFeature(transition.beta)
    }

  "The counterfactual weights" should "be those of the ratio scale of the original objective" in {
    Relabelling.CounterfactualWeights shouldBe TradeOff.BetaScale.Ratio.counterfactualWeights
  }

  "Relabelling on the linear scale" should "relabel for its weights and condition the states with its feature" in {
    val linear = TradeOff.BetaScale.Linear
    val steps = Relabelling.relabel(trajectory, ownWeight, gamma, encode, betaIndex, scale = linear)
    steps.take(weightsPerStep).map(_.beta) shouldBe Relabelling.targetWeights(ownWeight, linear.counterfactualWeights)
    steps.foreach { transition =>
      transition.state(betaIndex) shouldBe linear.feature(transition.beta)
      transition.nextState(betaIndex) shouldBe linear.feature(transition.beta)
    }
  }

  it should "link each step to the next one, and the last one to itself as a terminal transition" in {
    val firstOfEachStep = relabelled.grouped(weightsPerStep).map(_.head).toSeq
    firstOfEachStep.map(transition => (transition.state(0), transition.nextState(0), transition.isTerminal)) shouldBe
      Seq((0.0f, 1.0f, false), (1.0f, 7.0f, false), (7.0f, 7.0f, true))
  }

  it should "discount by gamma to the power of the wake-up period, never shorter than the minimum" in {
    val discounts = relabelled.grouped(weightsPerStep).map(_.head.discount).toSeq
    discounts shouldBe Seq(0.9f, math.pow(gamma, 3.0).toFloat, math.pow(gamma, Relabelling.MinDiscountPeriod).toFloat)
  }

  it should "attach the instability intensity of the step" in {
    relabelled.grouped(weightsPerStep).map(_.head.intensity).toSeq shouldBe
      trajectory.map(step => InstabilityIntensity.of(step.state))
  }

  "Mobile relabelling" should "share the features of a step across the target weights and condition them on demand" in {
    val costs = Map(0.0 -> 0.3, 5.0 -> 0.1)
    val outcome = MobileOutcome(2.0, costs, stableBackbone, terminal = false)
    val step = Step(sourceMovingTowardsMe, actionIndex = -1, action = 2.0, reward = 0.0, Some(outcome))
    val mobile = Relabelling.relabel(Seq(step), ownWeight, gamma, encode, betaIndex, mobile = true)
    mobile should have size (costs.size * weightsPerStep).toLong
    mobile.groupBy(_.errorIntegral).values.foreach { copies =>
      copies.map(_.state).distinct should have size 1
      copies.map(_.nextState).distinct should have size 1
    }
    mobile.foreach { copy =>
      copy.conditionedState.toSeq shouldBe Seq(sourceMovingTowardsMe.me.toFloat, copy.beta.toFloat)
      copy.conditionedNextState.toSeq shouldBe Seq(stableBackbone.me.toFloat, copy.beta.toFloat)
    }
  }

  it should "condition the features on the β feature of the learner's encoding" in {
    val outcome = MobileOutcome(2.0, Map(0.0 -> 0.3), stableBackbone, terminal = false)
    val step = Step(sourceMovingTowardsMe, actionIndex = -1, action = 2.0, reward = 0.0, Some(outcome))
    val scale = TradeOff.BetaScale.FullRange
    val mobile =
      Relabelling.relabel(Seq(step), ownWeight, gamma, encode, betaIndex, mobile = true, scale, scale.feature)
    mobile.map(_.beta).distinct shouldBe (ownWeight +: scale.counterfactualWeights).distinct
    mobile.foreach(copy => copy.conditionedState(betaIndex) shouldBe scale.feature(copy.beta))
  }
end RelabellingSpec
