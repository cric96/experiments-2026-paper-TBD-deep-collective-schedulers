package it.unibo.learning.agents.offline

import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

import scala.util.Random

class StratifiedTransitionsSpec extends AnyFlatSpec with Matchers:
  private val unstable = Seq("u1", "u2", "u3")
  private val stable = Seq("s1", "s2")

  private def transitionsWith(unstable: Seq[String], stable: Seq[String]): StratifiedTransitions[String] =
    val transitions = StratifiedTransitions[String]()
    unstable.foreach(transitions.add(_, isUnstable = true))
    stable.foreach(transitions.add(_, isUnstable = false))
    transitions

  "StratifiedTransitions" should "count the transitions of both regimes" in {
    transitionsWith(unstable, stable).size shouldBe 5
    transitionsWith(Seq.empty, Seq.empty).isEmpty shouldBe true
  }

  it should "sample the first half of a batch from the unstable regime and the rest from the stable one" in {
    val batch = transitionsWith(unstable, stable).sample(5, new Random(1))
    batch.take(2).foreach(unstable should contain(_))
    batch.drop(2).foreach(stable should contain(_))
  }

  it should "sample everything from the only regime available" in {
    transitionsWith(unstable, Seq.empty).sample(4, new Random(1)).foreach(unstable should contain(_))
    transitionsWith(Seq.empty, stable).sample(4, new Random(1)).foreach(stable should contain(_))
  }

  it should "be reproducible for a given seed" in {
    val transitions = transitionsWith(unstable, stable)
    transitions.sample(8, new Random(3)) shouldBe transitions.sample(8, new Random(3))
  }

  it should "draw indices in the order unstable first, then stable (reproducibility of the experiments)" in {
    val random = new Random(5)
    val expectedUnstable = Vector.fill(2)(unstable(random.nextInt(unstable.size)))
    val expectedStable = Vector.fill(3)(stable(random.nextInt(stable.size)))
    transitionsWith(unstable, stable).sample(5, new Random(5)) shouldBe expectedUnstable ++ expectedStable
  }

  it should "sample a single regime, or nothing when it is empty" in {
    transitionsWith(unstable, stable)
      .sampleRegime(6, new Random(1), unstable = true)
      .foreach(unstable should contain(_))
    transitionsWith(unstable, stable).sampleRegime(6, new Random(1), unstable = false).foreach(stable should contain(_))
    transitionsWith(unstable, Seq.empty).sampleRegime(6, new Random(1), unstable = false) shouldBe empty
  }

  it should "expose the oldest transitions of each regime" in {
    transitionsWith(unstable, stable).oldest(2) shouldBe Vector("u1", "u2", "s1", "s2")
  }

  it should "bound each regime and retain the most recent transitions in order" in {
    val buffer = StratifiedTransitions[Int](capacityPerRegime = 2)
    (1 to 5).foreach(value => buffer.add(value, isUnstable = true))
    buffer.add(10, isUnstable = false)
    buffer.size shouldBe 3
    buffer.oldest(2) shouldBe Vector(4, 5, 10)
    buffer.sample(50, new scala.util.Random(1)).toSet shouldBe Set(4, 5, 10)
  }
  it should "retain rare scenarios and balance their batches" in {
    val buffer = StratifiedTransitions[Int](capacityPerRegime = 4)
    buffer.scenario(0)
    buffer.add(0, isUnstable = false)
    buffer.scenario(1)
    (1 to 100).foreach(_ => buffer.add(1, isUnstable = true))
    val batch = buffer.sample(100, new scala.util.Random(1))
    batch.count(_ == 0) shouldBe 50
    batch.count(_ == 1) shouldBe 50
    buffer.size shouldBe 5
  }
end StratifiedTransitionsSpec
