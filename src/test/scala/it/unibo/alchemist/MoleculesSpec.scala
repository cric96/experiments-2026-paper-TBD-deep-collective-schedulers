package it.unibo.alchemist

import it.unibo.alchemist.Molecules.*
import it.unibo.alchemist.model.Position
import it.unibo.alchemist.model.environments.EmptyEnvironment
import it.unibo.alchemist.model.incarnations.ScafiIncarnation
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.nodes.GenericNode
import it.unibo.alchemist.model.scafi.nodes.SimpleNodeManager
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class MoleculesSpec extends AnyFlatSpec with Matchers:
  private def node[P <: Position[P]]: GenericNode[Any] =
    val created = GenericNode[Any](EmptyEnvironment[Any, P](ScafiIncarnation[Any, P](), 2))
    created.set(SimpleMolecule("source"), true)
    created.set(SimpleMolecule("output"), 12.5)
    created

  "The typed molecule access" should "read what the node manager of the incarnation reads" in {
    val sample = node
    val manager = SimpleNodeManager[Any](sample)
    Seq("source", "output", "missing").foreach { name =>
      sample.valueOf[Any](name) shouldBe manager.getOption[Any](name)
    }
    sample.valueOf[Boolean]("source") shouldBe Some(true)
    sample.valueOf[Double]("missing") shouldBe None
  }

  it should "see the updates of the node" in {
    val sample = node
    sample.valueOf[Double]("output") shouldBe Some(12.5)
    sample.set(SimpleMolecule("output"), 3.0)
    sample.valueOf[Double]("output") shouldBe Some(3.0)
    sample.removeAllConcentrationOf(SimpleMolecule("output"))
    sample.valueOf[Double]("output") shouldBe None
  }
end MoleculesSpec
