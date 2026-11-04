package it.unibo.alchemist

import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.{Molecule, Node}

import java.util.concurrent.ConcurrentHashMap

/** Typed access to the molecules of a node: the cast to the expected type happens here, at the Alchemist boundary.
  *
  * The mobile experiment reads the molecules of every node at every simulator event. A molecule hashes its name the
  * first time it is looked up, so the molecules are shared rather than created per read, and a read is a single lookup
  * in the node's current contents.
  */
object Molecules:
  private val byName: ConcurrentHashMap[String, Molecule] = ConcurrentHashMap()

  /** The shared molecule called `name`. */
  def molecule(name: String): Molecule = byName.computeIfAbsent(name, SimpleMolecule(_))

  extension [T](node: Node[T])
    def valueOf[A](name: String): Option[A] =
      Option(node.getContents.getCurrent.get(molecule(name))).map(_.asInstanceOf[A])
