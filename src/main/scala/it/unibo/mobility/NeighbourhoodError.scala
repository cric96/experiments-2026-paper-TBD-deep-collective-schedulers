package it.unibo.mobility

/** The error a node's schedule is charged for in its neighbourhood: the mean of its own error and of its neighbours'
  * ones. With the message timing of Alchemist 25 a node sends the value it computed only at its next round, so a node
  * that already holds a correct value still pays while its neighbours lack it: its own error alone does not reward
  * relaying the field.
  */
object NeighbourhoodError:

  /** The mean of `own` and of the `neighbours`' errors, each counted as at most `cap` (an unbounded error as `cap`):
    * with the loss scale as `cap` and no tolerance, its loss is the mean of the losses of the neighbourhood.
    */
  def of(own: Double, neighbours: Iterable[Double], cap: Double): Double =
    val errors = (own +: neighbours.toSeq).map(math.min(_, cap))
    errors.sum / errors.size
