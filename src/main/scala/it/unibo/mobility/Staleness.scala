package it.unibo.mobility

/** How far the gradient a node holds is from the one it would compute if it ran a round now, from the values its
  * current neighbours hold at their current distances: the part of the error that the node's own schedule removes.
  * Messages that ScaFi would still retain from nodes out of range are ignored.
  */
object Staleness:
  /** The gradient of a source. */
  val SourceGradient: Double = 0.0

  /** `neighbours` are (held value, current distance) pairs; unbounded when only one of the two values is defined. */
  def of(held: Double, isSource: Boolean, neighbours: Iterable[(Double, Double)]): Double =
    val fresh =
      if isSource then SourceGradient
      else neighbours.map((value, distance) => value + distance).minOption.getOrElse(Double.PositiveInfinity)
    if held.isFinite && fresh.isFinite then math.abs(held - fresh)
    else if held.isFinite == fresh.isFinite then 0.0
    else Double.PositiveInfinity
