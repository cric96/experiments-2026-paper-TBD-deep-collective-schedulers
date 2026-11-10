package it.unibo.mobility

/** The error a node's schedule is charged for in the whole network: the errors of the nodes it reaches, discounted by a
  * factor ρ at every hop, C_i = Σ_j ρ^hop(i,j) e_j. A node that sleeps while the field reaches it delays every node
  * downstream, so the charge is a sum, not a mean as in [[NeighbourhoodError]]: ρ = 1 charges every node of its
  * component alike (a team reward), ρ = 0 only the node itself; in between the charge fades with the distance, as in
  * the critics truncated to a few hops of networked multi-agent learning (Qu et al. 2020) and in the spatial discount
  * of MA2C (Chu et al. 2020).
  */
object NetworkError:

  /** The weight of a node `hops` away, ρ^hops: one for the node itself, also when ρ is zero. */
  def weight(discount: Double, hops: Int): Double = math.pow(discount, hops)

  /** Σ_j weights(j) · errors(j), each error counted as at most `cap` (an unbounded or undefined one as `cap`): with the
    * loss scale as `cap` it is the weighted sum of the losses of the nodes, times the scale. It runs for every node at
    * every simulator event, hence the index loop.
    */
  def of(errors: Array[Double], weights: Array[Double], cap: Double): Double =
    var sum = 0.0
    var node = 0
    while node < errors.length do
      sum += weights(node) * (if errors(node) < cap then errors(node) else cap)
      node += 1
    sum
end NetworkError
