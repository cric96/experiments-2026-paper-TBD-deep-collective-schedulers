package it.unibo.mobility

/** What changes halfway through an episode of the mobile experiment. */
enum EpisodeEvent:
  /** The source moves to a far node, while every node moves at the level of the episode all the time. */
  case SourceSwitch

  /** The source stays; nothing moves until halfway, then the nodes of a region ([[MovingRegion]]) start moving at the
    * level of the episode.
    */
  case RegionStartsMoving

  /** The source moves to a far node, while only the nodes of a region ([[MovingRegion]], chosen when the episode
    * starts) move at the level of the episode all the time; the others stay still.
    */
  case SourceSwitchWhileRegionMoves

  /** A composite training episode: nothing moves while the field spreads from the first source; halfway the source
    * moves to a far node, and after a delay every node moves at the level of the episode for a while, then the network
    * is still again ([[MobilityConfig.motionWindowAt]]).
    */
  case SourceSwitchThenMotion

/** The seconds from the start of an episode in which a halfway event moves the nodes: from `start` to `end`, excluded. */
final case class MotionWindow(start: Double, end: Double):
  require(start >= 0.0 && end >= start)
