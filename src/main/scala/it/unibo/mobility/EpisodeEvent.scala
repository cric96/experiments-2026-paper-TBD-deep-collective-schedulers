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
