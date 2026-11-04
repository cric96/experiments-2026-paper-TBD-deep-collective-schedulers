package it.unibo

import it.unibo.testing.{PyTorchNumerics, Simulations}
import it.unibo.testing.Simulations.Fingerprint
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

/** Pins the observable behaviour of every simulation file, so that refactorings can be proven behaviour-preserving.
  *
  * If a change is *meant* to alter the dynamics, update the expected fingerprints and say so in the commit message.
  */
class SimulationGoldenMasterSpec extends AnyFlatSpec with Matchers:

  /** End of the first episode, end of the second one (after one training), shortly after the second training. */
  private val checkpoints = Seq(199.5, 399.5, 450.0)
  private val seededWeight = Map("random" -> 2.0, "weight" -> 0.95)

  private val scenarios = Seq(
    (
      "simulation",
      seededWeight,
      Seq(
        Fingerprint(199.5, 100, 6702, 345.0, -616.3068116325582, 2850.897562215816, 29134.200526238164),
        Fingerprint(399.5, 100, 6852, 319.0, -587.2089032185885, 2160.97122011658, 27593.35841691538),
        Fingerprint(450.0, 100, 1821, 322.0, -129.51251801273494, 1023.405956911711, 21769.281388504758)
      )
    ),
    (
      "simulationBaseline",
      seededWeight + ("actions" -> 2.0),
      Seq(
        Fingerprint(199.5, 100, 10001, 200.0, -365.9833335434975, 1897.2229074453408, 29134.200526238164),
        Fingerprint(399.5, 100, 9950, 200.0, -376.0093218882406, 1654.332298605182, 27593.35841691538),
        Fingerprint(450.0, 100, 2500, 200.0, -101.18650314612303, 892.4721343753803, 21769.281388504758)
      )
    ),
    (
      "simulationDeepOffline",
      seededWeight,
      Seq(
        Fingerprint(199.5, 100, 6753, 377.0, -564.1635430875642, 3001.3044409385348, 29134.200526238164),
        Fingerprint(399.5, 100, 6753, 345.0, -663.1651526380653, 2302.4978141441925, 27593.35841691538),
        Fingerprint(450.0, 100, 1715, 335.0, -180.36420018633416, 1175.8632422327107, 21769.281388504758)
      )
    ),
    (
      "simulationDeepOfflineContinous",
      seededWeight,
      Seq(
        Fingerprint(199.5, 100, 7334, 302.55599896339515, -551.5776609338693, 2430.5891261503084, 29134.200526238164),
        Fingerprint(399.5, 100, 7405, 313.0471575669216, -582.0456965988096, 2089.955567697372, 27593.35841691538),
        Fingerprint(450.0, 100, 1860, 320.255405422509, -166.52846722601652, 1332.7362837883663, 21769.281388504758)
      )
    ),
    (
      "simulationDeepOfflineGNN",
      seededWeight,
      Seq(
        Fingerprint(199.5, 100, 7333, 334.85434673253013, -557.0964801188771, 2096.5005365736124, 29134.200526238164),
        Fingerprint(399.5, 100, 7665, 334.0414796701127, -564.4709762613563, 1816.9883779290249, 27593.35841691538),
        Fingerprint(450.0, 100, 1913, 312.7083369355091, -165.81193665580543, 949.8876551810008, 21769.281388504758)
      )
    ),
    (
      "simulationGeneralization", // algoId 3 = HeuristicAdaptiveAgent, progId 0 = gradient
      Map("random" -> 2.0, "algoId" -> 3, "progId" -> 0),
      Seq(
        Fingerprint(199.5, 100, 7201, 286.91588785046696, -289.42069509572775, 2015.0455725243496, 29134.200526238164),
        Fingerprint(399.5, 100, 7270, 286.91588785046696, -301.775676934332, 1795.81110406843, 27593.35841691538),
        Fingerprint(450.0, 100, 1804, 286.91588785046696, -92.3662630542563, 878.540323356459, 21769.281388504758)
      )
    )
  )

  /** The scenarios whose dynamics depend on networks trained with PyTorch. */
  private val neuralScenarios =
    Set("simulationDeepOffline", "simulationDeepOfflineContinous", "simulationDeepOfflineGNN")

  scenarios.foreach { case (yaml, variables, expected) =>
    val behaviour = yaml should "keep its observable dynamics"
    def check(): Unit = Simulations.run(yaml, variables, checkpoints) shouldBe expected
    if neuralScenarios.contains(yaml) then behaviour taggedAs PyTorchNumerics in check() else behaviour in check()
  }

  /** After the training episodes, the generalization scenario switches to its target program and cycles β. */
  private val afterSwitch = Seq(1999.5, 2199.5, 2250.0)
  private val endOfTraining =
    Fingerprint(1999.5, 100, 7227, 286.91588785046696, -292.28910450459847, 1762.998944695169, 23571.475879998467)

  private val unseenPrograms = Seq(
    (
      1,
      "hop gradient",
      Seq(
        endOfTraining,
        Fingerprint(2199.5, 100, 4603, 477.35849056603695, -775.9296971363633, 2566.093891509395, 575.0),
        Fingerprint(2250.0, 100, 1100, 477.35849056603695, -196.7892630206408, 1656.4339622641241, 401.0)
      )
    ),
    (
      2,
      "broadcast",
      Seq(
        endOfTraining,
        Fingerprint(2199.5, 100, 4310, 460.88985849056536, -558.4745596243488, 8433.778867924399, 1110760.0),
        Fingerprint(2250.0, 100, 1100, 449.67665094339554, -173.71405631770813, 3002.584905660329, 962920.0)
      )
    ),
    (
      3,
      "collect",
      Seq(
        endOfTraining,
        Fingerprint(2199.5, 100, 4866, 438.87995283018796, -1137.1204515065567, 1433.7097818396087, 660.0),
        Fingerprint(2250.0, 100, 1131, 425.42900943396165, -135.26492002871595, 345.89521816037234, 291.0)
      )
    )
  )

  unseenPrograms.foreach { case (program, name, expected) =>
    "simulationGeneralization" should s"keep its dynamics on an unseen program ($name)" in {
      val variables = Map("random" -> 2.0, "algoId" -> 3, "progId" -> program)
      Simulations.run("simulationGeneralization", variables, afterSwitch) shouldBe expected
    }
  }
end SimulationGoldenMasterSpec
