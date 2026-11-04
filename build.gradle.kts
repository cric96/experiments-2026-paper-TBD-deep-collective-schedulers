import org.gradle.api.tasks.testing.logging.TestExceptionFormat

plugins {
    application
    alias(libs.plugins.multiJvmTesting) // Pre-configures the Java toolchains
    alias(libs.plugins.taskTree) // Helps debugging dependencies among gradle tasks
    alias(libs.plugins.spotless)
    scala
}

repositories {
    mavenCentral()
}

dependencies {
    // Check the catalog at gradle/libs.versions.toml
    implementation(libs.bundles.alchemist)
    implementation(libs.bundles.djl)
    implementation(libs.scala3.library)
    testImplementation(libs.scalatest)
    testRuntimeOnly(libs.scalatest.junit.runner)
    testRuntimeOnly(libs.junit.platform.launcher)
}

tasks.test {
    useJUnitPlatform {
        includeEngines("scalatest")
        if (System.getenv("CI") == "true") {
            // Exact values of the PyTorch CPU kernels are pinned on the development machine only.
            excludeTags("it.unibo.testing.PyTorchNumerics")
        }
    }
    // Multi-threaded (and dynamically-threaded) BLAS kernels change the order of floating-point reductions,
    // which makes training, and hence the golden masters, non-reproducible: pin PyTorch to a single thread.
    systemProperty("ai.djl.pytorch.num_threads", "1")
    systemProperty("ai.djl.pytorch.num_interop_threads", "1")
    environment("OMP_NUM_THREADS", "1")
    environment("MKL_NUM_THREADS", "1")
    // A transitive dependency registers an obsolete SAX parser that cannot read logback-test.xml: use the JDK one.
    systemProperty(
        "javax.xml.parsers.SAXParserFactory",
        "com.sun.org.apache.xerces.internal.jaxp.SAXParserFactoryImpl",
    )
    testLogging {
        events("failed")
        exceptionFormat = TestExceptionFormat.FULL
    }
}

spotless {
    scala {
        scalafmt(libs.versions.scalafmt.get()).configFile(".scalafmt.conf")
    }
}

multiJvm {
    jvmVersionForCompilation.set(17)
    testByDefaultWith(17)
}

val batch: String by project
val variables: String by project

val alchemistGroup = "Run Alchemist"
/*
 * This task is used to run all experiments in sequence
 */
val runAll by tasks.register<DefaultTask>("runAll") {
    group = alchemistGroup
    description = "Launches all simulations"
}
/*
 * Scan the folder with the simulation files, and create a task for each one of them.
 */
/*
 * The variables of `variables` (gradle.properties) that a simulation file declares, i.e. those it can sweep.
 */
fun sweptVariablesOf(simulationFile: File): List<String> {
    val yaml = simulationFile.readText()
    return variables.split(",")
        .map(String::trim)
        .filter { it.isNotEmpty() && Regex("""(?m)^  ${Regex.escape(it)}\s*:""").containsMatchIn(yaml) }
}

/*
 * Scan the folder with the simulation files, and create a task for each one of them.
 */
File(rootProject.rootDir.path + "/src/main/yaml").listFiles()
    ?.filter { it.extension == "yml" } // pick all yml files in src/main/yaml
    ?.sortedBy { it.nameWithoutExtension } // sort them, we like reproducibility
    ?.forEach { simulationFile ->
        // one simulation file -> one gradle task
        val name = simulationFile.nameWithoutExtension
        val task by tasks.register<JavaExec>("run${name.replaceFirstChar { c -> c.uppercase() }}") {
            group = alchemistGroup // This is for better organization when running ./gradlew tasks
            description = "Launches simulation $name" // Just documentation
            mainClass.set("it.unibo.alchemist.Alchemist") // The class to launch
            classpath = sourceSets["main"].runtimeClasspath // The classpath to use
            javaLauncher.set(
                javaToolchains.launcherFor {
                    languageVersion.set(JavaLanguageVersion.of(17))
                },
            )
            args("run", simulationFile.absolutePath)
            if (name == "simulationGeneralization") {
                providers.gradleProperty("targetProgram").orNull?.toInt()?.let { program ->
                    require(program in 0..4) { "targetProgram must be a program code from 0 to 4" }
                    args(
                        "--override",
                        "variables: {progId: {type: ArbitraryVariable, parameters: [$program, [$program]]}}",
                    )
                }
            }
            if (name == "simulationMobility") {
                systemProperty("ai.djl.pytorch.num_threads", "1")
                systemProperty("ai.djl.pytorch.num_interop_threads", "1")
                environment("OMP_NUM_THREADS", "1")
                environment("MKL_NUM_THREADS", "1")
                val overrides = mapOf(
                    "mobilityDiffusion" to "diffusion", "mobilityTolerance" to "errorTolerance",
                    "mobilityAlgorithm" to "algoId", "mobilitySources" to "sourceChanges",
                    "mobilityTrainingEpisodes" to "trainingEpisodes", "mobilityEpisodes" to "episodes",
                    "mobilityEpisodeLength" to "episodeLength", "mobilitySteps" to "stepsPerEpisode",
                    "mobilityInterval" to "movementInterval",
                )
                overrides.forEach { (property, variable) ->
                    providers.gradleProperty(property).orNull?.let { raw ->
                        val value = raw.toDouble()
                        require(value.isFinite() && value >= 0) { "$property must be finite and non-negative" }
                        args("--override", "variables: {$variable: {type: ArbitraryVariable, parameters: [$value, [$value]]}}")
                    }
                }
            }
            if (System.getenv("CI") == "true" || batch == "true") {
                // Headless: run the cartesian product of the swept variables (the default values of the others).
                val swept = sweptVariablesOf(simulationFile).joinToString(", ")
                args("--override", "launcher: {type: DefaultLauncher, parameters: {batch: [$swept]}}")
            } else {
                // A graphics environment should be available: open the GUI with the effects named after the file.
                val effects = File(rootProject.rootDir, "effects/$name.json")
                val graphics = if (effects.exists()) ", parameters: {graphics: ${effects.absolutePath}}" else ""
                args("--override", "monitors: [{type: SwingGUI$graphics}]")
            }
        }
        runAll.dependsOn(task)
    }

tasks.withType<Tar>().configureEach {
    duplicatesStrategy = DuplicatesStrategy.WARN
}
tasks.withType<Zip>().configureEach {
    duplicatesStrategy = DuplicatesStrategy.WARN
}
// Portable runtime description for the isolated campaign runner; no old build-script dependency.
tasks.register("campaignRuntime") {
    dependsOn("classes")
    doLast {
        val launcher = javaToolchains.launcherFor {
            languageVersion.set(JavaLanguageVersion.of(17))
        }.get()
        val runtime = mapOf(
            "java" to launcher.executablePath.asFile.absolutePath,
            "classpath" to sourceSets["main"].runtimeClasspath.asPath,
        )
        layout.buildDirectory.file("campaign-runtime.json").get().asFile
            .writeText(groovy.json.JsonOutput.toJson(runtime))
    }
}
