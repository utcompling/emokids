package emokids

/**
 * An object that sets up the configuration for command-line options using
 * Scallop and returns the options, ready for use.
 */
object PolarityExperimentOpts {

  import org.rogach.scallop._
  
  def apply(args: Array[String]) = new ScallopConf(args) {
    banner("""
Classification application.

For usage see below:
	     """)
    val method = opt[String]("method", default=Some("L2R_L2"), descr="The type of solver to use. Possible values: majorit, lexicon, or any liblinear solver type.")

    val cost = opt[Double]("cost", default=Some(1.0), validate = (0<), descr="The cost parameter C. Bigger values means less regularization (more fidelity to the training set).")

    val trainfile = opt[String]("train", descr="The file containing training events.")

    val evalfile = opt[String]("eval", descr="The file containing evalualation events.")

    val extended = opt[Boolean]("extended", short = 'x', descr="Use extended features.")

    val help = opt[Boolean]("help", noshort = true, descr="Show this message")

    val verbose = opt[Boolean]("verbose")
  }
}


/**
 * A standalone application that is the entry point for running sentiment analysis
 * experiments on the Twitter datasets in data/classify.
 */
object PolarityExperiment {

  import java.io.File
  import nak.data.BowFeaturizer

  def main(args: Array[String]) {

    val opts = PolarityExperimentOpts(args)

    lazy val trainSource = new File(opts.trainfile())
    lazy val evalSource = new File(opts.evalfile())
    lazy val (trainingLabels, _, trainingTweets) = DatasetReader(trainSource).unzip3
    lazy val (evalLabels, _, evalTweets) = DatasetReader(evalSource).unzip3

    lazy val featurizer = 
      if (opts.extended()) ExtendedFeaturizer else BasicFeaturizer

    val classifier = opts.method() match {
      case "majority" => MajorityClassBaseline(trainingLabels)

      case "lexicon" => new LexiconRatioClassifier

      case solverDescription =>
        val solver = nak.liblinear.Solver(solverDescription)
        val config = new nak.liblinear.LiblinearConfig(solverType=solver,cost=opts.cost())
        NakClassifierTrainer(config, featurizer, trainingLabels, trainingTweets)

    }

    val tweetTexts = evalTweets.map(_.content)
    val (predictions, confidence) = evalTweets.map(classifier).unzip

    println(nak.util.ConfusionMatrix(evalLabels, predictions, tweetTexts))

  }

}


