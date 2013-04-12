package emokids

/**
 * This is a helper file for the homework. You do not need to modify anything in it
 * to complete the assignment. In fact, don't change anything in it. If you find
 * something that you think is a problem, contact the instructor to make a bug report.
 */

//import opennlp.scalabha.util.CollectionUtils._


/**
 * An object that sets up the configuration for command-line options using
 * Scallop and returns the options, ready for use.
 */
object TwitterSentimentOpts {

  import org.rogach.scallop._
  
  def apply(args: Array[String]) = new ScallopConf(args) {
    banner("""
Classification application.

For usage see below:
	     """)

    val methodTypes = Set("lexicon","majority","maxent")
    val method = opt[String]("method", default=Some("maxent"), validate = methodTypes, descr="The type of solver to use. Possible values: " + methodTypes.toSeq.sorted.mkString(",") )

    val cost = opt[Double]("cost", default=Some(1.0), validate = (0<), descr="The cost parameter C. Bigger values means less regularization (more fidelity to the training set). Note: if you are using the GIS solver, this option instead indicates the standard deviation of the Gaussian penalty (bigger values still mean less regularization).")

    val trainfile = opt[String]("train", required=true,descr="The file containing training events.")

    val evalfile = opt[String]("eval", descr="The file containing evalualation events.")

    val extended = opt[Boolean]("extended", noshort = true, descr="Use extended features.")

    val help = opt[Boolean]("help", noshort = true, descr="Show this message")

    val verbose = opt[Boolean]("verbose")
  }
}


/**
 * A standalone application that is the entry point for running sentiment analysis
 * experiments on the Twitter datasets in data/classify.
 */
object TwitterSentiment {

  import java.io.File

  def main(args: Array[String]) {

    val opts = TwitterSentimentOpts(args)

    lazy val trainSource = new File(opts.trainfile())
    lazy val evalSource = new File(opts.evalfile())
    lazy val (trainingLabels, _, trainingTweets) = DatasetReader(trainSource).unzip3
    lazy val (evalLabels, _, evalTweets) = DatasetReader(evalSource).unzip3

    lazy val featureExtractor = 
      if (opts.extended()) ExtendedFeatureExtractor
      else DefaultFeatureExtractor

    val classifier = opts.method() match {
      case "majority" =>
        val (majorityLabel, majorityProb) =
          MajorityClassCalculator(trainingLabels)
        new MajorityClassBaseline(majorityLabel, majorityProb)

      case "lexicon" => new LexiconRatioClassifier

      case "maxent" =>
        val trainer = new MaxentClassifierTrainer(featureExtractor)
        trainer(trainingLabels, trainingTweets, opts.cost())

      case _ => throw new MatchError("Invalid method: " + opts.method())
    }

    val tweetTexts = evalTweets.map(_.content)
    val (predictions, confidence) = tweetTexts.map(classifier).unzip

    val numCorrect = evalLabels.zip(predictions).count {
      case (gold, predicted) => gold == predicted
    }

    println(nak.util.ConfusionMatrix(evalLabels, predictions, tweetTexts))

  }

}


/**
 * An object that implements a function for calculating the majority
 * class label given a sequence of labels, and its probability in that
 * sequence. E.g. for the labels "yes, yes, no, yes" it should
 * return (yes, .75).
 */
object MajorityClassCalculator {
  import nak.util.CollectionUtil._

  def apply(labels: Seq[String]) = {
    val (majorityLabel, majorityProb) =
      labels
        .counts
        .mapValues(_.toDouble / labels.length)
        .maxBy(_._2)

    println(" ++ " + labels.counts.mapValues(_.toDouble/labels.length).mkString(" "))
    println(" * " + majorityLabel)
    (majorityLabel, majorityProb)
  }
}

/**
 * An object that implements a function for splitting a string into a
 * sequence of tokens.
 */
object Tokenizer {
  def apply(text: String) = chalk.lang.eng.Twokenize(text)
  //def apply(text: String) = text.split("\\s+")
}

/**
 * A class that simply implements the abstract methods in
 * AbstractLexiconRatioClassifier (see ClassifierUtil.scala).
 */
class LexiconRatioClassifier extends AbstractLexiconRatioClassifier {
  import Polarity._

  // Return the number of positive tokens in the token sequence.
  def numPositiveTokens(tokens: Seq[String]): Int = tokens.count(positive)

  // Return the number of negative tokens in the token sequence.
  def numNegativeTokens(tokens: Seq[String]): Int = tokens.count(negative)

}

/**
 * An implementation of a FeatureExtractor that extracts more information out
 * of a tweet than the DefaultFeatureExtractor defined in ClassifierUtil.scala.
 * This is the main part of the assignment.
 */
object ExtendedFeatureExtractor extends FeatureExtractor {

  // Import any classes and objects you need here. AttrVal is included already.
  import nak.core.AttrVal
  import scala.util.matching.Regex
  import Polarity._
  import English.stopwords

  // Define any fields, including regular expressions and helper objects, here.
  // For example, you may want to include the lexicon ration classifier here (hint),
  // and a Porter stemmer, and whatever else you think might help.

  // End of sentence marker
  private val EOS = "[-*-EOS-*-]"

  val lexClassifier = new LexiconRatioClassifier
  val stemmer = new chalk.lang.eng.PorterStemmer
  val Threepeat = """\w+(.)\1\1+\w+""".r
  val AllCaps = """[^\w]*[A-Z][A-Z]+[^\w]*""".r

  // A class to allow an implicit conversion for easy regex handling.
  class Matcher(regex: Regex) {
    def fullMatch(input: String) = regex.pattern.matcher(input).matches
    def hasMatch(input: String) = regex.findAllIn(input).length > 1
  }

  // The implicit conversion of Regex to Matcher.
  implicit def regexToMatcher(regex: Regex) = new Matcher(regex)

  def apply(content: String) = {
    val tokens = Tokenizer(content).toSeq
    val contentTokens = tokens.filter(stopwords)
    val stems = tokens.map(stemmer(_))

    val unigrams =
      tokens
        .filterNot(stopwords)
        .map(token => AttrVal("unigram", token))
        .toSeq

    val polarityFeatures =
      tokens
        .map(_.replaceAll("#", ""))
        .flatMap { token =>
          if (positive(token)) Some(AttrVal("polarity", "POSITIVE"))
          else if (negative(token)) Some(AttrVal("polarity", "NEGATIVE"))
          else None
        }

    val bigrams = (Seq(EOS) ++ stems ++ Seq(EOS)).sliding(2).flatMap {
      case List(first, second) =>
        Some(AttrVal("bigram", first + "::" + second))
      case _ => None
    }

    val emphasis = tokens.flatMap { token =>
      if (Threepeat.hasMatch(token)) Some("[-THREEPEAT-]")
      else if (AllCaps.fullMatch(token)) Some("[-ALLCAPS-]")
      else if (token.endsWith("!")) Some("[-EXCLAMATION-]")
      else None
    }
    
    val emphasisFeatures = emphasis.map(AttrVal("emphasis",_))

    //val trigrams = (Seq(EOS, EOS) ++ stems ++ Seq(EOS, EOS)).sliding(3).flatMap {
    //  case List(first, second, third) =>
    //    Some(AttrVal("trigram", first + "::" + second + "::" + third))
    //  case _ => None
    //}

    val subjectiveTokens =
      contentTokens.flatMap { token =>
        if (positive(token)) Some("[-POSITIVE-]")
        else if (negative(token)) Some("[-NEGATIVE-]")
        else Some("[-NEUTRAL-]")
      }

    val bigramPolarity = (Seq(EOS) ++ subjectiveTokens ++ Seq(EOS)).sliding(2).flatMap {
      case List(first, second) =>
        Some(AttrVal("bigramPolarity", first + "::" + second))
      case _ => None
    }.toSeq

    (unigrams
      ++ Seq(AttrVal("lexratio", lexClassifier(content)._1))
      ++ bigrams
      ++ bigramPolarity
      ++ emphasisFeatures
     //++ trigrams
      ++ polarityFeatures)
  }
}
