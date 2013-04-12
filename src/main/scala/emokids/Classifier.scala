package emokids

/**
 * This is a helper file for the homework. You do not need to modify anything in it
 * to complete the assignment. In fact, don't change anything in it. If you find
 * something that you think is a problem, contact the instructor to make a bug report.
 */

import nak.core.LinearModel
import nak.core.AttrVal

/**
 * An object that implements a function that extracts a sequence of features from
 * a string. Each feature is expressed as an AttrVal object, which is a simple case
 * class that holds an attribute (a String) and a value (also a String). For example,
 * you can create an AttrVal object like so:
 *   val foo = AttrVal("day","Sunday")
 * If you do foo.toString, you'll get "day=Sunday".
 *
 * See DefaultFeatureExtractor for a simple example feature extractor.
 */
trait FeatureExtractor extends (String => Seq[AttrVal])

/**
 * A feature extractor that tokenizes a String by whitespace, and then produces a
 * a feature (AttrVal object) with attribute "unigram" for each token.
 */
object DefaultFeatureExtractor extends FeatureExtractor {
  def apply(content: String) =
    content.split("\\s+").map(token => AttrVal("unigram", token)).toSeq
}

/**
 * A trait for text classification functions. The return value is a pair consisting of
 * the best label for the input and the confidence assigned to that label by the
 * classifier.
 */
trait TextClassifier extends (String => (String, Double))

/**
 * A text classifier that assigns a single label to every text that is given to it.
 */
class MajorityClassBaseline(majorityClass: String, prob: Double)
  extends TextClassifier {
  def apply(content: String) = (majorityClass, prob)
}

/**
 * A classifier that counts positive and negative terms in a text and picks a
 * label based on these counts. The label "neutral" is chosen when there are 
 * equal numbers of positive and negative tokens (or zero of both).
 */
abstract class AbstractLexiconRatioClassifier extends TextClassifier {
  import Polarity._

  def numPositiveTokens(tokens: Seq[String]): Int
  def numNegativeTokens(tokens: Seq[String]): Int

  def apply(content: String) = {

    val tokens = Tokenizer(content)
    val numPositive = numPositiveTokens(tokens)
    val numNegative = numNegativeTokens(tokens)

    // Add a small count to each so we don't get divide-by-zero error
    val positiveScore = numPositive + .1
    val negativeScore = numNegative + .1

    // Let neutral be preferred if nothing is found, and go with neutral
    // if pos and neg are the same.
    val neutralScore =
      if (numPositive == numNegative) tokens.length
      else .2

    // Calculate a denominator so we can pretend we have probabilities.
    val denominator = positiveScore + negativeScore + neutralScore

    // Create pseudo-probabilities based on the counts
    val predictions =
      List(("positive", positiveScore / denominator),
        ("negative", negativeScore / denominator),
        ("neutral", neutralScore / denominator))

    // Sort and return the top label and its confidence
    predictions.sortBy(_._2).last
  }
}

/**
 * An adaptor class that allows a maxent model trained via OpenNLP Maxent to be
 * used in a way that conforms with the TextClassifier trait defined above.
 */
class MaxentClassifier(model: LinearModel, extractor: FeatureExtractor)
  extends TextClassifier {
  val numOutcomes = model.getNumOutcomes
  val outcomes = (0 until numOutcomes).map(model.getOutcome(_))

  def apply(content: String) = {
    val prediction =
      model.eval(extractor(content).map(_.toString).toArray).toIndexedSeq
    val (prob, index) = prediction.zipWithIndex.maxBy(_._1)
    (outcomes(index), prob)
  }
}

/**
 * An adaptor that converts the tweets being classified into the event objects
 * needed to train an OpenNLP Maxent classifier, plus the ability to specify
 * the standard deviation (sigma) of the Gaussian prior, and the maximum number
 * of iterations. (Reasonable defaults for both are provided.)
 */
class MaxentClassifierTrainer(extractor: FeatureExtractor) {

  def apply(labels: Seq[String], tweets: Seq[Tweet], 
      sigma: Double = 1.0, maxIterations: Int = 100) = {
    val features = tweets.map(item => extractor(item.content).map(_.toString))
    val events = labels.zip(features).map {
      case (label, featuresForItem) =>
        new nak.core.Event(label, featuresForItem.toArray)
    }
    val eventStream = new IteratorEventStream(events.toIterator)
    //val model = nak.maxent.GIS.trainModel(eventStream, maxIterations, 1, sigma)
    val config = new nak.liblinear.LiblinearConfig(cost=sigma)
    val indexer = new nak.data.TwoPassDataIndexer(eventStream,0,false)
    val model = nak.liblinear.LiblinearTrainer.train(indexer, config)
    new MaxentClassifier(model, extractor)
  }

}

/**
 * A class that adapts a Scala Iterator of Event objects into an EventStream needed
 * by OpenNLP.
 */
class IteratorEventStream(events: Iterator[nak.core.Event])
  extends nak.data.AbstractEventStream {
  def next = events.next
  def hasNext = events.hasNext
}

