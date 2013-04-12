package emokids

/**
 * This is a helper file for the homework. You do not need to modify anything in it
 * to complete the assignment. In fact, don't change anything in it. If you find
 * something that you think is a problem, contact the instructor to make a bug report.
 */

/**
 * A simple case class to store information associated with a Tweet.
 */
case class Tweet(val tweetid: String, val username: String, val content: String)

/**
 * Read in a polarity labeled dataset from XML.
 */
object DatasetReader {

  import scala.xml._
  import java.io.File

  // Allow NodeSeqs to implicitly convert to Strings when needed.
  implicit def nodeSeqToString(ns: NodeSeq) = ns.text

  def apply(file: File): Seq[(String, String, Tweet)] = {
    val itemsXml = XML.loadFile(file)

    (itemsXml \ "item").flatMap { itemNode =>
      val label: String = itemNode \ "@label"

      // We only want the positive, negative and neutral items.
      label match {
        case "negative" | "positive" | "neutral" =>
          Some(label,
            (itemNode \ "@target").text,
            Tweet(itemNode \ "@tweetid", itemNode \ "@username", itemNode.text.trim))

        case _ => None
      }
    }
  }

}
