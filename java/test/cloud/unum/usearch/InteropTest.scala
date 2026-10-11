package cloud.unum.usearch

import scala.util.Using

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
// Aliased, since JUnit names its annotation `Test` and the Scala suite reads better with it renamed.
import org.junit.jupiter.api.{Test => JUnitTest}

import cloud.unum.usearch.DenseIndex.DenseMetric
import cloud.unum.usearch.DenseIndex.DenseOptions
import cloud.unum.usearch.Types.Executor
import cloud.unum.usearch.Types.Key
import cloud.unum.usearch.Types.RowsView
import cloud.unum.usearch.Types.Scalar

/**
 * The Java surface as a Scala caller meets it.
 *
 * Scala reads no nullness annotations, so what it covers is the other half: that the fluent builder, the array
 * arguments and the `AutoCloseable` handles all compose under a language whose collections are its own.
 */
class InteropTest {

  /** Scala's `Array[Key]` is a JVM array, so the batch verbs take it with no conversion. */
  @JUnitTest
  def batchesTakeScalaArrays(): Unit = {
    val options = new DenseOptions().dimensions(2).metric(DenseMetric.EUCLIDEAN).quantization(Scalar.BF16)
    Using.resources(Executor.forRam(Array.emptyIntArray, 2, 0), DenseIndex.initRam(options, Array.emptyIntArray)) {
      (pool, index) =>
        val receiver = index.on(pool)
        receiver.reserve(2)
        val keys: Array[Key] = Array(Key.of(1), Key.of(2))
        receiver.insert(keys, RowsView.of(Array(1.0f, 0.0f, 0.0f, 1.0f), 2))
        assertEquals(2L, receiver.describe().size())
    }
  }

  /** The identity vocabulary is the same string on every SDK. */
  @JUnitTest
  def keysRenderIdentically(): Unit = {
    val key = Key.of(42)
    assertEquals("00000000-0000-0000-0000-00000000002a", key.toString)
    assertEquals(key, Key.parse(key.toString))
  }

  @JUnitTest
  def theLibraryNamesItself(): Unit =
    assertTrue(!Types.version().isEmpty)
}
