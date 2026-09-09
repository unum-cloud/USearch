package cloud.unum.usearch

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
// Aliased, since the `class Test` below would otherwise shadow the annotation.
import org.junit.jupiter.api.{Test => JUnitTest}

import cloud.unum.usearch.DenseIndex.DenseMetric
import cloud.unum.usearch.DenseIndex.DenseOptions
import cloud.unum.usearch.Types.Executor
import cloud.unum.usearch.Types.Key
import cloud.unum.usearch.Types.Matrix
import cloud.unum.usearch.Types.ScalarKind

/**
 * The Java surface as a Scala caller meets it.
 *
 * Scala reads no nullness annotations, so what it covers is the other half: that the fluent builder, the array
 * arguments and the `AutoCloseable` handles all compose under a language whose collections are its own.
 */
class Test {

  /** Scala's `Array[Key]` is a JVM array, so the batch verbs take it with no conversion. */
  @JUnitTest
  def batchesTakeScalaArrays(): Unit = {
    val pool = Executor.forRam(2)
    try {
      val index = new DenseOptions()
        .dimensions(2)
        .metric(DenseMetric.EUCLIDEAN)
        .quantization(ScalarKind.F32)
        .open()
      try {
        index.reserve(2)
        val keys: Array[Key] = Array(Key.of(1), Key.of(2))
        val vectors = Matrix.of(Array(1.0f, 0.0f, 0.0f, 1.0f), 2)
        index.add(keys, vectors, pool)
        assertEquals(2L, index.size())
      } finally index.close()
    } finally pool.close()
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
