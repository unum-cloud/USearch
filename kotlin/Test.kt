package cloud.unum.usearch

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

import cloud.unum.usearch.DenseIndex.DenseMetric
import cloud.unum.usearch.DenseIndex.DenseOptions
import cloud.unum.usearch.Types.Executor
import cloud.unum.usearch.Types.Key
import cloud.unum.usearch.Types.Matrix
import cloud.unum.usearch.Types.ScalarKind

/**
 * The Java surface as a Kotlin caller meets it.
 *
 * Not a second copy of the Java suite: this covers what only another JVM language can observe - that `@NullMarked`
 * reaches Kotlin, so references arrive as real non-null types rather than platform types, and that the vocabulary reads
 * under Kotlin's own idioms.
 */
class Test {
    /**
     * The declarations are the test, not the assertions. `val index: DenseIndex` type-checks against a non-null type
     * only because Kotlin read the package's nullness contract; without `@NullMarked` these would be platform types and
     * the annotations below would say nothing.
     */
    @Test
    fun nullnessReachesKotlin() {
        val options: DenseOptions = DenseOptions().dimensions(2).quantization(ScalarKind.F32)
        val index: DenseIndex = options.open()
        index.use {
            val capabilities: String = it.capabilities()
            assertTrue(capabilities.isNotEmpty())
        }
    }

    /** `AutoCloseable` is Kotlin's `use`, so a handle's lifetime needs no `try`/`finally`. */
    @Test
    fun handlesCloseThroughUse() {
        Executor.forRam(2).use { pool ->
            DenseOptions().dimensions(2).metric(DenseMetric.EUCLIDEAN).quantization(ScalarKind.F32).open().use { index ->
                index.reserve(2)
                val keys = arrayOf(Key.of(1), Key.of(2))
                val vectors = Matrix.of(floatArrayOf(1f, 0f, 0f, 1f), 2)
                index.add(keys, vectors, pool)
                assertEquals(2L, index.size())
                pool.drain()
            }
        }
    }

    /** A 128-bit key renders and parses the same on every SDK, so Kotlin reads what Java wrote. */
    @Test
    fun keysRoundTripThroughTheirRendering() {
        val key: Key = Key.of(42)
        assertEquals("00000000-0000-0000-0000-00000000002a", key.toString())
        assertEquals(key, Key.parse(key.toString()))
    }
}
