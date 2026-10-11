package cloud.unum.usearch

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

import cloud.unum.usearch.DenseIndex.DenseMetric
import cloud.unum.usearch.DenseIndex.DenseOptions
import cloud.unum.usearch.Types.Arena
import cloud.unum.usearch.Types.Deferred
import cloud.unum.usearch.Types.Executor
import cloud.unum.usearch.Types.Key
import cloud.unum.usearch.Types.Matches
import cloud.unum.usearch.Types.Mood
import cloud.unum.usearch.Types.RowsView
import cloud.unum.usearch.Types.Scalar
import cloud.unum.usearch.Types.Status
import cloud.unum.usearch.Types.USearchException

/**
 * The Java surface as a Kotlin caller meets it.
 *
 * Not a second copy of the Java suite: this covers what only another JVM language can observe - that `@NullMarked`
 * reaches Kotlin, so references arrive as real non-null types rather than platform types, and that the vocabulary reads
 * under Kotlin's own idioms.
 */
class InteropTest {
    /**
     * The declarations are the test, not the assertions. `val index: DenseIndex` type-checks against a non-null type
     * only because Kotlin read the package's nullness contract; without `@NullMarked` these would be platform types and
     * the annotations below would say nothing.
     */
    @Test
    fun nullnessReachesKotlin() {
        val options: DenseOptions = DenseOptions().dimensions(2).quantization(Scalar.F32)
        val index: DenseIndex = DenseIndex.initRam(options, intArrayOf())
        index.use {
            val capabilities: String = it.describe().capabilities()
            assertTrue(capabilities.isNotEmpty())
        }
    }

    /** `AutoCloseable` is Kotlin's `use`, so a handle's lifetime needs no `try`/`finally`. */
    @Test
    fun handlesCloseThroughUse() {
        Executor.forRam(intArrayOf(), 2, 0).use { pool ->
            val options = DenseOptions().dimensions(2).metric(DenseMetric.EUCLIDEAN).quantization(Scalar.F32)
            DenseIndex.initRam(options, intArrayOf()).use { index ->
                val receiver = index.on(pool)
                receiver.reserve(2)
                val keys = arrayOf(Key.of(1), Key.of(2))
                receiver.insert(keys, RowsView.of(floatArrayOf(1f, 0f, 0f, 1f), 2))
                assertEquals(2L, receiver.describe().size())
                // The arena is `AutoCloseable` too, and the view it answers reads in place until its next call.
                Arena(pool).use { arena -> assertEquals(1L, receiver.get(keys[0], Scalar.F32, arena).rows()) }
                pool.drain(Mood.PARKED)
            }
        }
    }

    /** A deferred call is `AutoCloseable` too; a coroutine waits with `withContext(Dispatchers.IO) { call.join() }`. */
    @Test
    fun deferredCallsJoinWhereKotlinWaits() {
        Executor.forRam(intArrayOf(), 0, 0).use { pool ->
            val options = DenseOptions().dimensions(2).quantization(Scalar.F32)
            DenseIndex.initRam(options, intArrayOf()).use { index ->
                index.insert(arrayOf(Key.of(1), Key.of(2)), RowsView.of(floatArrayOf(1f, 0f, 0f, 1f), 2))
                val query = RowsView.of(floatArrayOf(1f, 0f), 2)
                Arena(pool).use { arena ->
                    val call: Deferred<Matches> = index.deferred(pool).search(query, 1, arena)
                    val held =
                        call.use {
                            assertEquals(Status.DEFERRED, it.status())
                            // The deferred call holds the arena until joined, so a blocking one naming it is refused.
                            val refused = assertFailsWith<USearchException> { index.search(query, 1, arena) }
                            assertEquals(Status.CONTENDED, refused.status())
                            it.join()
                        }
                    assertEquals(Key.of(1), held.key(0, 0))
                    // The next call rewinds the arena, so the joined view refuses to read.
                    assertEquals(Key.of(1), index.search(query, 1, arena).key(0, 0))
                    assertFailsWith<IllegalStateException> { held.key(0, 0) }
                }
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
