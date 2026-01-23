#include <jni.h>

#include "cloud_unum_usearch_Index.h"

#include <cstdio> // `std::snprintf`

#include <usearch/index_dense.hpp>

using namespace unum::usearch;
using namespace unum;

using f32_span_t = unum::usearch::span_gt<float>;
using f64_span_t = unum::usearch::span_gt<double>;
using i8_span_t = unum::usearch::span_gt<std::int8_t>;
static_assert(sizeof(jlong) == sizeof(index_dense_t::vector_key_t));

static inline jsize to_jsize(JNIEnv* env, std::size_t n) {
    if (n > static_cast<std::size_t>(std::numeric_limits<jsize>::max())) {
        jclass jc = env->FindClass("java/lang/IllegalArgumentException");
        if (jc)
            env->ThrowNew(jc, "Size exceeds jsize range");
        return 0;
    }
    return static_cast<jsize>(n);
}

JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_Index_c_1create( //
    JNIEnv* env, jclass,                                         //
    jstring metric, jstring quantization,                        //
    jlong dimensions, jlong capacity, jlong connectivity,        //
    jlong expansion_add, jlong expansion_search) {

    jlong result{};
    char const* metric_cstr{};
    char const* quantization_cstr{};
    try {

        metric_cstr = (*env).GetStringUTFChars(metric, 0);
        std::size_t metric_length = (*env).GetStringUTFLength(metric);
        quantization_cstr = (*env).GetStringUTFChars(quantization, 0);
        std::size_t quantization_length = (*env).GetStringUTFLength(quantization);
        metric_kind_t metric_kind = metric_from_name(metric_cstr, metric_length);
        scalar_kind_t quantization = scalar_kind_from_name(quantization_cstr, quantization_length);
        index_dense_config_t config(static_cast<std::size_t>(connectivity), static_cast<std::size_t>(expansion_add),
                                    static_cast<std::size_t>(expansion_search));
        metric_punned_t metric(static_cast<std::size_t>(dimensions), metric_kind, quantization);
        if (metric.missing()) {
            jclass jc = (*env).FindClass("java/lang/Error");
            if (jc)
                (*env).ThrowNew(jc, "Failed to initialize the metric!");
            goto cleanup;
        }

        index_dense_t index = index_dense_t::make(metric, config);
        if (!index.try_reserve(static_cast<std::size_t>(capacity))) {
            jclass jc = (*env).FindClass("java/lang/Error");
            if (jc)
                (*env).ThrowNew(jc, "Failed to reserve desired capacity!");
        } else {
            index_dense_t* result_ptr = new index_dense_t(std::move(index));
            std::memcpy(&result, &result_ptr, sizeof(jlong));
        }

    } catch (...) {
        jclass jc = (*env).FindClass("java/lang/Error");
        if (jc)
            (*env).ThrowNew(jc, "Failed to initialize the vector index!");
    }

cleanup:
    (*env).ReleaseStringUTFChars(metric, metric_cstr);
    (*env).ReleaseStringUTFChars(quantization, quantization_cstr);
    return result;
}

JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_Index_c_1createFromFile(JNIEnv* env, jclass, jstring path,
                                                                        jboolean view) {
    char const* path_cstr = env->GetStringUTFChars(path, 0);
    index_dense_t::state_result_t make_result = index_dense_t::make(path_cstr, view);
    env->ReleaseStringUTFChars(path, path_cstr);
    if (!make_result) {
        jclass jc = env->FindClass("java/lang/Error");
        if (jc) {
            env->ThrowNew(jc, make_result.error.release());
        }
    }
    index_dense_t* result_ptr = new index_dense_t(std::move(make_result.index));
    jlong result;
    std::memcpy(&result, &result_ptr, sizeof(jlong));
    return result;
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1save(JNIEnv* env, jclass, jlong c_ptr, jstring path) {
    char const* path_cstr = (*env).GetStringUTFChars(path, 0);
    serialization_result_t result = reinterpret_cast<index_dense_t*>(c_ptr)->save(path_cstr);
    if (!result) {
        jclass jc = (*env).FindClass("java/lang/Error");
        if (jc)
            (*env).ThrowNew(jc, result.error.release());
    }
    (*env).ReleaseStringUTFChars(path, path_cstr);
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1load(JNIEnv* env, jclass, jlong c_ptr, jstring path) {
    char const* path_cstr = (*env).GetStringUTFChars(path, 0);
    serialization_result_t result = reinterpret_cast<index_dense_t*>(c_ptr)->load(path_cstr);
    if (!result) {
        jclass jc = (*env).FindClass("java/lang/Error");
        if (jc)
            (*env).ThrowNew(jc, result.error.release());
    }
    (*env).ReleaseStringUTFChars(path, path_cstr);
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1view(JNIEnv* env, jclass, jlong c_ptr, jstring path) {
    char const* path_cstr = (*env).GetStringUTFChars(path, 0);
    serialization_result_t result = reinterpret_cast<index_dense_t*>(c_ptr)->view(path_cstr);
    if (!result) {
        jclass jc = (*env).FindClass("java/lang/Error");
        if (jc)
            (*env).ThrowNew(jc, result.error.release());
    }
    (*env).ReleaseStringUTFChars(path, path_cstr);
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1destroy(JNIEnv*, jclass, jlong c_ptr) {
    delete reinterpret_cast<index_dense_t*>(c_ptr);
}

JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_Index_c_1size(JNIEnv*, jclass, jlong c_ptr) {
    return reinterpret_cast<index_dense_t*>(c_ptr)->size();
}

JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_Index_c_1connectivity(JNIEnv*, jclass, jlong c_ptr) {
    return reinterpret_cast<index_dense_t*>(c_ptr)->connectivity();
}

JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_Index_c_1dimensions(JNIEnv*, jclass, jlong c_ptr) {
    return reinterpret_cast<index_dense_t*>(c_ptr)->dimensions();
}

JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_Index_c_1capacity(JNIEnv*, jclass, jlong c_ptr) {
    return reinterpret_cast<index_dense_t*>(c_ptr)->capacity();
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1reserve(JNIEnv* env, jclass, jlong c_ptr, jlong capacity,
                                                                jlong threads_add, jlong threads_search) {
    std::size_t t_add = static_cast<std::size_t>(threads_add);
    std::size_t t_search = static_cast<std::size_t>(threads_search);
    if (t_add == 0 || t_search == 0) {
        std::size_t hc = std::thread::hardware_concurrency();
        if (hc == 0)
            hc = 1; // fallback to 1 if the runtime can't  report
        if (t_add == 0)
            t_add = hc;
        if (t_search == 0)
            t_search = hc;
    }
    index_limits_t limits(static_cast<std::size_t>(capacity), t_add);
    limits.threads_search = t_search;
    if (!reinterpret_cast<index_dense_t*>(c_ptr)->try_reserve(limits)) {
        jclass jc = (*env).FindClass("java/lang/Error");
        if (jc)
            (*env).ThrowNew(jc, "Failed to grow vector index!");
    }
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1add_1f32( //
    JNIEnv* env, jclass, jlong c_ptr, jlong key, jfloatArray vector) {

    jfloat* vector_data = (*env).GetFloatArrayElements(vector, 0);
    jsize vector_length = (*env).GetArrayLength(vector);

    auto index = reinterpret_cast<index_dense_t*>(c_ptr);
    size_t dimensions = index->dimensions();

    using vector_key_t = typename index_dense_t::vector_key_t;
    using add_result_t = typename index_dense_t::add_result_t;

    // Handle both single and batch processing uniformly
    if (vector_length % dimensions != 0) {
        (*env).ReleaseFloatArrayElements(vector, vector_data, 0);
        jclass jc = (*env).FindClass("java/lang/IllegalArgumentException");
        if (jc)
            (*env).ThrowNew(jc, "Vector length must be a multiple of dimensions");
        return;
    }

    size_t num_vectors = vector_length / dimensions;
    for (size_t i = 0; i < num_vectors; i++) {
        f32_span_t vector_span = f32_span_t{vector_data + i * dimensions, dimensions};
        add_result_t result = index->add(static_cast<vector_key_t>(key + i), vector_span);
        if (!result) {
            (*env).ReleaseFloatArrayElements(vector, vector_data, 0);
            jclass jc = (*env).FindClass("java/lang/Error");
            if (jc)
                (*env).ThrowNew(jc, result.error.release());
            return;
        }
    }
    (*env).ReleaseFloatArrayElements(vector, vector_data, 0);
}

JNIEXPORT jfloatArray JNICALL Java_cloud_unum_usearch_Index_c_1get(JNIEnv* env, jclass, jlong c_ptr, jlong key) {
    auto index = reinterpret_cast<index_dense_t*>(c_ptr);
    size_t dim = index->dimensions();
    std::unique_ptr<jfloat[]> vector(new jfloat[dim]);
    if (index->get(key, vector.get()) == 0) {
        jclass jc = env->FindClass("java/lang/IllegalArgumentException");
        if (jc) {
            env->ThrowNew(jc, "key not found");
        }
    }
    jfloatArray jvector = env->NewFloatArray(to_jsize(env, dim));
    if (jvector == nullptr) { // out of memory
        return nullptr;
    }
    env->SetFloatArrayRegion(jvector, 0, to_jsize(env, dim), vector.get());
    return jvector;
}

JNIEXPORT jlongArray JNICALL Java_cloud_unum_usearch_Index_c_1search_1f32( //
    JNIEnv* env, jclass, jlong c_ptr, jfloatArray vector, jlong wanted) {

    jfloat* vector_data = (*env).GetFloatArrayElements(vector, 0);
    jsize vector_dims = (*env).GetArrayLength(vector);
    f32_span_t vector_span = f32_span_t{vector_data, static_cast<std::size_t>(vector_dims)};

    using vector_key_t = typename index_dense_t::vector_key_t;
    using search_result_t = typename index_dense_t::search_result_t;

    search_result_t result =
        reinterpret_cast<index_dense_t*>(c_ptr)->search(vector_span, static_cast<std::size_t>(wanted));
    (*env).ReleaseFloatArrayElements(vector, vector_data, 0);

    if (result) {
        std::size_t found = result.count;
        jlongArray matches = (*env).NewLongArray(to_jsize(env, found));
        if (matches == NULL)
            return NULL;

        jlong* matches_data = (*env).GetLongArrayElements(matches, 0);
        result.dump_to(reinterpret_cast<vector_key_t*>(matches_data));
        (*env).ReleaseLongArrayElements(matches, matches_data, JNI_COMMIT);

        return matches;
    } else {
        jclass jc = (*env).FindClass("java/lang/Error");
        if (jc)
            (*env).ThrowNew(jc, result.error.release());
        return NULL;
    }
}

JNIEXPORT jboolean JNICALL Java_cloud_unum_usearch_Index_c_1remove(JNIEnv* env, jclass, jlong c_ptr, jlong key) {
    using vector_key_t = typename index_dense_t::vector_key_t;
    using labeling_result_t = typename index_dense_t::labeling_result_t;
    labeling_result_t result = reinterpret_cast<index_dense_t*>(c_ptr)->remove(static_cast<vector_key_t>(key));
    if (!result) {
        jclass jc = (*env).FindClass("java/lang/Error");
        if (jc)
            (*env).ThrowNew(jc, result.error.release());
    }
    return result.completed ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jboolean JNICALL Java_cloud_unum_usearch_Index_c_1rename(JNIEnv* env, jclass, jlong c_ptr, jlong from,
                                                                   jlong to) {
    using vector_key_t = typename index_dense_t::vector_key_t;
    using labeling_result_t = typename index_dense_t::labeling_result_t;
    labeling_result_t result =
        reinterpret_cast<index_dense_t*>(c_ptr)->rename(static_cast<vector_key_t>(from), static_cast<vector_key_t>(to));
    if (!result) {
        jclass jc = (*env).FindClass("java/lang/Error");
        if (jc)
            (*env).ThrowNew(jc, result.error.release());
    }
    return result.completed ? JNI_TRUE : JNI_FALSE;
}

JNIEXPORT jlong JNICALL Java_cloud_unum_usearch_Index_c_1memory_1usage(JNIEnv*, jclass, jlong c_ptr) {
    auto index = reinterpret_cast<index_dense_t*>(c_ptr);
    return static_cast<jlong>(index->memory_usage());
}

JNIEXPORT jstring JNICALL Java_cloud_unum_usearch_Index_c_1hardware_1acceleration(JNIEnv* env, jclass, jlong c_ptr) {
    auto index = reinterpret_cast<index_dense_t*>(c_ptr);
    return env->NewStringUTF(index->metric().isa_name());
}

JNIEXPORT jstring JNICALL Java_cloud_unum_usearch_Index_c_1metric_1kind(JNIEnv* env, jclass, jlong c_ptr) {
    auto index = reinterpret_cast<index_dense_t*>(c_ptr);
    return env->NewStringUTF(metric_kind_name(index->metric().metric_kind()));
}

JNIEXPORT jstring JNICALL Java_cloud_unum_usearch_Index_c_1scalar_1kind(JNIEnv* env, jclass, jlong c_ptr) {
    auto index = reinterpret_cast<index_dense_t*>(c_ptr);
    return env->NewStringUTF(scalar_kind_name(index->metric().scalar_kind()));
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1add_1f64( //
    JNIEnv* env, jclass, jlong c_ptr, jlong key, jdoubleArray vector) {

    jdouble* vector_data = (*env).GetDoubleArrayElements(vector, 0);
    jsize vector_length = (*env).GetArrayLength(vector);

    auto index = reinterpret_cast<index_dense_t*>(c_ptr);
    size_t dimensions = index->dimensions();

    using vector_key_t = typename index_dense_t::vector_key_t;
    using add_result_t = typename index_dense_t::add_result_t;

    // Handle both single and batch processing uniformly
    if (vector_length % dimensions != 0) {
        (*env).ReleaseDoubleArrayElements(vector, vector_data, 0);
        jclass jc = (*env).FindClass("java/lang/IllegalArgumentException");
        if (jc)
            (*env).ThrowNew(jc, "Vector length must be a multiple of dimensions");
        return;
    }

    size_t num_vectors = vector_length / dimensions;
    for (size_t i = 0; i < num_vectors; i++) {
        f64_span_t vector_span = f64_span_t{vector_data + i * dimensions, dimensions};
        add_result_t result = index->add(static_cast<vector_key_t>(key + i), vector_span);
        if (!result) {
            (*env).ReleaseDoubleArrayElements(vector, vector_data, 0);
            jclass jc = (*env).FindClass("java/lang/Error");
            if (jc)
                (*env).ThrowNew(jc, result.error.release());
            return;
        }
    }
    (*env).ReleaseDoubleArrayElements(vector, vector_data, 0);
}

JNIEXPORT jlongArray JNICALL Java_cloud_unum_usearch_Index_c_1search_1f64( //
    JNIEnv* env, jclass, jlong c_ptr, jdoubleArray vector, jlong wanted) {

    jdouble* vector_data = (*env).GetDoubleArrayElements(vector, 0);
    jsize vector_dims = (*env).GetArrayLength(vector);
    f64_span_t vector_span = f64_span_t{vector_data, static_cast<std::size_t>(vector_dims)};

    using vector_key_t = typename index_dense_t::vector_key_t;
    using search_result_t = typename index_dense_t::search_result_t;

    search_result_t result =
        reinterpret_cast<index_dense_t*>(c_ptr)->search(vector_span, static_cast<std::size_t>(wanted));
    (*env).ReleaseDoubleArrayElements(vector, vector_data, 0);

    if (result) {
        std::size_t found = result.count;
        jlongArray matches = (*env).NewLongArray(to_jsize(env, found));
        if (matches == NULL)
            return NULL;

        jlong* matches_data = (*env).GetLongArrayElements(matches, 0);
        result.dump_to(reinterpret_cast<vector_key_t*>(matches_data));
        (*env).ReleaseLongArrayElements(matches, matches_data, JNI_COMMIT);

        return matches;
    } else {
        jclass jc = (*env).FindClass("java/lang/Error");
        if (jc)
            (*env).ThrowNew(jc, result.error.release());
        return NULL;
    }
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1get_1into_1f64(JNIEnv* env, jclass, jlong c_ptr, jlong key,
                                                                       jdoubleArray buffer) {
    auto index = reinterpret_cast<index_dense_t*>(c_ptr);
    jdouble* buffer_data = (*env).GetDoubleArrayElements(buffer, 0);

    if (index->get(key, buffer_data) == 0) {
        (*env).ReleaseDoubleArrayElements(buffer, buffer_data, JNI_ABORT);
        jclass jc = env->FindClass("java/lang/IllegalArgumentException");
        if (jc) {
            env->ThrowNew(jc, "key not found");
        }
        return;
    }
    (*env).ReleaseDoubleArrayElements(buffer, buffer_data, JNI_COMMIT);
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1add_1i8( //
    JNIEnv* env, jclass, jlong c_ptr, jlong key, jbyteArray vector) {

    jbyte* vector_data = (*env).GetByteArrayElements(vector, 0);
    jsize vector_length = (*env).GetArrayLength(vector);

    auto index = reinterpret_cast<index_dense_t*>(c_ptr);
    size_t dimensions = index->dimensions();

    using vector_key_t = typename index_dense_t::vector_key_t;
    using add_result_t = typename index_dense_t::add_result_t;

    // Handle both single and batch processing uniformly
    if (vector_length % dimensions != 0) {
        (*env).ReleaseByteArrayElements(vector, vector_data, 0);
        jclass jc = (*env).FindClass("java/lang/IllegalArgumentException");
        if (jc)
            (*env).ThrowNew(jc, "Vector length must be a multiple of dimensions");
        return;
    }

    size_t num_vectors = vector_length / dimensions;
    for (size_t i = 0; i < num_vectors; i++) {
        i8_span_t vector_span = i8_span_t{reinterpret_cast<std::int8_t*>(vector_data + i * dimensions), dimensions};
        add_result_t result = index->add(static_cast<vector_key_t>(key + i), vector_span);
        if (!result) {
            (*env).ReleaseByteArrayElements(vector, vector_data, 0);
            jclass jc = (*env).FindClass("java/lang/Error");
            if (jc)
                (*env).ThrowNew(jc, result.error.release());
            return;
        }
    }
    (*env).ReleaseByteArrayElements(vector, vector_data, 0);
}

JNIEXPORT jlongArray JNICALL Java_cloud_unum_usearch_Index_c_1search_1i8( //
    JNIEnv* env, jclass, jlong c_ptr, jbyteArray vector, jlong wanted) {

    jbyte* vector_data = (*env).GetByteArrayElements(vector, 0);
    jsize vector_dims = (*env).GetArrayLength(vector);
    i8_span_t vector_span =
        i8_span_t{reinterpret_cast<std::int8_t*>(vector_data), static_cast<std::size_t>(vector_dims)};

    using vector_key_t = typename index_dense_t::vector_key_t;
    using search_result_t = typename index_dense_t::search_result_t;

    search_result_t result =
        reinterpret_cast<index_dense_t*>(c_ptr)->search(vector_span, static_cast<std::size_t>(wanted));
    (*env).ReleaseByteArrayElements(vector, vector_data, 0);

    if (result) {
        std::size_t found = result.count;
        jlongArray matches = (*env).NewLongArray(to_jsize(env, found));
        if (matches == NULL)
            return NULL;

        jlong* matches_data = (*env).GetLongArrayElements(matches, 0);
        result.dump_to(reinterpret_cast<vector_key_t*>(matches_data));
        (*env).ReleaseLongArrayElements(matches, matches_data, JNI_COMMIT);

        return matches;
    } else {
        jclass jc = (*env).FindClass("java/lang/Error");
        if (jc)
            (*env).ThrowNew(jc, result.error.release());
        return NULL;
    }
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1get_1into_1i8(JNIEnv* env, jclass, jlong c_ptr, jlong key,
                                                                      jbyteArray buffer) {
    auto index = reinterpret_cast<index_dense_t*>(c_ptr);
    jbyte* buffer_data = (*env).GetByteArrayElements(buffer, 0);

    if (index->get(key, reinterpret_cast<std::int8_t*>(buffer_data)) == 0) {
        (*env).ReleaseByteArrayElements(buffer, buffer_data, JNI_ABORT);
        jclass jc = env->FindClass("java/lang/IllegalArgumentException");
        if (jc) {
            env->ThrowNew(jc, "key not found");
        }
        return;
    }
    (*env).ReleaseByteArrayElements(buffer, buffer_data, JNI_COMMIT);
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1add_1f32_1buffer( //
    JNIEnv* env, jclass, jlong c_ptr, jlong key, jobject vector_buffer) {

    float* vector_data = static_cast<float*>(env->GetDirectBufferAddress(vector_buffer));
    if (!vector_data) {
        jclass jc = env->FindClass("java/lang/IllegalArgumentException");
        if (jc) {
            env->ThrowNew(jc, "Buffer must be direct ByteBuffer");
        }
        return;
    }

    auto index = reinterpret_cast<index_dense_t*>(c_ptr);
    size_t dimensions = index->dimensions();

    using vector_key_t = typename index_dense_t::vector_key_t;
    using add_result_t = typename index_dense_t::add_result_t;

    f32_span_t vector_span = f32_span_t{vector_data, dimensions};
    add_result_t result = index->add(static_cast<vector_key_t>(key), vector_span);

    if (!result) {
        jclass jc = env->FindClass("java/lang/Error");
        if (jc) {
            env->ThrowNew(jc, result.error.release());
        }
    }
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1add_1f64_1buffer( //
    JNIEnv* env, jclass, jlong c_ptr, jlong key, jobject vector_buffer) {

    double* vector_data = static_cast<double*>(env->GetDirectBufferAddress(vector_buffer));
    if (!vector_data) {
        jclass jc = env->FindClass("java/lang/IllegalArgumentException");
        if (jc) {
            env->ThrowNew(jc, "Buffer must be direct ByteBuffer");
        }
        return;
    }

    auto index = reinterpret_cast<index_dense_t*>(c_ptr);
    size_t dimensions = index->dimensions();

    using vector_key_t = typename index_dense_t::vector_key_t;
    using add_result_t = typename index_dense_t::add_result_t;

    f64_span_t vector_span = f64_span_t{vector_data, dimensions};
    add_result_t result = index->add(static_cast<vector_key_t>(key), vector_span);

    if (!result) {
        jclass jc = env->FindClass("java/lang/Error");
        if (jc) {
            env->ThrowNew(jc, result.error.release());
        }
    }
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1add_1i8_1buffer( //
    JNIEnv* env, jclass, jlong c_ptr, jlong key, jobject vector_buffer) {

    std::int8_t* vector_data = static_cast<std::int8_t*>(env->GetDirectBufferAddress(vector_buffer));
    if (!vector_data) {
        jclass jc = env->FindClass("java/lang/IllegalArgumentException");
        if (jc) {
            env->ThrowNew(jc, "Buffer must be direct ByteBuffer");
        }
        return;
    }

    jlong capacity = env->GetDirectBufferCapacity(vector_buffer);
    jlong vector_dims = capacity;

    auto index = reinterpret_cast<index_dense_t*>(c_ptr);
    size_t dimensions = index->dimensions();

    if (vector_dims != static_cast<jlong>(dimensions)) {
        jclass jc = env->FindClass("java/lang/IllegalArgumentException");
        if (jc) {
            env->ThrowNew(jc, "Vector dimensions mismatch");
        }
        return;
    }

    using vector_key_t = typename index_dense_t::vector_key_t;
    using add_result_t = typename index_dense_t::add_result_t;

    i8_span_t vector_span = i8_span_t{vector_data, static_cast<std::size_t>(vector_dims)};
    add_result_t result = index->add(static_cast<vector_key_t>(key), vector_span);

    if (!result) {
        jclass jc = env->FindClass("java/lang/Error");
        if (jc) {
            env->ThrowNew(jc, result.error.release());
        }
    }
}

JNIEXPORT jlongArray JNICALL Java_cloud_unum_usearch_Index_c_1search_1f32_1buffer( //
    JNIEnv* env, jclass, jlong c_ptr, jobject vector_buffer, jlong wanted) {

    float* vector_data = static_cast<float*>(env->GetDirectBufferAddress(vector_buffer));
    if (!vector_data) {
        jclass jc = env->FindClass("java/lang/IllegalArgumentException");
        if (jc) {
            env->ThrowNew(jc, "Buffer must be direct ByteBuffer");
        }
        return nullptr;
    }

    // Dimensions are validated on Java side
    jlong capacity = env->GetDirectBufferCapacity(vector_buffer);
    jlong vector_dims = capacity / sizeof(float);
    f32_span_t vector_span = f32_span_t{vector_data, static_cast<std::size_t>(vector_dims)};

    using vector_key_t = typename index_dense_t::vector_key_t;
    using search_result_t = typename index_dense_t::search_result_t;

    search_result_t result =
        reinterpret_cast<index_dense_t*>(c_ptr)->search(vector_span, static_cast<std::size_t>(wanted));

    if (result) {
        std::size_t found = result.count;
        jlongArray matches = env->NewLongArray(to_jsize(env, found));
        if (matches == nullptr)
            return nullptr;

        jlong* matches_data = env->GetLongArrayElements(matches, 0);
        result.dump_to(reinterpret_cast<vector_key_t*>(matches_data));
        env->ReleaseLongArrayElements(matches, matches_data, JNI_COMMIT);

        return matches;
    } else {
        jclass jc = env->FindClass("java/lang/Error");
        if (jc)
            env->ThrowNew(jc, result.error.release());
        return nullptr;
    }
}

JNIEXPORT jlongArray JNICALL Java_cloud_unum_usearch_Index_c_1search_1f64_1buffer( //
    JNIEnv* env, jclass, jlong c_ptr, jobject vector_buffer, jlong wanted) {

    double* vector_data = static_cast<double*>(env->GetDirectBufferAddress(vector_buffer));
    if (!vector_data) {
        jclass jc = env->FindClass("java/lang/IllegalArgumentException");
        if (jc) {
            env->ThrowNew(jc, "Buffer must be direct ByteBuffer");
        }
        return nullptr;
    }

    // Dimensions are validated on Java side
    jlong capacity = env->GetDirectBufferCapacity(vector_buffer);
    jlong vector_dims = capacity / sizeof(double);
    f64_span_t vector_span = f64_span_t{vector_data, static_cast<std::size_t>(vector_dims)};

    using vector_key_t = typename index_dense_t::vector_key_t;
    using search_result_t = typename index_dense_t::search_result_t;

    search_result_t result =
        reinterpret_cast<index_dense_t*>(c_ptr)->search(vector_span, static_cast<std::size_t>(wanted));

    if (result) {
        std::size_t found = result.count;
        jlongArray matches = env->NewLongArray(to_jsize(env, found));
        if (matches == nullptr)
            return nullptr;

        jlong* matches_data = env->GetLongArrayElements(matches, 0);
        result.dump_to(reinterpret_cast<vector_key_t*>(matches_data));
        env->ReleaseLongArrayElements(matches, matches_data, JNI_COMMIT);

        return matches;
    } else {
        jclass jc = env->FindClass("java/lang/Error");
        if (jc)
            env->ThrowNew(jc, result.error.release());
        return nullptr;
    }
}

JNIEXPORT jlongArray JNICALL Java_cloud_unum_usearch_Index_c_1search_1i8_1buffer( //
    JNIEnv* env, jclass, jlong c_ptr, jobject vector_buffer, jlong wanted) {

    std::int8_t* vector_data = static_cast<std::int8_t*>(env->GetDirectBufferAddress(vector_buffer));
    if (!vector_data) {
        jclass jc = env->FindClass("java/lang/IllegalArgumentException");
        if (jc) {
            env->ThrowNew(jc, "Buffer must be direct ByteBuffer");
        }
        return nullptr;
    }

    jlong capacity = env->GetDirectBufferCapacity(vector_buffer);
    jlong vector_dims = capacity;
    i8_span_t vector_span = i8_span_t{vector_data, static_cast<std::size_t>(vector_dims)};

    using vector_key_t = typename index_dense_t::vector_key_t;
    using search_result_t = typename index_dense_t::search_result_t;

    search_result_t result =
        reinterpret_cast<index_dense_t*>(c_ptr)->search(vector_span, static_cast<std::size_t>(wanted));

    if (result) {
        std::size_t found = result.count;
        jlongArray matches = env->NewLongArray(to_jsize(env, found));
        if (matches == nullptr)
            return nullptr;

        jlong* matches_data = env->GetLongArrayElements(matches, 0);
        result.dump_to(reinterpret_cast<vector_key_t*>(matches_data));
        env->ReleaseLongArrayElements(matches, matches_data, JNI_COMMIT);

        return matches;
    } else {
        jclass jc = env->FindClass("java/lang/Error");
        if (jc)
            env->ThrowNew(jc, result.error.release());
        return nullptr;
    }
}

JNIEXPORT void JNICALL Java_cloud_unum_usearch_Index_c_1get_1into_1f32(JNIEnv* env, jclass, jlong c_ptr, jlong key,
                                                                       jfloatArray buffer) {
    auto index = reinterpret_cast<index_dense_t*>(c_ptr);
    jfloat* buffer_data = (*env).GetFloatArrayElements(buffer, 0);

    if (index->get(key, buffer_data) == 0) {
        (*env).ReleaseFloatArrayElements(buffer, buffer_data, JNI_ABORT);
        jclass jc = env->FindClass("java/lang/IllegalArgumentException");
        if (jc) {
            env->ThrowNew(jc, "key not found");
        }
        return;
    }
    (*env).ReleaseFloatArrayElements(buffer, buffer_data, JNI_COMMIT);
}

// Zero-allocation searchInto methods
JNIEXPORT jint JNICALL Java_cloud_unum_usearch_Index_c_1search_1into_1f32_1buffer( //
    JNIEnv* env, jclass, jlong c_ptr, jobject query_buffer, jobject results_buffer, jlong max_count) {

    float* query_data = static_cast<float*>(env->GetDirectBufferAddress(query_buffer));
    if (!query_data) {
        jclass jc = env->FindClass("java/lang/IllegalArgumentException");
        if (jc) {
            env->ThrowNew(jc, "Query buffer must be direct ByteBuffer");
        }
        return 0;
    }

    long* results_data = static_cast<long*>(env->GetDirectBufferAddress(results_buffer));
    if (!results_data) {
        jclass jc = env->FindClass("java/lang/IllegalArgumentException");
        if (jc) {
            env->ThrowNew(jc, "Results buffer must be direct ByteBuffer");
        }
        return 0;
    }

    // Dimensions are validated on Java side
    jlong query_capacity = env->GetDirectBufferCapacity(query_buffer);
    jlong query_dims = query_capacity / sizeof(float);
    f32_span_t query_span = f32_span_t{query_data, static_cast<std::size_t>(query_dims)};

    using vector_key_t = typename index_dense_t::vector_key_t;
    using search_result_t = typename index_dense_t::search_result_t;

    search_result_t result =
        reinterpret_cast<index_dense_t*>(c_ptr)->search(query_span, static_cast<std::size_t>(max_count));

    if (result) {
        std::size_t found = result.count;

        // Copy results directly into the LongBuffer
        auto* keys_ptr = reinterpret_cast<vector_key_t*>(results_data);
        result.dump_to(keys_ptr);

        // Advance the LongBuffer position
        jclass bufferClass = env->GetObjectClass(results_buffer);
        jmethodID positionMethod = env->GetMethodID(bufferClass, "position", "(I)Ljava/nio/Buffer;");
        jmethodID getPositionMethod = env->GetMethodID(bufferClass, "position", "()I");
        jint currentPos = env->CallIntMethod(results_buffer, getPositionMethod);
        env->CallObjectMethod(results_buffer, positionMethod, currentPos + static_cast<jint>(found));

        return static_cast<jint>(found);
    } else {
        jclass jc = env->FindClass("java/lang/Error");
        if (jc)
            env->ThrowNew(jc, result.error.release());
        return 0;
    }
}

JNIEXPORT jint JNICALL Java_cloud_unum_usearch_Index_c_1search_1into_1f64_1buffer( //
    JNIEnv* env, jclass, jlong c_ptr, jobject query_buffer, jobject results_buffer, jlong max_count) {

    double* query_data = static_cast<double*>(env->GetDirectBufferAddress(query_buffer));
    if (!query_data) {
        jclass jc = env->FindClass("java/lang/IllegalArgumentException");
        if (jc) {
            env->ThrowNew(jc, "Query buffer must be direct ByteBuffer");
        }
        return 0;
    }

    long* results_data = static_cast<long*>(env->GetDirectBufferAddress(results_buffer));
    if (!results_data) {
        jclass jc = env->FindClass("java/lang/IllegalArgumentException");
        if (jc) {
            env->ThrowNew(jc, "Results buffer must be direct ByteBuffer");
        }
        return 0;
    }

    // Dimensions are validated on Java side
    jlong query_capacity = env->GetDirectBufferCapacity(query_buffer);
    jlong query_dims = query_capacity / sizeof(double);
    f64_span_t query_span = f64_span_t{query_data, static_cast<std::size_t>(query_dims)};

    using vector_key_t = typename index_dense_t::vector_key_t;
    using search_result_t = typename index_dense_t::search_result_t;

    search_result_t result =
        reinterpret_cast<index_dense_t*>(c_ptr)->search(query_span, static_cast<std::size_t>(max_count));

    if (result) {
        std::size_t found = result.count;

        // Copy results directly into the LongBuffer
        auto* keys_ptr = reinterpret_cast<vector_key_t*>(results_data);
        result.dump_to(keys_ptr);

        // Advance the LongBuffer position
        jclass bufferClass = env->GetObjectClass(results_buffer);
        jmethodID positionMethod = env->GetMethodID(bufferClass, "position", "(I)Ljava/nio/Buffer;");
        jmethodID getPositionMethod = env->GetMethodID(bufferClass, "position", "()I");
        jint currentPos = env->CallIntMethod(results_buffer, getPositionMethod);
        env->CallObjectMethod(results_buffer, positionMethod, currentPos + static_cast<jint>(found));

        return static_cast<jint>(found);
    } else {
        jclass jc = env->FindClass("java/lang/Error");
        if (jc)
            env->ThrowNew(jc, result.error.release());
        return 0;
    }
}

JNIEXPORT jint JNICALL Java_cloud_unum_usearch_Index_c_1search_1into_1i8_1buffer( //
    JNIEnv* env, jclass, jlong c_ptr, jobject query_buffer, jobject results_buffer, jlong max_count) {

    std::int8_t* query_data = static_cast<std::int8_t*>(env->GetDirectBufferAddress(query_buffer));
    if (!query_data) {
        jclass jc = env->FindClass("java/lang/IllegalArgumentException");
        if (jc) {
            env->ThrowNew(jc, "Query buffer must be direct ByteBuffer");
        }
        return 0;
    }

    long* results_data = static_cast<long*>(env->GetDirectBufferAddress(results_buffer));
    if (!results_data) {
        jclass jc = env->FindClass("java/lang/IllegalArgumentException");
        if (jc) {
            env->ThrowNew(jc, "Results buffer must be direct ByteBuffer");
        }
        return 0;
    }

    // Dimensions are validated on Java side
    jlong query_capacity = env->GetDirectBufferCapacity(query_buffer);
    i8_span_t query_span = i8_span_t{query_data, static_cast<std::size_t>(query_capacity)};

    using vector_key_t = typename index_dense_t::vector_key_t;
    using search_result_t = typename index_dense_t::search_result_t;

    search_result_t result =
        reinterpret_cast<index_dense_t*>(c_ptr)->search(query_span, static_cast<std::size_t>(max_count));

    if (result) {
        std::size_t found = result.count;

        // Copy results directly into the LongBuffer
        auto* keys_ptr = reinterpret_cast<vector_key_t*>(results_data);
        result.dump_to(keys_ptr);

        // Advance the LongBuffer position
        jclass bufferClass = env->GetObjectClass(results_buffer);
        jmethodID positionMethod = env->GetMethodID(bufferClass, "position", "(I)Ljava/nio/Buffer;");
        jmethodID getPositionMethod = env->GetMethodID(bufferClass, "position", "()I");
        jint currentPos = env->CallIntMethod(results_buffer, getPositionMethod);
        env->CallObjectMethod(results_buffer, positionMethod, currentPos + static_cast<jint>(found));

        return static_cast<jint>(found);
    } else {
        jclass jc = env->FindClass("java/lang/Error");
        if (jc)
            env->ThrowNew(jc, result.error.release());
        return 0;
    }
}

JNIEXPORT jobjectArray JNICALL Java_cloud_unum_usearch_Index_c_1hardware_1acceleration_1available(JNIEnv* env, jclass) {
#if USEARCH_USE_SIMSIMD
    simsimd_capability_t caps = simsimd_capabilities();

    // Define capability mappings
    struct {
        simsimd_capability_t cap;
        const char* name;
    } capabilities[] = {
        {simsimd_cap_serial_k, "serial"},     {simsimd_cap_haswell_k, "haswell"},
        {simsimd_cap_skylake_k, "skylake"},   {simsimd_cap_ice_k, "ice"},
        {simsimd_cap_genoa_k, "genoa"},       {simsimd_cap_sapphire_k, "sapphire"},
        {simsimd_cap_turin_k, "turin"},       {simsimd_cap_sierra_k, "sierra"},
        {simsimd_cap_neon_k, "neon"},         {simsimd_cap_neon_i8_k, "neon_i8"},
        {simsimd_cap_neon_f16_k, "neon_f16"}, {simsimd_cap_neon_bf16_k, "neon_bf16"},
        {simsimd_cap_sve_k, "sve"},           {simsimd_cap_sve_i8_k, "sve_i8"},
        {simsimd_cap_sve_f16_k, "sve_f16"},   {simsimd_cap_sve_bf16_k, "sve_bf16"},
        {simsimd_cap_sve2_k, "sve2"},         {simsimd_cap_sve2p1_k, "sve2p1"},
    };
    int const cap_count = sizeof(capabilities) / sizeof(capabilities[0]);

    // Count supported capabilities
    int supported_count = 0;
    for (int i = 0; i < cap_count; i++)
        if (caps & capabilities[i].cap)
            supported_count++;

    // Create Java string array
    jclass stringClass = env->FindClass("java/lang/String");
    jobjectArray result = env->NewObjectArray(supported_count, stringClass, nullptr);

    int index = 0;
    for (int i = 0; i < cap_count; i++) {
        if (caps & capabilities[i].cap) {
            jstring capName = env->NewStringUTF(capabilities[i].name);
            env->SetObjectArrayElement(result, index++, capName);
            env->DeleteLocalRef(capName);
        }
    }

    return result;
#else
    // If SimSIMD is not enabled, return only serial
    jclass stringClass = env->FindClass("java/lang/String");
    jobjectArray result = env->NewObjectArray(1, stringClass, nullptr);
    jstring serialCap = env->NewStringUTF("serial");
    env->SetObjectArrayElement(result, 0, serialCap);
    env->DeleteLocalRef(serialCap);
    return result;
#endif
}

JNIEXPORT jobjectArray JNICALL Java_cloud_unum_usearch_Index_c_1hardware_1acceleration_1compiled(JNIEnv* env, jclass) {
#if USEARCH_USE_SIMSIMD
    // Define compile-time capabilities based on preprocessor macros
    struct {
        int compiled;
        char const* name;
    } compiled_capabilities[] = {
        {1, "serial"}, // Always available
        {SIMSIMD_TARGET_HASWELL, "haswell"},
        {SIMSIMD_TARGET_SKYLAKE, "skylake"},
        {SIMSIMD_TARGET_ICE, "ice"},
        {SIMSIMD_TARGET_GENOA, "genoa"},
        {SIMSIMD_TARGET_SAPPHIRE, "sapphire"},
        {SIMSIMD_TARGET_TURIN, "turin"},
        {SIMSIMD_TARGET_SIERRA, "sierra"},
        {SIMSIMD_TARGET_NEON, "neon"},
        {SIMSIMD_TARGET_NEON_I8, "neon_i8"},
        {SIMSIMD_TARGET_NEON_F16, "neon_f16"},
        {SIMSIMD_TARGET_NEON_BF16, "neon_bf16"},
        {SIMSIMD_TARGET_SVE, "sve"},
        {SIMSIMD_TARGET_SVE_I8, "sve_i8"},
        {SIMSIMD_TARGET_SVE_F16, "sve_f16"},
        {SIMSIMD_TARGET_SVE_BF16, "sve_bf16"},
        {SIMSIMD_TARGET_SVE2, "sve2"},
    };
    int const cap_count = sizeof(compiled_capabilities) / sizeof(compiled_capabilities[0]);

    // Count compiled capabilities
    int compiled_count = 0;
    for (int i = 0; i < cap_count; i++)
        if (compiled_capabilities[i].compiled)
            compiled_count++;

    // Create Java string array
    jclass stringClass = env->FindClass("java/lang/String");
    jobjectArray result = env->NewObjectArray(compiled_count, stringClass, nullptr);

    int index = 0;
    for (int i = 0; i < cap_count; i++) {
        if (compiled_capabilities[i].compiled) {
            jstring capName = env->NewStringUTF(compiled_capabilities[i].name);
            env->SetObjectArrayElement(result, index++, capName);
            env->DeleteLocalRef(capName);
        }
    }

    return result;
#else
    // If SimSIMD is not enabled, return only serial
    jclass stringClass = env->FindClass("java/lang/String");
    jobjectArray result = env->NewObjectArray(1, stringClass, nullptr);
    jstring serialCap = env->NewStringUTF("serial");
    env->SetObjectArrayElement(result, 0, serialCap);
    env->DeleteLocalRef(serialCap);
    return result;
#endif
}

JNIEXPORT jstring JNICALL Java_cloud_unum_usearch_Index_c_1library_1version(JNIEnv* env, jclass) {
    char version_str[32];
    std::snprintf(version_str, sizeof(version_str), "%d.%d.%d", //
                  USEARCH_VERSION_MAJOR, USEARCH_VERSION_MINOR, USEARCH_VERSION_PATCH);
    return env->NewStringUTF(version_str);
}

JNIEXPORT jboolean JNICALL Java_cloud_unum_usearch_Index_c_1uses_1dynamic_1dispatch(JNIEnv* env, jclass) {
#if USEARCH_USE_SIMSIMD
    return simsimd_uses_dynamic_dispatch() ? JNI_TRUE : JNI_FALSE;
#else
    return JNI_FALSE;
#endif
}
