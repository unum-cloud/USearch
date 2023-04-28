/**
 *  @file javascript.cpp
 *  @author Ashot Vardanian
 *  @brief JavaScript bindings for Unum Index.
 *  @date 2023-04-26
 *
 *  @copyright Copyright (c) 2023
 *
 *  @see NodeJS docs: https://nodejs.org/api/addons.html#hello-world
 *
 */
#include <napi.h>
#include <node_api.h>

#include <usearch/usearch.hpp>

using namespace unum;

using real_t = float;
using label_t = std::uint32_t;
using neighbor_t = std::uint32_t;
using distance_t = float;
using dim_t = usearch::dim_t;
using distance_function_t = distance_t (*)(real_t const*, real_t const*, dim_t, dim_t);
using index_t = usearch::index_gt<distance_function_t, label_t, neighbor_t, real_t>;

template <typename distance_function_at>
static distance_t type_punned_distance_function(real_t const* a, real_t const* b, dim_t a_dim, dim_t b_dim) noexcept {
    return distance_function_at{}(a, b, a_dim, b_dim);
}

class Index : public Napi::ObjectWrap<Index> {
  public:
    static Napi::Object Init(Napi::Env env, Napi::Object exports);
    Index(Napi::CallbackInfo const& ctx);

  private:
    Napi::Value GetDim(Napi::CallbackInfo const& ctx);
    Napi::Value GetSize(Napi::CallbackInfo const& ctx);
    Napi::Value GetCapacity(Napi::CallbackInfo const& ctx);
    Napi::Value GetConnectivity(Napi::CallbackInfo const& ctx);

    void Add(Napi::CallbackInfo const& ctx);
    Napi::Value Search(Napi::CallbackInfo const& ctx);

    std::unique_ptr<index_t> native_;
};

Napi::Object Index::Init(Napi::Env env, Napi::Object exports) {
    Napi::Function func = DefineClass( //
        env, "Index",
        {
            InstanceMethod("dim", &Index::GetDim),
            InstanceMethod("size", &Index::GetSize),
            InstanceMethod("capacity", &Index::GetCapacity),
            InstanceMethod("connectivity", &Index::GetConnectivity),
            InstanceMethod("add", &Index::Add),
            InstanceMethod("search", &Index::Search),
        });

    Napi::FunctionReference* constructor = new Napi::FunctionReference();
    *constructor = Napi::Persistent(func);
    env.SetInstanceData(constructor);

    exports.Set("Index", func);
    return exports;
}

Index::Index(Napi::CallbackInfo const& ctx) : Napi::ObjectWrap<Index>(ctx) {
    Napi::Env env = ctx.Env();

    int length = ctx.Length();
    if (length > 0 != (length == 1 && ctx[0].IsObject())) {
        Napi::TypeError::New(
            env, "Pass args as named objects: dim: uint, capacity: uint, metric: [ip, cos, l2_sq, haversine]")
            .ThrowAsJavaScriptException();
        return;
    }

    usearch::config_t config;
    distance_function_t distance_function = &type_punned_distance_function<usearch::cos_gt<real_t>>;

    if (length) {
        Napi::Object params = ctx[0].As<Napi::Object>();
        if (params.Has("dim"))
            config.dim = params.Get("dim").As<Napi::Number>().Uint32Value();
        if (params.Has("capacity"))
            config.max_elements = params.Get("capacity").As<Napi::Number>().Uint32Value();
        if (params.Has("connectivity"))
            config.connectivity = params.Get("connectivity").As<Napi::Number>().Uint32Value();
        if (params.Has("metric")) {
            std::string name = params.Get("metric").As<Napi::String>().Utf8Value();
            if (name == "l2_sq" || name == "euclidean_sq") {
                distance_function = type_punned_distance_function<usearch::l2_squared_gt<real_t>>;
            } else if (name == "ip" || name == "inner" || name == "dot") {
                distance_function = type_punned_distance_function<usearch::ip_gt<real_t>>;
            } else if (name == "cos" || name == "angular") {
                distance_function = type_punned_distance_function<usearch::cos_gt<real_t>>;
            } else if (name == "haversine") {
                distance_function = type_punned_distance_function<usearch::haversine_gt<real_t>>;
            } else {
                Napi::TypeError::New(env, "Supported metrics are: [ip, cos, l2_sq, haversine]")
                    .ThrowAsJavaScriptException();
                return;
            }
        }
    }

    native_ = std::make_unique<index_t>(config);
    native_->adjust_metric(distance_function);
}

Napi::Value Index::GetDim(Napi::CallbackInfo const& ctx) { return Napi::Number::New(ctx.Env(), native_->dim()); }
Napi::Value Index::GetSize(Napi::CallbackInfo const& ctx) { return Napi::Number::New(ctx.Env(), native_->size()); }
Napi::Value Index::GetConnectivity(Napi::CallbackInfo const& ctx) {
    return Napi::Number::New(ctx.Env(), native_->connectivity());
}
Napi::Value Index::GetCapacity(Napi::CallbackInfo const& ctx) {
    return Napi::Number::New(ctx.Env(), native_->capacity());
}

void Index::Add(Napi::CallbackInfo const& ctx) {
    Napi::Env env = ctx.Env();
    if (ctx.Length() < 2 || !ctx[0].IsNumber() || !ctx[1].IsTypedArray()) {
        Napi::TypeError::New(env, "Expects an integral label and a float vector").ThrowAsJavaScriptException();
        return;
    }

    Napi::Number label_js = ctx[0].As<Napi::Number>();
    Napi::Float32Array vector_js = ctx[1].As<Napi::Float32Array>();

    label_t label = label_js.Uint32Value();
    real_t const* vector = vector_js.Data();
    dim_t dim = static_cast<dim_t>(vector_js.ElementLength());

    if (native_->size() + 1 >= native_->capacity())
        native_->reserve(usearch::ceil2(native_->size() + 1));
    native_->add(label, vector, dim);
}

Napi::Value Index::Search(Napi::CallbackInfo const& ctx) {
    Napi::Env env = ctx.Env();
    if (ctx.Length() < 2 || !ctx[0].IsTypedArray() || !ctx[1].IsNumber()) {
        Napi::TypeError::New(env, "Expects a float vector and the number of wanted results")
            .ThrowAsJavaScriptException();
        return {};
    }

    Napi::Float32Array vector_js = ctx[0].As<Napi::Float32Array>();
    Napi::Number wanted_js = ctx[1].As<Napi::Number>();

    using match_t = std::pair<label_t, distance_t>;
    std::vector<match_t> results;

    real_t const* vector = vector_js.Data();
    dim_t dim = static_cast<dim_t>(vector_js.ElementLength());
    uint32_t wanted = wanted_js.Uint32Value();

    results.reserve(wanted);
    native_->search(vector, dim, wanted, [&](label_t l, distance_t d) { results.push_back({l, d}); });
    std::sort(results.begin(), results.end(), [](match_t const& a, match_t const& b) { return a.second < b.second; });

    Napi::Uint32Array matches_js = Napi::Uint32Array::New(env, results.size());
    Napi::Float32Array distances_js = Napi::Float32Array::New(env, results.size());
    for (size_t i = 0; i != results.size(); ++i) {
        matches_js[i] = results[i].first;
        distances_js[i] = results[i].second;
    }

    Napi::Object result_js = Napi::Object::New(env);
    result_js.Set("labels", matches_js);
    result_js.Set("distances", distances_js);
    return result_js;
}

Napi::Object InitAll(Napi::Env env, Napi::Object exports) { return Index::Init(env, exports); }

NODE_API_MODULE(usearch, InitAll)
