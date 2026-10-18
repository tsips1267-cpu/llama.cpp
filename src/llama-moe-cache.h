#pragma once

#include "ggml-backend.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

struct llama_model;

// keeps the most recently used experts of host-resident MoE layers in a device buffer
// each layer has a slot map in host memory: when the scheduler copies it to the device, the copy callback uploads the missing experts
class llama_moe_cache {
public:
    // backends are all the backends of the context, each GPU gets its own cache of the given size for the layers assigned to it
    // hybrid: the GPU runs the cached experts while the CPU runs the misses, which are cached by access counts with this half-life in decode steps
    llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size,
                    bool hybrid = false, float half_life = 64.0f, int32_t prefetch_depth = 0, const float * split = nullptr, int32_t gpu_misses = 0);
    ~llama_moe_cache();

    // the device that caches layer il
    ggml_backend_t backend(int32_t il) const;

    // the slot map of layer il, if its experts can be read from the cache for n_tokens tokens, nullptr otherwise
    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const;

    // the experts of w in the cache, nullptr if w is not cached
    ggml_tensor * get_experts(const ggml_tensor * w) const;

    // hybrid mode: the GPU reads the misses from all-zero slots and the CPU runs them
    bool hybrid() const;

    // hybrid mode, in host memory and written by the lookup of layer il for the CPU half:
    // F32 [n_embd, n] the inputs of the layer, read by the lookup from the view of them named cpu_input_name(il)
    // I32 [n_expert_used, n] the expert id of each miss by position, -1 for a hit
    ggml_tensor * get_cpu_inp(int32_t il) const;
    ggml_tensor * get_cpu_ids(int32_t il) const;
    static std::string cpu_input_name(int32_t il);

    // hybrid mode: the first of the n_expert_used zero slots of layer il, a miss at position k of a selection reads slot zero + k
    int32_t get_zero_slot(int32_t il) const;

    // ggml_backend_sched copy callback, returns false if src is not a slot map
    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph);

    // for large batches: copy the experts of w that are in the cache or were prefetched, starting at expert e and up to expert last,
    // to the copy dst of w; returns the number of experts copied, 0 if expert e is in neither
    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last);

    // for large batches: called after the experts of w were copied, upload the experts of the next expert tensors
    // the prefetch uploads every uncached expert of a tensor, the plain upload only those a batch reads: measured, the prefetch
    // wins at 2048-token ubatches and loses below about 1024 tokens (144 vs 186 t/s at 512)
    static constexpr int64_t prefetch_min_batch = 1536;
    void prefetch_next(ggml_backend_t backend, const ggml_tensor * w, int64_t n_tokens);

    std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const;

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};

using llama_moe_cache_ptr = std::unique_ptr<llama_moe_cache>;
