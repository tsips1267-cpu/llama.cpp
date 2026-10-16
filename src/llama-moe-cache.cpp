#include "llama-moe-cache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-cpp.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace {

// LRU of the experts of a group of layers, the slot of each expert is kept in the slot map of its layer
struct moe_cache_lru {
    int32_t n_expert = 0;
    int32_t n_slots  = 0;

    std::vector<int32_t *> slot_map; // [n_layer] data of the slot maps, -1 if the expert is not cached
    std::vector<int32_t>   key_of;   // [n_slots] il*n_expert + expert, -1 if empty

    // doubly linked list of the slots, head is the least recently used
    std::vector<int32_t> prev;
    std::vector<int32_t> next;
    int32_t head = -1;
    int32_t tail = -1;

    std::vector<uint32_t> seen; // [n_expert]
    uint32_t seen_gen = 0;
    std::vector<int32_t> uniq;

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots) {
        this->n_expert = n_expert;
        this->n_slots  = n_slots;
        slot_map.assign(n_layer, nullptr);
        key_of.assign(n_slots, -1);
        prev.resize(n_slots);
        next.resize(n_slots);
        for (int32_t s = 0; s < n_slots; ++s) {
            prev[s] = s - 1;
            next[s] = s + 1 < n_slots ? s + 1 : -1;
        }
        head = 0;
        tail = n_slots - 1;
        seen.assign(n_expert, 0);
    }

    // move slot s to the tail (most recently used)
    void touch(int32_t s) {
        if (s == tail) {
            return;
        }
        if (prev[s] >= 0) {
            next[prev[s]] = next[s];
        } else {
            head = next[s];
        }
        prev[next[s]] = prev[s];

        prev[s] = tail;
        next[s] = -1;
        next[tail] = s;
        tail = s;
    }

    struct fill {
        int32_t expert;
        int32_t slot;
    };

    // give a slot to each expert selected by ids in layer il, the misses evict the least recently used experts
    // returns false if the ids select more distinct experts than there are slots
    bool plan(int32_t il, const int32_t * ids, size_t n_ids, std::vector<fill> & fills, size_t & n_hit) {
        fills.clear();
        n_hit = 0;

        if (++seen_gen == 0) {
            std::fill(seen.begin(), seen.end(), 0);
            seen_gen = 1;
        }
        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            if (seen[ids[i]] != seen_gen) {
                seen[ids[i]] = seen_gen;
                uniq.push_back(ids[i]);
            }
        }
        if (uniq.size() > (size_t) n_slots) {
            return false;
        }

        int32_t * slots = slot_map[il];

        // hits go to the tail first, so the head can be evicted below
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                touch(slots[e]);
                n_hit++;
            }
        }
        // sorted misses usually get consecutive slots, so the uploads can be merged
        std::sort(uniq.begin(), uniq.end());
        for (int32_t e : uniq) {
            if (slots[e] >= 0) {
                continue;
            }
            const int32_t s = head;
            if (key_of[s] >= 0) {
                slot_map[key_of[s] / n_expert][key_of[s] % n_expert] = -1;
            }
            key_of[s] = il*n_expert + e;
            slots[e] = s;
            touch(s);
            fills.push_back({ e, s });
        }
        return true;
    }
};

// hybrid mode: the misses run on the CPU, so a miss is only uploaded if it is hotter than the coldest cached expert
// heat is an access count that decays with a half-life of half_life steps, kept in log2 for all experts of the group
struct moe_cache_lfu {
    int32_t n_expert  = 0;
    int32_t n_slots   = 0;
    double  half_life = 64.0;

    std::vector<int32_t> slot_of; // [n_layer*n_expert] slot of each expert, -1 if not cached
    std::vector<int32_t> key_of;  // [n_slots] il*n_expert + expert, -1 if empty
    std::vector<double>  heat;    // [n_layer*n_expert] log2 of the decayed access count
    std::set<std::pair<double, int32_t>> by_heat; // (heat, slot) of the cached experts

    // experts used by the current lookup cannot be evicted by it
    std::vector<uint32_t> pinned;
    uint32_t gen = 0;

    int32_t n_filled = 0;
    double  t        = 0.0; // steps, one per lookup of the first layer of the group
    int32_t il_first = -1;

    std::vector<int32_t> uniq;
    std::vector<int32_t> miss;

    void init(int32_t n_layer, int32_t n_expert, int32_t n_slots, double half_life, int32_t il_first) {
        this->n_expert  = n_expert;
        this->n_slots   = n_slots;
        this->half_life = half_life;
        this->il_first  = il_first;
        slot_of.assign((size_t) n_layer*n_expert, -1);
        key_of.assign(n_slots, -1);
        heat.assign((size_t) n_layer*n_expert, -INFINITY);
        pinned.assign((size_t) n_layer*n_expert, 0);
    }

    void bump(int32_t key) {
        const double a = t/half_life;
        const double b = heat[key];
        const double v = b == -INFINITY ? a : std::max(a, b) + std::log2(1.0 + std::exp2(-std::fabs(a - b)));
        if (slot_of[key] >= 0) {
            by_heat.erase({ b, slot_of[key] });
            by_heat.insert({ v, slot_of[key] });
        }
        heat[key] = v;
    }

    // slots[e] gets the slot of each selected expert, -1 for a miss
    // admitted misses are uploaded now but only read from the next lookup on
    void plan(int32_t il, const int32_t * ids, size_t n_ids, int32_t * slots,
              std::vector<moe_cache_lru::fill> & fills, size_t & n_hit) {
        fills.clear();
        n_hit = 0;

        if (il == il_first) {
            t += 1.0;
        }
        if (++gen == 0) {
            std::fill(pinned.begin(), pinned.end(), 0);
            gen = 1;
        }

        uniq.clear();
        for (size_t i = 0; i < n_ids; ++i) {
            GGML_ASSERT(ids[i] >= 0 && ids[i] < n_expert);
            const int32_t key = il*n_expert + ids[i];
            if (pinned[key] != gen) {
                pinned[key] = gen;
                uniq.push_back(ids[i]);
            }
        }

        miss.clear();
        for (int32_t e : uniq) {
            const int32_t key = il*n_expert + e;
            bump(key);
            const int32_t s = slot_of[key];
            slots[e] = s;
            if (s >= 0) {
                n_hit++;
            } else {
                miss.push_back(e);
            }
        }

        // hottest misses first, so the loop can stop at the first one that is not hotter than the victim
        std::sort(miss.begin(), miss.end(), [&](int32_t a, int32_t b) { return heat[il*n_expert + a] > heat[il*n_expert + b]; });
        for (int32_t e : miss) {
            const int32_t key = il*n_expert + e;
            int32_t s = -1;
            if (n_filled < n_slots) {
                s = n_filled++;
            } else {
                auto it = by_heat.begin();
                while (it != by_heat.end() && pinned[key_of[it->second]] == gen) {
                    ++it;
                }
                if (it == by_heat.end() || it->first >= heat[key]) {
                    break;
                }
                s = it->second;
                slot_of[key_of[s]] = -1;
                by_heat.erase(it);
            }
            key_of[s]    = key;
            slot_of[key] = s;
            by_heat.insert({ heat[key], s });
            fills.push_back({ e, s });
        }
        // consecutive experts in consecutive slots are uploaded together
        std::sort(fills.begin(), fills.end(), [](const moe_cache_lru::fill & a, const moe_cache_lru::fill & b) { return a.expert < b.expert; });
    }
};

// gate, up, down or gate_up, down
static std::vector<ggml_tensor *> llama_moe_cache_layer_experts(const llama_layer & layer) {
    std::vector<ggml_tensor *> res;
    for (ggml_tensor * t : { layer.ffn_gate_up_exps, layer.ffn_gate_exps, layer.ffn_up_exps, layer.ffn_down_exps }) {
        if (t != nullptr) {
            res.push_back(t);
        }
    }
    return res;
}

static bool llama_moe_cache_same_layout(const std::vector<ggml_tensor *> & a, const std::vector<ggml_tensor *> & b) {
    if (a.size() != b.size()) {
        return false;
    }
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i]->type != b[i]->type || !ggml_are_same_shape(a[i], b[i]) || a[i]->nb[2] != b[i]->nb[2]) {
            return false;
        }
    }
    return true;
}

static bool llama_moe_cache_is_host_weight(const ggml_tensor * t) {
    return t->buffer != nullptr &&
        ggml_backend_buffer_get_usage(t->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
        ggml_backend_buffer_is_host(t->buffer);
}

}

struct llama_moe_cache::impl {
    // a GPU with its own budget and banks, it caches the layers assigned to it
    struct device {
        ggml_backend_t backend;
        ggml_backend_buffer_type_t buft;
        size_t host_bytes = 0; // host experts of the layers it caches
        double split = 0.0;    // share of the budget

        // banks and their views
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        size_t buf_size = 0;

        // GPU misses: the admissions the GPU does not run now upload on a second backend of the device, for its stream
        ggml_backend_ptr admit;
    };

    // layers of the same device with the same expert tensor layout share the banks and the LRU of a group
    struct group {
        int32_t id;                       // device
        std::vector<ggml_tensor *> ref;   // expert tensors of the first layer
        std::vector<int32_t> layers;
        std::vector<ggml_tensor *> banks; // device storage of all slots, one per expert tensor
        size_t host_bytes = 0;
        int32_t n_slots = 0;
        moe_cache_lru lru;
        moe_cache_lfu lfu; // hybrid mode
    };

    struct layer {
        int32_t ig = -1;                    // -1 if the layer is not cached
        ggml_tensor * slot_map = nullptr;   // I32 [1, n_expert] in host memory
        // hybrid mode, in host memory and written by the lookup for the CPU half: the inputs of the layer and the misses by position
        ggml_tensor * cpu_inp = nullptr;    // F32 [n_embd, max_batch_hybrid]
        ggml_tensor * cpu_ids = nullptr;    // I32 [n_expert_used, max_batch_hybrid], the expert id of a miss, -1 for a hit
        std::vector<ggml_tensor *> experts; // host expert tensors, in the order of the banks
        // GPU misses: recorded after the admissions of the layer, its next lookup waits for it
        ggml_backend_event_ptr ev_admit;
        bool admit_pending = false;
    };

    struct binding {
        int32_t il;
        int32_t ip;           // index of the bank
        ggml_tensor * cached; // view of the bank used in place of the host experts
    };

    struct stats {
        size_t hits   = 0;
        size_t misses = 0;
        size_t bytes  = 0;
    };

    static constexpr int64_t max_batch = 32;

    // hybrid mode: batches up to this size run their misses on the CPU, the scheduler offloads larger ones (GGML_OP_OFFLOAD_MIN_BATCH)
    static constexpr int64_t max_batch_hybrid = 16;

    int32_t n_expert_used;

    // the misses run on the CPU while the GPU runs the hits, see moe_cache_lfu
    bool   hybrid    = false;
    double half_life = 64.0; // in steps
    size_t n_admit   = 0;

    // hybrid mode: up to this many misses per lookup and group are uploaded to transient slots and run on the GPU, see split_misses
    int32_t n_gpu_miss  = 0;
    size_t  n_gpu_missed = 0;
    size_t  n_gpu_missed_window = 0;
    std::vector<uint8_t> n_tok_of;  // [n_expert] tokens of each selected expert in the current lookup
    std::vector<int32_t> admit_slot; // [n_expert] slot of an expert admitted by the current lookup, -1 if none
    std::vector<moe_cache_lru::fill> fills_now; // uploaded on the compute backend, read by the current lookup

    stats stats_small; // up to 8 tokens per ubatch
    stats stats_large;
    stats stats_copy;  // experts copied from the cache for large batches
    stats stats_hybrid; // misses run on the CPU, bytes are the admitted experts
    stats stats_window; // stats_hybrid at the start of the current window of steps
    int32_t n_window = 0;
    int32_t il_last  = -1; // the last cached layer, which ends a step

    // large batches read every expert of a layer: the experts of the next expert tensors, in the order the splits
    // read them, are uploaded on a second stream while the current ones compute, see prefetch_next
    // the scheduler offloads the large batches of every layer to its first backend (op offload), so the prefetch
    // serves that device, for the layers cached on another device too
    struct prefetch_buf {
        ggml_context_ptr ctx;
        ggml_backend_buffer_ptr buf;
        ggml_tensor * t = nullptr;        // I8 over the whole buffer
        const ggml_tensor * w = nullptr;  // the host experts it holds or receives, null if free
        bool read  = false;               // the compute stream read it since the upload
        bool freed = false;               // ev_free was recorded
        std::vector<uint8_t> ready;       // per expert: uploaded
        ggml_backend_event_ptr ev_done;   // copy stream: the upload finished
        ggml_backend_event_ptr ev_free;   // compute stream: the last read finished
    };

    // the order of the expert tensors is learned from the large batches: a ubatch crops the last layer to its outputs, so that one
    // takes the small batch path, and the splits may read gate and up in either order
    struct prefetcher {
        int32_t        id      = -1;      // the device
        ggml_backend_t compute = nullptr; // its backend, which runs the large batches
        ggml_backend_ptr copy;            // a second backend of the device, for its stream
        std::vector<const ggml_tensor *> seq; // host experts in layer order, the order until one is learned
        std::unordered_map<const ggml_tensor *, const ggml_tensor *> next; // the tensor read after each, as last seen
        const ggml_tensor * last = nullptr;   // the last tensor a large batch read
        size_t bytes = 0;                     // per buffer, also without an allocation (memory breakdown)
        std::vector<prefetch_buf> bufs;
        std::vector<const ggml_tensor *> window;
    };

    // ubatches from this size on read nearly every expert of a layer, so the prefetch uploads all the uncached ones
    static constexpr int64_t prefetch_min_batch = 256;

    int32_t prefetch_depth = 0;
    prefetcher pf;
    std::unordered_map<const ggml_tensor *, int32_t> seq_pos; // host experts -> position in pf.seq
    stats stats_prefetch; // hits: experts copied from a prefetch buffer, bytes: uploaded by the prefetch

    std::vector<device> devices;
    std::vector<group> groups;
    std::vector<layer> layers;
    std::unordered_map<const ggml_tensor *, binding> bindings; // host experts -> cached experts
    std::unordered_map<const ggml_tensor *, int32_t> layer_of; // slot map -> layer

    std::vector<int32_t> ids;
    std::vector<moe_cache_lru::fill> fills;

    // slot maps in host memory
    ggml_context_ptr ctx_host;
    ggml_backend_buffer_ptr buf_host;
    size_t buf_host_size = 0;

    // views used by copy_experts
    ggml_context_ptr ctx_views;

    impl(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size,
         bool hybrid, float half_life, int32_t prefetch_depth, const float * split, int32_t gpu_misses) :
            n_expert_used(model.hparams.n_expert_used_max()), hybrid(hybrid), half_life(std::max(1.0f, half_life)),
            n_gpu_miss(hybrid ? std::max(0, gpu_misses) : 0),
            prefetch_depth(std::max(0, prefetch_depth)), layers(model.layers.size()) {
        if (!hybrid && gpu_misses > 0) {
            LLAMA_LOG_WARN("%s: GPU misses need the hybrid MoE cache, ignored\n", __func__);
        }
        for (size_t i = 0; i < backends.size(); ++i) {
            const auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backends[i]));
            if (dev_type == GGML_BACKEND_DEVICE_TYPE_GPU || dev_type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
                auto & d = devices.emplace_back();
                d.backend = backends[i];
                d.buft    = bufts[i];
            }
        }
        if (devices.empty()) {
            throw std::runtime_error("MoE cache requires a GPU backend");
        }
        // the prefetch and the GPU misses order their streams against the compute stream with events
        if (prefetch_depth > 0 || n_gpu_miss > 0) {
            for (const device & d : devices) {
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(ggml_backend_get_device(d.backend), &props);
                if (!props.caps.events) {
                    LLAMA_LOG_WARN("%s: %s has no events, the MoE cache prefetch and GPU misses are disabled\n", __func__, ggml_backend_name(d.backend));
                    prefetch_depth = 0;
                    n_gpu_miss     = 0;
                    break;
                }
            }
        }
        if (model.split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
            throw std::runtime_error("MoE cache does not support tensor parallelism");
        }
        if (model.hparams.n_expert == 0 || n_expert_used == 0) {
            throw std::runtime_error("MoE cache requires a MoE model");
        }

        // only cache layers that keep all of their experts in host memory, on the device the layer is assigned to
        // the trailing MTP layers run in a draft context, which has no MoE cache
        for (size_t il = 0; il + model.hparams.n_layer_nextn < model.layers.size(); ++il) {
            auto experts = llama_moe_cache_layer_experts(model.layers[il]);
            if (experts.empty() || !std::all_of(experts.begin(), experts.end(), llama_moe_cache_is_host_weight)) {
                continue;
            }
            const auto it_dev = std::find_if(devices.begin(), devices.end(), [&](const device & d) { return ggml_backend_get_device(d.backend) == model.dev_layer(il); });
            if (it_dev == devices.end()) {
                continue;
            }
            const int32_t id = (int32_t) (it_dev - devices.begin());
            auto it = std::find_if(groups.begin(), groups.end(), [&](const group & g) { return g.id == id && llama_moe_cache_same_layout(g.ref, experts); });
            if (it == groups.end()) {
                groups.emplace_back();
                it = groups.end() - 1;
                it->id  = id;
                it->ref = experts;
            }
            it->layers.push_back(il);
            for (const ggml_tensor * t : experts) {
                it->host_bytes         += ggml_nbytes(t);
                devices[id].host_bytes += ggml_nbytes(t);
            }
        }
        if (groups.empty()) {
            LLAMA_LOG_WARN("%s: no layer has all of its experts in host memory, MoE cache is disabled\n", __func__);
            return;
        }

        // one extra slot at the end, CUDA MMQ can read past the last expert
        // the hybrid mode adds n_expert_used all-zero slots before it, the misses read them on the GPU
        // one per position of the selection: mul_mat_id requires the experts of a token to be distinct
        const int32_t n_extra = hybrid ? n_expert_used + n_gpu_miss + 1 : 1;
        auto alloc_size = [&](const group & g, int32_t n_slots) {
            const size_t alignment = ggml_backend_buft_get_alignment(devices[g.id].buft);
            size_t res = 0;
            for (const ggml_tensor * t : g.ref) {
                res += GGML_PAD(t->nb[2]*(n_slots + n_extra), alignment);
            }
            return res;
        };

        // the budget is split among the devices with host experts as given, else like the layers: by the tensor split or by default by free memory
        const float * tensor_split = split != nullptr ? split : model.tensor_split();
        const bool split_by_free = tensor_split == nullptr ||
            std::all_of(tensor_split, tensor_split + model.n_devices(), [](float x) { return x == 0.0f; });
        double split_sum = 0.0;
        for (device & d : devices) {
            if (d.host_bytes == 0) {
                continue;
            }
            ggml_backend_dev_t dev = ggml_backend_get_device(d.backend);
            if (split_by_free) {
                size_t free;
                size_t total;
                ggml_backend_dev_memory(dev, &free, &total);
                d.split = (double) free;
            } else {
                const auto it = std::find_if(model.devices.begin(), model.devices.end(), [&](const llama_device & ld) { return ld.dev == dev; });
                GGML_ASSERT(it != model.devices.end());
                d.split = (double) tensor_split[it - model.devices.begin()];
            }
            split_sum += d.split;
        }
        if (split_sum == 0.0) {
            // the devices do not report their free memory
            for (device & d : devices) {
                d.split    = d.host_bytes > 0 ? 1.0 : 0.0;
                split_sum += d.split;
            }
        }

        // within a device the budget is split by the size of the experts, so each group caches the same fraction of its experts
        std::vector<size_t> n_tensors(devices.size(), 0);
        size_t n_tensors_host = 0;
        for (group & g : groups) {
            const device & d = devices[g.id];
            const int32_t n_expert  = g.ref[0]->ne[2];
            const size_t  budget    = (size_t) ((double) size*d.split/split_sum*g.host_bytes/d.host_bytes);
            const int32_t max_slots = g.layers.size()*n_expert;
            while (g.n_slots < max_slots && alloc_size(g, g.n_slots + 1) <= budget) {
                g.n_slots++;
            }
            if (g.n_slots < n_expert_used) {
                LLAMA_LOG_WARN("%s: MoE cache budget is too small for %zu layers, they are not cached\n", __func__, g.layers.size());
                g.n_slots = 0;
                continue;
            }
            if (hybrid) {
                g.lfu.init(model.layers.size(), n_expert, g.n_slots, half_life, g.layers[0]);
            } else {
                g.lru.init(model.layers.size(), n_expert, g.n_slots);
            }
            n_tensors[g.id] += g.ref.size()*(1 + g.layers.size());
            n_tensors_host  += g.layers.size()*(hybrid ? 3 : 1);
        }
        if (n_tensors_host == 0) {
            throw std::runtime_error("MoE cache is too small to hold the experts of one token");
        }

        auto init_ctx = [](size_t n_tensors) {
            ggml_init_params params = {
                /*.mem_size   =*/ n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ nullptr,
                /*.no_alloc   =*/ true,
            };
            ggml_context_ptr res(ggml_init(params));
            if (!res) {
                throw std::runtime_error("failed to create the MoE cache context");
            }
            return res;
        };
        for (size_t id = 0; id < devices.size(); ++id) {
            if (n_tensors[id] > 0) {
                devices[id].ctx = init_ctx(n_tensors[id]);
            }
        }
        ctx_host  = init_ctx(n_tensors_host);
        ctx_views = init_ctx(2);

        ggml_backend_buffer_type_t buft_host = ggml_backend_cpu_buffer_type();
        const size_t alignment_host = ggml_backend_buft_get_alignment(buft_host);

        for (size_t ig = 0; ig < groups.size(); ++ig) {
            group & g = groups[ig];
            if (g.n_slots == 0) {
                continue;
            }
            ggml_context * ctx = devices[g.id].ctx.get();
            for (const ggml_tensor * t : g.ref) {
                ggml_tensor * bank = ggml_new_tensor_3d(ctx, t->type, t->ne[0], t->ne[1], g.n_slots + n_extra);
                GGML_ASSERT(bank->nb[2] == t->nb[2]);
                ggml_format_name(bank, "moe_cache.%zu.%s", ig, t->name);
                g.banks.push_back(bank);
            }
            for (int32_t il : g.layers) {
                layer & l = layers[il];
                l.ig      = (int32_t) ig;
                l.experts = llama_moe_cache_layer_experts(model.layers[il]);
                for (size_t ip = 0; ip < l.experts.size(); ++ip) {
                    ggml_tensor * bank   = g.banks[ip];
                    ggml_tensor * cached = ggml_view_3d(ctx, bank, bank->ne[0], bank->ne[1], g.n_slots + (hybrid ? n_expert_used + n_gpu_miss : 0), bank->nb[1], bank->nb[2], 0);
                    ggml_format_name(cached, "moe_cache.%s", l.experts[ip]->name);
                    bindings[l.experts[ip]] = { il, (int32_t) ip, cached };
                }
                l.slot_map = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, 1, g.ref[0]->ne[2]);
                ggml_format_name(l.slot_map, "moe_cache.slot_map-%d", il);
                layer_of[l.slot_map] = il;
                buf_host_size += GGML_PAD(ggml_nbytes(l.slot_map), alignment_host);
                if (hybrid) {
                    l.cpu_inp = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_F32, model.hparams.n_embd, max_batch_hybrid);
                    l.cpu_ids = ggml_new_tensor_2d(ctx_host.get(), GGML_TYPE_I32, n_expert_used, max_batch_hybrid);
                    ggml_format_name(l.cpu_inp, "moe_cache.cpu_inp-%d", il);
                    ggml_format_name(l.cpu_ids, "moe_cache.cpu_ids-%d", il);
                    buf_host_size += GGML_PAD(ggml_nbytes(l.cpu_inp), alignment_host) + GGML_PAD(ggml_nbytes(l.cpu_ids), alignment_host);
                }
            }
            devices[g.id].buf_size += alloc_size(g, g.n_slots);
        }

        if (model.hparams.no_alloc) {
            // only used to measure the memory use, see llama_context::memory_breakdown
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_buft_alloc_buffer(d.buft, 0));
                for (ggml_tensor * t = ggml_get_first_tensor(d.ctx.get()); t != nullptr; t = ggml_get_next_tensor(d.ctx.get(), t)) {
                    t->buffer = d.buf.get();
                }
            }
            buf_host.reset(ggml_backend_buft_alloc_buffer(buft_host, 0));
            for (ggml_tensor * t = ggml_get_first_tensor(ctx_host.get()); t != nullptr; t = ggml_get_next_tensor(ctx_host.get(), t)) {
                t->buffer = buf_host.get();
            }
        } else {
            for (device & d : devices) {
                if (!d.ctx) {
                    continue;
                }
                d.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(d.ctx.get(), d.buft));
                if (!d.buf) {
                    throw std::runtime_error("failed to allocate the MoE cache buffers");
                }
                ggml_backend_buffer_clear(d.buf.get(), 0);
                d.buf_size = ggml_backend_buffer_get_size(d.buf.get());
            }
            buf_host.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx_host.get(), buft_host));
            if (!buf_host) {
                throw std::runtime_error("failed to allocate the MoE cache buffers");
            }
            ggml_backend_buffer_clear(buf_host.get(), 0xff); // all slots are -1
            buf_host_size = ggml_backend_buffer_get_size(buf_host.get());

            for (group & g : groups) {
                for (int32_t il : g.layers) {
                    if (!hybrid && layers[il].slot_map != nullptr) {
                        g.lru.slot_map[il] = (int32_t *) layers[il].slot_map->data;
                    }
                }
            }
        }

        // as weights, the ops that read the banks run on the device and the slot maps are copied with the copy callback
        for (device & d : devices) {
            if (d.buf) {
                ggml_backend_buffer_set_usage(d.buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        ggml_backend_buffer_set_usage(buf_host.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);

        for (size_t id = 0; id < devices.size(); ++id) {
            const device & d = devices[id];
            if (d.host_bytes == 0) {
                continue;
            }
            LLAMA_LOG_INFO("%s: %10s MoE cache size = %8.2f MiB for %.2f MiB of host experts\n", __func__,
                ggml_backend_buft_name(d.buft), d.buf_size/1024.0/1024.0, d.host_bytes/1024.0/1024.0);
            for (const group & g : groups) {
                if (g.id == (int32_t) id) {
                    LLAMA_LOG_INFO("%s: %2zu layers, %s: %5d slots (%.1f%%)%s\n", __func__,
                        g.layers.size(), ggml_type_name(g.ref.back()->type), g.n_slots, 100.0*g.n_slots/(g.layers.size()*g.ref[0]->ne[2]),
                        hybrid ? ", hybrid: misses run on the CPU" : "");
                }
            }
        }

        if (prefetch_depth > 0) {
            // sized also without an allocation, the memory breakdown counts the buffers
            init_prefetch(!model.hparams.no_alloc);
        }
        if (n_gpu_miss > 0 && !model.hparams.no_alloc) {
            init_gpu_misses();
        }
    }

    void init_gpu_misses() {
        for (size_t id = 0; id < devices.size(); ++id) {
            if (devices[id].buf) {
                devices[id].admit.reset(ggml_backend_dev_init(ggml_backend_get_device(devices[id].backend), nullptr));
                if (!devices[id].admit) {
                    throw std::runtime_error("failed to create the MoE cache admission backend");
                }
            }
        }
        for (layer & l : layers) {
            if (l.ig >= 0) {
                l.ev_admit.reset(ggml_backend_event_new(ggml_backend_get_device(devices[groups[l.ig].id].backend)));
                if (!l.ev_admit) {
                    throw std::runtime_error("the MoE cache GPU misses need backend events");
                }
            }
        }
        LLAMA_LOG_INFO("%s: hybrid: up to %d misses per lookup run on the GPU from transient slots\n", __func__, n_gpu_miss);
    }

    void init_prefetch(bool alloc) {
        // the devices are in the order of the scheduler's backends, op offload picks the first
        const int32_t id = 0;
        for (int32_t il = 0; il < (int32_t) layers.size(); ++il) {
            const layer & l = layers[il];
            if (l.ig < 0) {
                continue;
            }
            for (ggml_tensor * w : l.experts) {
                seq_pos[w] = (int32_t) pf.seq.size();
                pf.seq.push_back(w);
                pf.bytes = std::max(pf.bytes, ggml_nbytes(w));
            }
        }
        if (pf.seq.empty()) {
            return;
        }
        pf.id      = id;
        pf.compute = devices[id].backend;
        LLAMA_LOG_INFO("%s: %10s MoE prefetch for large batches: %d x %.2f MiB, %zu expert tensors\n", __func__,
            ggml_backend_buft_name(devices[id].buft), prefetch_depth, pf.bytes/1024.0/1024.0, pf.seq.size());
        if (!alloc) {
            return;
        }

        ggml_backend_dev_t dev = ggml_backend_get_device(devices[id].backend);
        pf.copy.reset(ggml_backend_dev_init(dev, nullptr));
        if (!pf.copy) {
            throw std::runtime_error("failed to create the MoE cache prefetch backend");
        }
        ggml_init_params params = {
            /*.mem_size   =*/ ggml_tensor_overhead(),
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        pf.bufs.resize(prefetch_depth);
        for (prefetch_buf & b : pf.bufs) {
            b.ctx.reset(ggml_init(params));
            b.t = ggml_new_tensor_1d(b.ctx.get(), GGML_TYPE_I8, (int64_t) pf.bytes);
            b.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(b.ctx.get(), devices[id].buft));
            if (!b.buf) {
                throw std::runtime_error("failed to allocate the MoE cache prefetch buffers");
            }
            b.ev_done.reset(ggml_backend_event_new(dev));
            b.ev_free.reset(ggml_backend_event_new(dev));
            if (!b.ev_done || !b.ev_free) {
                throw std::runtime_error("the MoE cache prefetch needs backend events");
            }
        }
    }

    static std::string cpu_input_name(int32_t il) {
        return "moe_cache.inp-" + std::to_string(il);
    }

    ggml_backend_t backend(int32_t il) const {
        return devices[groups[layers[il].ig].id].backend;
    }

    ~impl() {
        log_stats();
        // the uploads still running write to the banks and the prefetch buffers
        for (device & d : devices) {
            if (d.admit) {
                ggml_backend_synchronize(d.admit.get());
            }
        }
        if (pf.copy) {
            ggml_backend_synchronize(pf.copy.get());
        }
    }

    ggml_tensor * get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
        if (il < 0 || il >= (int32_t) layers.size() || layers[il].ig < 0) {
            return nullptr;
        }
        const layer & l = layers[il];

        // large batches use most experts of a layer, so they gain little from the cache and would evict the experts used in generation
        if (hybrid) {
            return n_tokens > 0 && n_tokens <= max_batch_hybrid ? l.slot_map : nullptr;
        }
        if (n_tokens == 0 || n_tokens > max_batch || std::min(n_tokens*n_expert_used, l.slot_map->ne[1]) > groups[l.ig].n_slots) {
            return nullptr;
        }
        return l.slot_map;
    }

    ggml_tensor * get_experts(const ggml_tensor * w) const {
        const auto it = bindings.find(w);
        return it != bindings.end() ? it->second.cached : nullptr;
    }

    int64_t copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
        const auto it = bindings.find(w);
        if (it == bindings.end()) {
            return 0;
        }
        const binding & b = it->second;
        const group & g = groups[layers[b.il].ig];

        // large batches only read the cache, so the experts used in generation stay in it
        // the cache of a layer is on its own device, op offload may run the batch on another
        const int32_t * slots = backend != devices[g.id].backend ? nullptr :
            hybrid ? g.lfu.slot_of.data() + (size_t) b.il*g.lfu.n_expert : g.lru.slot_map[b.il];
        if (slots == nullptr || slots[e] < 0) {
            return copy_prefetched(backend, w, dst, e, last, slots);
        }
        // the admissions of the layer may still be uploading on the admission backend
        layer & lb = layers[b.il];
        if (lb.admit_pending) {
            ggml_backend_event_wait(backend, lb.ev_admit.get());
            lb.admit_pending = false;
        }
        int64_t n = 1;
        while (e + n <= last && slots[e + n] == slots[e] + n) {
            n++;
        }

        ggml_tensor * bank = g.banks[b.ip];
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_view_3d(ctx_views.get(), bank, bank->ne[0], bank->ne[1], n, bank->nb[1], bank->nb[2], slots[e]*bank->nb[2]);
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst,  dst->ne[0],  dst->ne[1],  n, dst->nb[1],  dst->nb[2],  e*dst->nb[2]);
        ggml_backend_view_init(src_view);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_copy.hits  += n;
        stats_copy.bytes += ggml_nbytes(src_view);

        return n;
    }

    prefetch_buf * prefetch_find(const ggml_tensor * w) {
        for (prefetch_buf & b : pf.bufs) {
            if (b.w == w) {
                return &b;
            }
        }
        return nullptr;
    }

    // the tensor a large batch reads after w: as last seen, or the next one in layer order
    const ggml_tensor * prefetch_after(const ggml_tensor * w) const {
        const auto it = pf.next.find(w);
        if (it != pf.next.end()) {
            return it->second;
        }
        const auto ip = seq_pos.find(w);
        return ip == seq_pos.end() ? nullptr : pf.seq[(ip->second + 1) % pf.seq.size()];
    }

    // the uncached experts from e on that the prefetch uploaded, copied from its buffer
    int64_t copy_prefetched(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last,
                            const int32_t * slots) {
        if (backend != pf.compute) {
            return 0;
        }
        prefetch_buf * b = prefetch_find(w);
        if (b == nullptr || !b->ready[e]) {
            return 0;
        }
        int64_t n = 1;
        while (e + n <= last && b->ready[e + n] && (slots == nullptr || slots[e + n] < 0)) {
            n++;
        }

        if (!b->read) {
            ggml_backend_event_wait(backend, b->ev_done.get());
            b->read = true;
        }

        // the buffer holds the experts at the offsets of the host tensor, read them with its layout
        ggml_reset(ctx_views.get());
        ggml_tensor * src_view = ggml_new_tensor_3d(ctx_views.get(), dst->type, dst->ne[0], dst->ne[1], n);
        GGML_ASSERT(src_view->nb[1] == dst->nb[1] && src_view->nb[2] == dst->nb[2]);
        src_view->buffer = b->buf.get();
        src_view->data   = (uint8_t *) b->t->data + e*dst->nb[2];
        ggml_tensor * dst_view = ggml_view_3d(ctx_views.get(), dst, dst->ne[0], dst->ne[1], n, dst->nb[1], dst->nb[2], e*dst->nb[2]);
        ggml_backend_view_init(dst_view);
        ggml_backend_tensor_copy_async(backend, backend, src_view, dst_view);

        stats_prefetch.hits += n;

        return n;
    }

    void prefetch_release(prefetch_buf & b, ggml_backend_t backend) {
        if (b.read) {
            ggml_backend_event_record(b.ev_free.get(), backend);
            b.freed = true;
        }
        b.w    = nullptr;
        b.read = false;
    }

    // after a large batch read w: free its buffer and keep the next prefetch_depth expert tensors on their way
    void prefetch_next(ggml_backend_t backend, const ggml_tensor * w, int64_t n_tokens) {
        if (pf.bufs.empty() || backend != pf.compute || seq_pos.find(w) == seq_pos.end()) {
            return;
        }
        if (pf.last != nullptr && pf.last != w) {
            pf.next[pf.last] = w;
        }
        pf.last = w;

        if (prefetch_buf * b = prefetch_find(w)) {
            prefetch_release(*b, backend);
        }

        if (n_tokens < prefetch_min_batch) {
            return;
        }

        pf.window.clear();
        for (const ggml_tensor * q = prefetch_after(w); q != nullptr && q != w && (int32_t) pf.window.size() < prefetch_depth; q = prefetch_after(q)) {
            if (std::find(pf.window.begin(), pf.window.end(), q) != pf.window.end()) {
                break;
            }
            pf.window.push_back(q);
        }

        for (const ggml_tensor * q : pf.window) {
            if (prefetch_find(q) != nullptr) {
                continue;
            }
            // a free buffer, or one that holds a tensor outside of the window
            prefetch_buf * b = nullptr;
            for (prefetch_buf & c : pf.bufs) {
                if (c.w == nullptr || std::find(pf.window.begin(), pf.window.end(), c.w) == pf.window.end()) {
                    b = &c;
                    break;
                }
            }
            if (b == nullptr) {
                break;
            }
            prefetch_release(*b, backend);

            const binding & bd = bindings.at(q);
            const int32_t il = bd.il;
            // a layer cached on another device has all its experts uploaded
            const group & g = groups[layers[il].ig];
            const int32_t * slots = g.id != pf.id ? nullptr :
                hybrid ? g.lfu.slot_of.data() + (size_t) il*g.lfu.n_expert : g.lru.slot_map[il];
            const int64_t n_expert    = q->ne[2];
            const size_t  expert_size = q->nb[2];

            // the compute stream may still read the buffer
            if (b->freed) {
                ggml_backend_event_wait(pf.copy.get(), b->ev_free.get());
            }
            b->ready.assign(n_expert, 0);
            for (int64_t e0 = 0; e0 < n_expert; ) {
                if (slots != nullptr && slots[e0] >= 0) {
                    e0++;
                    continue;
                }
                int64_t e1 = e0 + 1;
                while (e1 < n_expert && (slots == nullptr || slots[e1] < 0)) {
                    e1++;
                }
                ggml_backend_tensor_set_async(pf.copy.get(), b->t, (const uint8_t *) q->data + e0*expert_size, e0*expert_size, (e1 - e0)*expert_size);
                std::fill(b->ready.begin() + e0, b->ready.begin() + e1, 1);
                stats_prefetch.bytes += (e1 - e0)*expert_size;
                e0 = e1;
            }
            ggml_backend_event_record(b->ev_done.get(), pf.copy.get());
            b->w = q;
        }
    }

    // the experts of fills to their slots in the banks, consecutive experts going to consecutive slots together
    size_t upload(ggml_backend_t backend, const layer & l, const group & g, const std::vector<moe_cache_lru::fill> & f) {
        size_t bytes = 0;
        for (size_t ip = 0; ip < l.experts.size(); ++ip) {
            const ggml_tensor * w    = l.experts[ip];
            ggml_tensor       * bank = g.banks[ip];
            const size_t expert_size = w->nb[2];
            for (size_t i = 0; i < f.size();) {
                size_t n = 1;
                while (i + n < f.size() && f[i + n].expert == f[i].expert + (int32_t) n && f[i + n].slot == f[i].slot + (int32_t) n) {
                    n++;
                }
                ggml_backend_tensor_set_async(backend, bank, (const uint8_t *) w->data + f[i].expert*expert_size, f[i].slot*expert_size, n*expert_size);
                bytes += n*expert_size;
                i += n;
            }
        }
        return bytes;
    }

    // cost model of a lookup in us: the CPU half runs the misses, one expert per token at a time; the GPU half first uploads
    // its misses, then runs every expert it has; measured with 8 threads of a Zen 2 Threadripper, PCIe 4.0 x16
    static constexpr double cpu_us_base     = 25.0;
    static constexpr double cpu_us_token    = 60.0;
    static constexpr double gpu_us_token    = 4.0;
    static constexpr double pcie_bytes_us   = 20e3;

    // move misses to transient slots on the GPU while that shortens the longer half, the experts with the most tokens first:
    // the CPU time of an expert grows with its tokens, its upload does not; the admitted experts stay on the CPU, they are uploaded
    // to their cache slots anyway
    void split_misses(const layer & l, group & g, int32_t * slots) {
        fills_now.clear();
        const int32_t n_expert = g.lfu.n_expert;
        n_tok_of.resize(n_expert, 0);
        admit_slot.resize(n_expert, -1);
        for (const int32_t e : ids) {
            n_tok_of[e]++;
        }
        for (const auto & f : fills) {
            admit_slot[f.expert] = f.slot;
        }

        size_t expert_bytes = 0;
        for (const ggml_tensor * w : l.experts) {
            expert_bytes += w->nb[2];
        }
        const double up = expert_bytes/pcie_bytes_us;

        double t_cpu = 0.0;
        double t_gpu = 0.0;
        std::vector<int32_t> & cand = g.lfu.miss;
        cand.clear();
        for (const int32_t e : g.lfu.uniq) {
            if (slots[e] >= 0) {
                t_gpu += gpu_us_token*n_tok_of[e];
            } else {
                t_cpu += cpu_us_token*n_tok_of[e];
                if (admit_slot[e] < 0) {
                    cand.push_back(e);
                }
            }
        }
        t_cpu += t_cpu > 0.0 ? cpu_us_base : 0.0;
        std::sort(cand.begin(), cand.end(), [&](int32_t a, int32_t b) { return n_tok_of[a] > n_tok_of[b]; });

        const int32_t tmp0 = g.n_slots + n_expert_used;
        for (const int32_t e : cand) {
            if ((int32_t) fills_now.size() == n_gpu_miss) {
                break;
            }
            const double tc = cpu_us_token*n_tok_of[e];
            const double tg = up + gpu_us_token*n_tok_of[e];
            if (t_gpu + tg >= t_cpu - tc) {
                break;
            }
            const int32_t s = tmp0 + (int32_t) fills_now.size();
            slots[e] = s;
            fills_now.push_back({ e, s });
            t_gpu += tg;
            t_cpu -= tc;
        }
        n_gpu_missed += fills_now.size();

        for (const int32_t e : g.lfu.uniq) {
            n_tok_of[e]   = 0;
            admit_slot[e] = -1;
        }
    }

    bool copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
        const auto it = layer_of.find(src);
        if (it == layer_of.end()) {
            return false;
        }
        const int32_t il = it->second;
        layer & l = layers[il];
        group & g = groups[l.ig];

        GGML_ASSERT(backend == devices[g.id].backend);

        // the get_rows that looks up the slots of the selected experts
        const int n_nodes = ggml_graph_n_nodes(graph);
        const ggml_tensor * lookup = nullptr;
        for (int i = 0; i < n_nodes && lookup == nullptr; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node->op == GGML_OP_GET_ROWS && node->src[0] == dst) {
                lookup = node;
            }
        }
        GGML_ASSERT(lookup != nullptr);

        // the selected experts must be computed in an earlier split
        // the scheduler starts a new split at the lookup because it reads a host weight, but only if the split already has inputs
        const ggml_tensor * sel = lookup->src[1];
        for (int i = 0; i < n_nodes; ++i) {
            const ggml_tensor * node = ggml_graph_node(graph, i);
            if (node == sel || node == sel->view_src) {
                GGML_ABORT("the experts of layer %d are selected in the same split as their MoE cache lookup", il);
            }
        }
        GGML_ASSERT(ggml_is_contiguous(sel));

        ids.resize(ggml_nelements(sel));
        ggml_backend_tensor_get_async(backend, sel, ids.data(), 0, ggml_nbytes(sel));

        // hybrid mode: the inputs of the CPU half come with the ids, see llama_moe_cache::cpu_input_name
        if (hybrid) {
            const ggml_tensor * inp = nullptr;
            for (int i = 0; i < n_nodes && inp == nullptr; ++i) {
                const ggml_tensor * node = ggml_graph_node(graph, i);
                if (node->view_src != nullptr && strcmp(node->name, cpu_input_name(il).c_str()) == 0) {
                    inp = node;
                }
            }
            GGML_ASSERT(inp != nullptr && ggml_is_contiguous(inp) && inp->type == GGML_TYPE_F32);
            GGML_ASSERT(ggml_nelements(inp) <= ggml_nelements(l.cpu_inp) && ggml_nelements(sel) <= ggml_nelements(l.cpu_ids));
            ggml_backend_tensor_get_async(backend, inp, l.cpu_inp->data, 0, ggml_nbytes(inp));
        }
        ggml_backend_synchronize(backend);

        size_t n_hit  = 0;
        size_t n_miss = 0;
        if (hybrid) {
            // the misses go to the CPU by position, on the GPU they read the zero slots after the last one
            int32_t * slots = (int32_t *) src->data;
            g.lfu.plan(il, ids.data(), ids.size(), slots, fills, n_hit);
            n_miss = g.lfu.uniq.size() - n_hit;
            if (n_gpu_miss > 0) {
                // the hits may read slots that earlier lookups of this layer admitted on the admission backend
                if (l.admit_pending) {
                    ggml_backend_event_wait(backend, l.ev_admit.get());
                    l.admit_pending = false;
                }
                split_misses(l, g, slots);
            }
            int32_t * cpu_ids = (int32_t *) l.cpu_ids->data;
            for (size_t i = 0; i < ids.size(); ++i) {
                cpu_ids[i] = slots[ids[i]] < 0 ? ids[i] : -1;
            }
        } else {
            if (!g.lru.plan(il, ids.data(), ids.size(), fills, n_hit)) {
                GGML_ABORT("the MoE cache is too small for the experts selected in layer %d", il);
            }
            n_miss = fills.size();
        }

        // upload the missing experts: the GPU misses now, before the hits read them, the admissions on their own backend so that the hits
        // do not wait for them, they are read from the next lookup of the layer on
        size_t bytes = 0;
        if (n_gpu_miss > 0) {
            bytes += upload(backend, l, g, fills_now);
            if (!fills.empty()) {
                bytes += upload(devices[g.id].admit.get(), l, g, fills);
                ggml_backend_event_record(l.ev_admit.get(), devices[g.id].admit.get());
                l.admit_pending = true;
            }
        } else {
            bytes += upload(backend, l, g, fills);
        }

        stats & st = hybrid ? stats_hybrid : ids.size() <= (size_t) 8*n_expert_used ? stats_small : stats_large;
        st.hits   += n_hit;
        st.misses += n_miss;
        st.bytes  += bytes;
        n_admit   += hybrid ? fills.size() : 0;

        // the hit rate of the recent lookups, the totals at exit mix every workload the server saw
        if (il_last < 0) {
            for (int32_t i = 0; i < (int32_t) layers.size(); ++i) {
                il_last = layers[i].ig >= 0 ? i : il_last;
            }
        }
        if (hybrid && il == il_last && ++n_window == 256) {
            const stats & w = stats_window;
            const size_t  n = st.hits - w.hits + st.misses - w.misses;
            LLAMA_LOG_INFO("llama_moe_cache: hybrid, last %d steps: hit rate = %.2f%%, %.1f misses per step, %.1f of them on the GPU, uploaded %.1f MiB per step\n",
                n_window, 100.0*(st.hits - w.hits)/std::max<size_t>(n, 1), double(st.misses - w.misses)/n_window,
                double(n_gpu_missed - n_gpu_missed_window)/n_window, (st.bytes - w.bytes)/1024.0/1024.0/n_window);
            n_gpu_missed_window = n_gpu_missed;
            stats_window = st;
            n_window     = 0;
        }

        // the next copy synchronizes the backend before it changes the slot map again
        ggml_backend_tensor_set_async(backend, dst, src->data, 0, ggml_nbytes(src));

        return true;
    }

    void log_stats() const {
        auto log = [](const char * name, const stats & st) {
            const size_t n = st.hits + st.misses;
            if (n == 0) {
                return;
            }
            LLAMA_LOG_INFO("llama_moe_cache: %s: hits = %zu, misses = %zu, hit rate = %.2f%%, uploaded = %.2f MiB\n",
                name, st.hits, st.misses, 100.0*st.hits/n, st.bytes/1024.0/1024.0);
        };
        log("ubatch <= 8", stats_small);
        log("ubatch  > 8", stats_large);
        if (stats_hybrid.hits + stats_hybrid.misses > 0) {
            const size_t n = stats_hybrid.hits + stats_hybrid.misses;
            LLAMA_LOG_INFO("llama_moe_cache: hybrid: hits = %zu, misses = %zu (%zu on the GPU), hit rate = %.2f%%, admitted = %zu, uploaded = %.2f MiB\n",
                stats_hybrid.hits, stats_hybrid.misses, n_gpu_missed, 100.0*stats_hybrid.hits/n, n_admit, stats_hybrid.bytes/1024.0/1024.0);
        }
        if (stats_copy.hits > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the cache, %.2f MiB\n", stats_copy.hits, stats_copy.bytes/1024.0/1024.0);
        }
        if (stats_prefetch.bytes > 0) {
            LLAMA_LOG_INFO("llama_moe_cache: large batches: %zu experts copied from the prefetch, %.2f MiB prefetched\n", stats_prefetch.hits, stats_prefetch.bytes/1024.0/1024.0);
        }
    }
};

llama_moe_cache::llama_moe_cache(const llama_model & model, const std::vector<ggml_backend_t> & backends, const std::vector<ggml_backend_buffer_type_t> & bufts, size_t size,
                                 bool hybrid, float half_life, int32_t prefetch_depth, const float * split, int32_t gpu_misses) :
    pimpl(new impl(model, backends, bufts, size, hybrid, half_life, prefetch_depth, split, gpu_misses)) {
}

llama_moe_cache::~llama_moe_cache() = default;

ggml_backend_t llama_moe_cache::backend(int32_t il) const {
    return pimpl->backend(il);
}

ggml_tensor * llama_moe_cache::get_slot_map(int32_t il, int64_t n_tokens, int64_t n_expert_used) const {
    return pimpl->get_slot_map(il, n_tokens, n_expert_used);
}

ggml_tensor * llama_moe_cache::get_experts(const ggml_tensor * w) const {
    return pimpl->get_experts(w);
}

bool llama_moe_cache::hybrid() const {
    return pimpl->hybrid;
}

ggml_tensor * llama_moe_cache::get_cpu_inp(int32_t il) const {
    return il >= 0 && il < (int32_t) pimpl->layers.size() ? pimpl->layers[il].cpu_inp : nullptr;
}

ggml_tensor * llama_moe_cache::get_cpu_ids(int32_t il) const {
    return il >= 0 && il < (int32_t) pimpl->layers.size() ? pimpl->layers[il].cpu_ids : nullptr;
}

std::string llama_moe_cache::cpu_input_name(int32_t il) {
    return impl::cpu_input_name(il);
}

bool llama_moe_cache::copy(ggml_backend_t backend, const ggml_tensor * src, ggml_tensor * dst, ggml_cgraph * graph) {
    return pimpl->copy(backend, src, dst, graph);
}

int64_t llama_moe_cache::copy_experts(ggml_backend_t backend, const ggml_tensor * w, ggml_tensor * dst, int64_t e, int64_t last) {
    return pimpl->copy_experts(backend, w, dst, e, last);
}

void llama_moe_cache::prefetch_next(ggml_backend_t backend, const ggml_tensor * w, int64_t n_tokens) {
    pimpl->prefetch_next(backend, w, n_tokens);
}

std::map<ggml_backend_buffer_type_t, size_t> llama_moe_cache::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, size_t> res;
    for (const auto & d : pimpl->devices) {
        if (d.buf) {
            res[ggml_backend_buffer_get_type(d.buf.get())] += d.buf_size;
        }
    }
    if (pimpl->buf_host) {
        res[ggml_backend_buffer_get_type(pimpl->buf_host.get())] += pimpl->buf_host_size;
    }
    // counted also when the memory is only measured (no_alloc), the buffers are not allocated then
    if (pimpl->pf.id >= 0) {
        res[pimpl->devices[pimpl->pf.id].buft] += (size_t) pimpl->prefetch_depth*pimpl->pf.bytes;
    }
    return res;
}

int32_t llama_moe_cache::get_zero_slot(int32_t il) const {
    GGML_ASSERT(il >= 0 && il < (int32_t) pimpl->layers.size() && pimpl->layers[il].ig >= 0);
    return pimpl->groups[pimpl->layers[il].ig].n_slots;
}
