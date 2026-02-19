#pragma once
#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <vector>
#include <atomic>
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>
#include "bfloat16.hpp"
#include "CV2_Tokenizer.hpp"  // CV2 Tokenizer API (Init with bos/eos, Encode with ImageInfo)
#include "LLMEmbedSelector.hpp"
#include "ax_model_runner/ax_model_runner_ax650.hpp"
#include "ax_cmm_utils.hpp"
#include "cqdm.h"
#include "timer.hpp"
#include "opencv2/opencv.hpp"
// AXCL host mode: no direct ax_sys_api, use axcl_manager instead
#include "axcl_manager.h"
#include "utils/sampling.hpp"
#include "utils/utils.hpp"
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif

using SpeechToken = int;
// The container for speech tokens. std::deque is efficient for front/back operations.
using TokenBuffer = std::deque<SpeechToken>;

typedef void (*LLMRuningCallback)(int *p_token, int n_token, float token_per_sec, void *reserve);

struct LLMAttrType
{
    std::string template_filename_axmodel = "qwen2_p128_l%d_together.axmodel";
    int axmodel_num = 24;  // CosyVoice3: 24 layers (CV2: 22)

    std::string filename_post_axmodel = "qwen2_post.axmodel";
    std::string filename_decoder_axmodel ;

    int prefill_token_num = 96; // auto calc
    int prefill_max_token_num = 512;
    std::vector<int> prefill_max_kv_cache_num_grp;
    int precompute_len = 0;
    int prefill_grpid = -1;

    TokenizerType tokenizer_type = TKT_HTTP;
    std::string filename_tokenizer_model = "http://127.0.0.1:12345";
    bool b_bos = false, b_eos = false;
    std::string filename_tokens_embed = "model.embed_tokens.weight.bfloat16.bin";
    std::string filename_llm_embed = "llm.llm_embedding.bfloat16.bin";
    std::string filename_speech_embed = "llm.speech_embedding.bfloat16.bin";
    int tokens_embed_num = 151936;
    int tokens_embed_size = 896;

    int llm_embed_num = 2;
    int llm_embed_size = 896;
    int speech_embed_num = 6761;  // CosyVoice3: 6761 (CV2: 6564) - embedding table size
    int speech_embed_size = 896;

    // CosyVoice3: Explicit EOS token ID (replaces speech_embed_num-3 formula)
    // CV2: EOS=6561 (computed as 6564-3)
    // CV3: Any token >= speech_token_size (6561) is a stop token
    //      SOS=6561, EOS=6562, task_id=6563, fill=6564, ...6760
    //      stop_token_ids = [6561..6760] (200 tokens)
    //      Primary check: token >= 6561 means stop
    int eos_token_id = 6561;

    int max_token_len = 127; // auto calc

    int kv_cache_num = 1024; // auto calc
    int kv_cache_size = 256; // auto calc

    bool b_use_mmap_load_embed = false;
    bool b_dynamic_load_axmodel_layer = false;

    bool b_use_mmap_load_layer = true;

    bool b_use_topk = false;

    int dev_id = 0;  // AXCL device ID (PCIe)

    // Sampling parameters
    float temperature = 1.0f;
    int top_k = 25;

    // CPU decoder: bypass NPU U16 quantization by running decoder on CPU
    // The decoder is just Linear(896, 6761) - ~6M multiply-adds, ~2ms on ARM
    std::string filename_decoder_weight;  // float32 binary (6761 * 896 * 4 bytes)
    bool b_use_cpu_decoder = false;

    // bool b_live_print = true;
    LLMRuningCallback runing_callback = nullptr;
    void *reserve = nullptr;

};

class LLM
{
private:
    std::shared_ptr<BaseTokenizer> tokenizer;
    LLaMaEmbedSelector embed_selector;
    LLaMaEmbedSelector llm_embed_selector;
    LLaMaEmbedSelector speech_embed_selector;

    LLMAttrType _attr;

    struct LLMLayer
    {
        ax_runner_ax650 layer;
        std::string filename;
        MMap layer_buffer;
        std::vector<char> layer_buffer_vec;
    };

    std::vector<LLMLayer> llama_layers;
    ax_runner_ax650 llama_post;
    ax_runner_ax650 llm_decoder;

    // CPU decoder weights: Linear(896, 6761), stored as [6761, 896] row-major
    std::vector<float> cpu_decoder_weight;
    static const int DECODER_IN_DIM = 896;
    static const int DECODER_OUT_DIM = 6761;

    // Debug: dump CPU decoder logit stats for every iteration (CSV format)
    int cpu_dump_count = 0;
    void dump_cpu_decoder_logits(const std::vector<float> &scores, int iter_id) {
        FILE *df = fopen("cpu_decoder_logits.csv", cpu_dump_count == 0 ? "w" : "a");
        if (!df) return;
        if (cpu_dump_count == 0) {
            fprintf(df, "iter,best_speech_id,best_speech_val,best_eos_id,best_eos_val,gap,mean,eos_in_top25\n");
        }
        // Find best speech and best EOS logits
        float best_speech = -1e9, best_eos = -1e9;
        int best_speech_id = 0, best_eos_id = 0;
        float sum = 0;
        for (int i = 0; i < (int)scores.size(); i++) {
            sum += scores[i];
            if (i < _attr.eos_token_id) {
                if (scores[i] > best_speech) { best_speech = scores[i]; best_speech_id = i; }
            } else {
                if (scores[i] > best_eos) { best_eos = scores[i]; best_eos_id = i; }
            }
        }
        // Count EOS in top 25
        std::vector<std::pair<float, int>> indexed;
        indexed.reserve(scores.size());
        for (int i = 0; i < (int)scores.size(); i++) indexed.push_back({scores[i], i});
        std::partial_sort(indexed.begin(), indexed.begin() + std::min(25, (int)indexed.size()),
                         indexed.end(), std::greater<std::pair<float,int>>());
        int eos_in_top25 = 0;
        for (int i = 0; i < 25 && i < (int)indexed.size(); i++) {
            if (indexed[i].second >= _attr.eos_token_id) eos_in_top25++;
        }
        float gap = best_speech - best_eos;
        fprintf(df, "%d,%d,%.4f,%d,%.4f,%.4f,%.6f,%d\n",
                iter_id, best_speech_id, best_speech, best_eos_id, best_eos,
                gap, sum / scores.size(), eos_in_top25);
        fclose(df);
        cpu_dump_count++;
    }

    // CPU decoder: output = input @ weight.T (no bias)
    // NEON-optimized: processes 4 floats per cycle on ARM A76/A55
    void cpu_decoder_forward(const float *input, float *output) {
        // input: [DECODER_IN_DIM=896], output: [DECODER_OUT_DIM=6761]
        // weight: [DECODER_OUT_DIM, DECODER_IN_DIM] row-major
        // 896 is divisible by 16 (896/16=56), so no remainder handling needed
        const float *W = cpu_decoder_weight.data();
#if defined(__ARM_NEON) || defined(__aarch64__)
        for (int o = 0; o < DECODER_OUT_DIM; o++) {
            const float *row = W + o * DECODER_IN_DIM;
            float32x4_t sum0 = vdupq_n_f32(0.0f);
            float32x4_t sum1 = vdupq_n_f32(0.0f);
            float32x4_t sum2 = vdupq_n_f32(0.0f);
            float32x4_t sum3 = vdupq_n_f32(0.0f);
            for (int i = 0; i < DECODER_IN_DIM; i += 16) {
                sum0 = vfmaq_f32(sum0, vld1q_f32(input + i),      vld1q_f32(row + i));
                sum1 = vfmaq_f32(sum1, vld1q_f32(input + i + 4),  vld1q_f32(row + i + 4));
                sum2 = vfmaq_f32(sum2, vld1q_f32(input + i + 8),  vld1q_f32(row + i + 8));
                sum3 = vfmaq_f32(sum3, vld1q_f32(input + i + 12), vld1q_f32(row + i + 12));
            }
            sum0 = vaddq_f32(sum0, sum1);
            sum2 = vaddq_f32(sum2, sum3);
            sum0 = vaddq_f32(sum0, sum2);
            output[o] = vaddvq_f32(sum0);
        }
#else
        for (int o = 0; o < DECODER_OUT_DIM; o++) {
            float sum = 0.0f;
            const float *row = W + o * DECODER_IN_DIM;
            for (int i = 0; i < DECODER_IN_DIM; i++) {
                sum += input[i] * row[i];
            }
            output[o] = sum;
        }
#endif
    }

    // int prefill_grpid = 1;
    int decode_grpid = 0;
    bool b_stop = false;
    int min_len = -1;
    int max_len = -1;

public:
    bool Init(LLMAttrType attr)
    {
        ALOGI("LLM init start (CosyVoice3)");
        t_cqdm cqdm = create_cqdm(attr.axmodel_num + 3, 32);
        this->_attr = attr;
        tokenizer = CreateTokenizer(attr.tokenizer_type);
        if (!tokenizer->Init(attr.filename_tokenizer_model, attr.b_bos, attr.b_eos))
        {
            ALOGE("tokenizer.Init(%s, %d, %d) failed", attr.filename_tokenizer_model.c_str(), attr.b_bos, attr.b_eos);
            return false;
        }
        update_cqdm(&cqdm, 0, "count", "tokenizer init ok");

        if (!embed_selector.Init(attr.filename_tokens_embed, attr.tokens_embed_num, attr.tokens_embed_size, attr.b_use_mmap_load_embed))
        {
            ALOGE("embed_selector.Init(%s, %d, %d) failed", attr.filename_tokens_embed.c_str(), attr.tokens_embed_num, attr.tokens_embed_size);
            return false;
        }
        if (!llm_embed_selector.Init(attr.filename_llm_embed, attr.llm_embed_num, attr.llm_embed_size, attr.b_use_mmap_load_embed))
        {
            ALOGE("llm_embed_selector.Init(%s, %d, %d) failed", attr.filename_llm_embed.c_str(), attr.llm_embed_num, attr.llm_embed_size);
            return false;
        }
        if (!speech_embed_selector.Init(attr.filename_speech_embed, attr.speech_embed_num, attr.speech_embed_size, attr.b_use_mmap_load_embed))
        {
            ALOGE("speech_embed_selector.Init(%s, %d, %d) failed", attr.filename_tokens_embed.c_str(), attr.speech_embed_num, attr.speech_embed_size);
            return false;
        }
        update_cqdm(&cqdm, 1, "count", "embed_selector init ok");

        llama_layers.resize(attr.axmodel_num);
        ALOGI("attr.axmodel_num:%d",attr.axmodel_num);
        char axmodel_path[1024];
        for (int i = 0; i < attr.axmodel_num; i++)
        {
            sprintf(axmodel_path, attr.template_filename_axmodel.c_str(), i);
            llama_layers[i].filename = axmodel_path;

            if (!attr.b_dynamic_load_axmodel_layer)
            {
                int ret = llama_layers[i].layer.init(llama_layers[i].filename.c_str(), _attr.dev_id);
                if (ret != 0)
                {
                    ALOGE("init axmodel(%s) failed", llama_layers[i].filename.c_str());
                    return false;
                }
                int remain_cmm = get_pcie_remaining_cmm_size(_attr.dev_id);
                sprintf(axmodel_path, "init %d axmodel ok,remain_cmm(%d MB)", i, remain_cmm);
                update_cqdm(&cqdm, i + 2, "count", axmodel_path);
            }
            else
            {
                if (!attr.b_use_mmap_load_layer)
                {
                    if (!read_file(llama_layers[i].filename, llama_layers[i].layer_buffer_vec))
                    {
                        ALOGE("read_file(%s) failed", llama_layers[i].filename.c_str());
                        return false;
                    }
                }
                else
                {
                    llama_layers[i].layer_buffer.open_file(llama_layers[i].filename.c_str());
                }

                sprintf(axmodel_path, "read_file %s ok", llama_layers[i].filename.c_str());
                update_cqdm(&cqdm, i + 2, "count", axmodel_path);
            }
        }

        int ret = llama_post.init(attr.filename_post_axmodel.c_str(), _attr.dev_id);
        if (ret != 0)
        {
            ALOGE("init post axmodel(%s) failed", attr.filename_post_axmodel.c_str());
            return false;
        }
        // CPU decoder: load float32 weight matrix instead of NPU axmodel
        if (!attr.filename_decoder_weight.empty()) {
            FILE *f = fopen(attr.filename_decoder_weight.c_str(), "rb");
            if (f) {
                fseek(f, 0, SEEK_END);
                long fsize = ftell(f);
                fseek(f, 0, SEEK_SET);
                long expected = (long)DECODER_OUT_DIM * DECODER_IN_DIM * sizeof(float);
                if (fsize == expected) {
                    cpu_decoder_weight.resize(DECODER_OUT_DIM * DECODER_IN_DIM);
                    fread(cpu_decoder_weight.data(), sizeof(float), DECODER_OUT_DIM * DECODER_IN_DIM, f);
                    _attr.b_use_cpu_decoder = true;
                    ALOGI("Loaded CPU decoder weight: %s (%.1f MB)", attr.filename_decoder_weight.c_str(), fsize / 1024.0 / 1024.0);
                } else {
                    ALOGE("CPU decoder weight size mismatch: %ld vs expected %ld", fsize, expected);
                }
                fclose(f);
            } else {
                ALOGE("Failed to open CPU decoder weight: %s", attr.filename_decoder_weight.c_str());
            }
        }

        if (!_attr.b_use_cpu_decoder) {
            ret = llm_decoder.init(attr.filename_decoder_axmodel.c_str(), _attr.dev_id);
            if (ret != 0)
            {
                ALOGE("init llm decoder axmodel(%s) failed", attr.filename_decoder_axmodel.c_str());
                return false;
            }
        }

        // CRITICAL: Enable host<->device memory sync for PCIe (AXCL) mode
        // On PCIe, host memory (pVirAddr via malloc) and device memory (phyAddr via axcl_Malloc)
        // are separate. Without sync, memcpy to pVirAddr does NOT reach the NPU.
        // On BSP (SoC) mode this is not needed as memory is shared.
        ALOGI("Enabling auto sync for PCIe mode (all LLM models)");
        for (int i = 0; i < attr.axmodel_num; i++)
        {
            if (!attr.b_dynamic_load_axmodel_layer)
            {
                llama_layers[i].layer.set_auto_sync_before_inference(true);
                llama_layers[i].layer.set_auto_sync_after_inference(true);
            }
        }
        llama_post.set_auto_sync_before_inference(true);
        llama_post.set_auto_sync_after_inference(true);
        llm_decoder.set_auto_sync_before_inference(true);
        llm_decoder.set_auto_sync_after_inference(true);

        int remain_cmm = get_pcie_remaining_cmm_size(_attr.dev_id);
        sprintf(axmodel_path, "init post axmodel ok,remain_cmm(%d MB)", remain_cmm);
        update_cqdm(&cqdm, attr.axmodel_num + 2, "count", axmodel_path);

        if (attr.b_dynamic_load_axmodel_layer)
        {
            auto &layer = llama_layers[0];
            int ret;
            if (_attr.b_use_mmap_load_layer)
            {
                ret = layer.layer.init((char *)layer.layer_buffer.data(), layer.layer_buffer.size(), _attr.dev_id);
            }
            else
            {
                ret = layer.layer.init(layer.layer_buffer_vec.data(), layer.layer_buffer_vec.size(), _attr.dev_id);
            }
            if (ret != 0)
            {
                ALOGE("init axmodel(%s) failed", layer.filename.c_str());
            }
        }

        {
            _attr.max_token_len = llama_layers[0].layer.get_input("mask").nSize / sizeof(unsigned short) - 1;
            printf("\n");
            ALOGI("max_token_len : %d", _attr.max_token_len);
            _attr.kv_cache_size = llama_layers[0].layer.get_output("K_cache_out").nSize / sizeof(unsigned short);
            _attr.kv_cache_num = llama_layers[0].layer.get_input("K_cache").nSize / _attr.kv_cache_size / sizeof(unsigned short);
            ALOGI("kv_cache_size : %d, kv_cache_num: %d", _attr.kv_cache_size, _attr.kv_cache_num);
            if (_attr.max_token_len > _attr.kv_cache_num)
            {
                ALOGE("max_token_len(%d) > kv_cache_num(%d)", _attr.max_token_len, _attr.kv_cache_num);
                return false;
            }

            _attr.prefill_token_num = llama_layers[0].layer.get_input(1, "indices").vShape[1];
            ALOGI("prefill_token_num : %d", _attr.prefill_token_num);
			for (size_t i = 0; i < llama_layers[0].layer.get_num_input_groups() - 1; i++)
            {
                int prefill_max_kv_cache_num = llama_layers[0].layer.get_input(i + 1, "K_cache").vShape[1];
                ALOGI("grp: %ld, prefill_max_token_num : %d", i + 1, prefill_max_kv_cache_num);
                _attr.prefill_max_kv_cache_num_grp.push_back(prefill_max_kv_cache_num);
            }
            _attr.prefill_max_token_num = _attr.prefill_max_kv_cache_num_grp[_attr.prefill_max_kv_cache_num_grp.size() - 1];
            ALOGI("prefill_max_token_num : %d", _attr.prefill_max_token_num);
        }
        if (attr.b_dynamic_load_axmodel_layer)
        {
            for(int i=0; i<attr.axmodel_num;i++)
            {
                auto &layer = llama_layers[i];
                layer.layer.deinit();
            }
        }

        ALOGI("LLM init ok (CosyVoice3, eos_token_id=%d, temperature=%.2f, top_k=%d)", _attr.eos_token_id, _attr.temperature, _attr.top_k);
        return true;
    }

    LLMAttrType *getAttr()
    {
        return &_attr;
    }

    void Deinit()
    {
        for (int i = 0; i < _attr.axmodel_num; i++)
        {
            llama_layers[i].layer.deinit();
        }
        llama_post.deinit();
        llm_decoder.deinit();
        embed_selector.Deinit();
        llm_embed_selector.Deinit();
        speech_embed_selector.Deinit();
    }

    void Stop()
    {
        b_stop = true;
    }

    int TextToken2Embeds(std::vector<int> &token_ids,  std::vector<unsigned short> &token_embeds)
    {
        if(token_embeds.empty() || token_embeds.size() < token_ids.size()* _attr.tokens_embed_size)
        {
            token_embeds.resize(token_ids.size()* _attr.tokens_embed_size);
        }

        for (size_t i = 0; i < token_ids.size(); i++)
        {
            embed_selector.getByIndex(token_ids[i], token_embeds.data() + i * _attr.tokens_embed_size);
        }
        return token_embeds.size();
    }

    int SpeechToken2Embeds(std::vector<int> &token_ids,  std::vector<unsigned short> &token_embeds)
    {
        if(token_embeds.empty() || token_embeds.size() < token_ids.size()* _attr.speech_embed_size)
        {
            token_embeds.resize(token_ids.size()* _attr.speech_embed_size);
        }

        for (size_t i = 0; i < token_ids.size(); i++)
        {
            speech_embed_selector.getByIndex(token_ids[i], token_embeds.data() + i * _attr.speech_embed_size);
        }
        return token_embeds.size();
    }

    int Encode(std::vector<unsigned short> &out_embed, std::vector<std::vector<int>>& position_ids,  std::string text, std::vector<unsigned short> & prompt_text_embeds, std::vector<unsigned short> &prompt_speech_embeds)
    {
        ImageInfo img_info;
        img_info.img_prompt = false;
        std::vector<int> text_ids = tokenizer->Encode(text, img_info);
        int prompt_ids_size = prompt_text_embeds.size() / _attr.tokens_embed_size ;
        int total_size = prompt_ids_size + text_ids.size() + 2 + prompt_speech_embeds.size() / _attr.speech_embed_size;
        if (total_size > _attr.prefill_max_token_num)
        {
            ALOGE("input embeding size(%d) > prefill_max_token_num(%d)", total_size, _attr.prefill_max_token_num);
            return -1;
        }
        out_embed.resize(total_size * _attr.tokens_embed_size);


        llm_embed_selector.getByIndex(0, out_embed.data() + 0 * _attr.tokens_embed_size);

        memcpy(out_embed.data() + _attr.tokens_embed_size, prompt_text_embeds.data(), prompt_text_embeds.size() * sizeof(unsigned short));

        for (size_t i = 0; i < text_ids.size(); i++)
        {
            embed_selector.getByIndex(text_ids[i], out_embed.data() + (1+prompt_ids_size+i) * _attr.tokens_embed_size);
        }

        llm_embed_selector.getByIndex(1, out_embed.data() + (1+prompt_ids_size+text_ids.size()) * _attr.tokens_embed_size);

        memcpy(out_embed.data() + (1+prompt_ids_size+text_ids.size()+1) * _attr.tokens_embed_size,  prompt_speech_embeds.data(), prompt_speech_embeds.size() * sizeof(unsigned short));

        std::vector<int> pos_ids;
        for (size_t i = 0; i < total_size; i++)
        {
            pos_ids.push_back(i);
        }
        position_ids.push_back(pos_ids);

        min_len = text_ids.size() * 2;
        max_len = text_ids.size() * 20;
        return 0;
    }

    int Run(std::string input_str, std::vector<unsigned short> & prompt_text_embeds, std::vector<unsigned short> &prompt_speech_embeds,
            TokenBuffer& token_buffer,
            std::mutex& buffer_mutex,
            std::condition_variable& buffer_cv,
            std::atomic<bool>& llm_finished
        )
    {
        std::vector<unsigned short> text_embed;
        std::vector<std::vector<int>> position_ids;
        Encode(text_embed, position_ids, input_str, prompt_text_embeds, prompt_speech_embeds);
        return Run(text_embed, position_ids, token_buffer, buffer_mutex, buffer_cv, llm_finished);
    }

    int Run(std::vector<unsigned short>& text_embed, std::vector<std::vector<int>>& position_ids,
            TokenBuffer& token_buffer,
            std::mutex& buffer_mutex,
            std::condition_variable& buffer_cv,
            std::atomic<bool>& llm_finished
    )
    {
        b_stop = false;
        std::string final_out;

        bfloat16 bf16 = -65536.f;
        std::vector<unsigned short> mask(_attr.kv_cache_num + 1, bf16.data);
        std::vector<unsigned short> embed(_attr.speech_embed_size, 0);

        std::vector<int> cached_token;
        std::vector<int> token_ids;


        int input_embed_num = text_embed.size() / _attr.tokens_embed_size;
        int prefill_split_num = ceil((double)input_embed_num / _attr.prefill_token_num);
        ALOGI("input token num : %d, prefill_split_num : %d", input_embed_num, prefill_split_num);
        if (input_embed_num > _attr.prefill_max_token_num)
        {
            ALOGE("input token num(%d) > prefill_max_token_num(%d)", input_embed_num, _attr.prefill_max_token_num);
            return -1;
        }

        int kv_cache_num;
        mask[_attr.kv_cache_num] = 0;
        for (size_t i = 0; i < input_embed_num; i++)
        {
            mask[i] = 0;
        }
        timer t_cost;
        timer ttft_timer;
        ttft_timer.start();

        int max_pos_id=0;
        for (size_t p = 0; p < prefill_split_num; p++)
        {
            if (b_stop)
            {
                break;
            }
            _attr.prefill_grpid = p + 1;
            kv_cache_num = p * _attr.prefill_token_num;
            std::vector<unsigned short> mask_tmp;
            mask_tmp.resize(1 * _attr.prefill_token_num * (kv_cache_num + _attr.prefill_token_num), bf16.data);
            int input_num_token = _attr.prefill_token_num;
            if (p == prefill_split_num - 1)
            {
                input_num_token = input_embed_num - p * _attr.prefill_token_num;
            }

            ALOGI("input_num_token:%d", input_num_token);
            for (size_t i = 0; i < _attr.prefill_token_num; i++)
            {
                if (i < input_num_token)
                {
                    int mask_current_start = kv_cache_num;
                    auto mask_ptr = mask_tmp.data() + i * (kv_cache_num + _attr.prefill_token_num);

                    for (int j = 0; j < _attr.precompute_len + p * _attr.prefill_token_num; j++)
                    {
                        mask_ptr[j] = 0;
                    }

                    for (int j = mask_current_start; j < mask_current_start + i + 1; j++)
                    {
                        mask_ptr[j] = 0;
                    }
                }
            }

            std::vector<unsigned short> embed_tmp(_attr.prefill_token_num * _attr.tokens_embed_size, 0);
            if (p == (prefill_split_num - 1))
            {
                memcpy(embed_tmp.data(), text_embed.data() + p * _attr.prefill_token_num * _attr.tokens_embed_size, (input_embed_num - p * _attr.prefill_token_num) * _attr.tokens_embed_size * sizeof(unsigned short));
            }
            else
            {
                memcpy(embed_tmp.data(), text_embed.data() + p * _attr.prefill_token_num * _attr.tokens_embed_size, _attr.prefill_token_num * _attr.tokens_embed_size * sizeof(unsigned short));
            }

            for (unsigned int m = 0; m < _attr.axmodel_num; m++)
            {
                if (b_stop)
                {
                    break;
                }

                auto &layer = llama_layers[m];

                if (_attr.b_dynamic_load_axmodel_layer)
                {
                    int ret;
                    if (_attr.b_use_mmap_load_layer)
                    {
                        ret = layer.layer.init((char *)layer.layer_buffer.data(), layer.layer_buffer.size(), _attr.dev_id);
                    }
                    else
                    {
                        ret = layer.layer.init(layer.layer_buffer_vec.data(), layer.layer_buffer_vec.size(), _attr.dev_id);
                    }
                    if (ret != 0)
                    {
                        ALOGE("init axmodel(%s) failed", layer.filename.c_str());
                    }
                }


                // set indices
                auto &input_indices = layer.layer.get_input(_attr.prefill_grpid, "indices");
                unsigned int *input_indices_ptr = (unsigned int *)input_indices.pVirAddr;
                memset(input_indices_ptr, 0, input_indices.nSize);
                for(unsigned int i=0; i< position_ids.size(); i++){
                    for(unsigned int j=_attr.precompute_len + p * _attr.prefill_token_num, jj=0; j<_attr.precompute_len + (p + 1) * _attr.prefill_token_num; j++,jj++){
                        if(j<position_ids[i].size()){
                            input_indices_ptr[ i*_attr.prefill_token_num+jj ] = position_ids[i][j];
                            if(position_ids[i][j]>max_pos_id){
                                max_pos_id = position_ids[i][j];
                            }
                        }
                    }
                }

                // set mask
                auto &input_mask = layer.layer.get_input(_attr.prefill_grpid, "mask");
                memcpy((void *)input_mask.pVirAddr, (void *)mask_tmp.data(), mask_tmp.size() * sizeof(unsigned short));
                // set input
                auto &input_input = layer.layer.get_input(_attr.prefill_grpid, "input");
                memcpy((void *)input_input.pVirAddr, embed_tmp.data(), embed_tmp.size() * sizeof(unsigned short));

                layer.layer.inference(_attr.prefill_grpid);

                auto &input_decoder_k_cache = layer.layer.get_input(decode_grpid, "K_cache");
                auto &input_decoder_v_cache = layer.layer.get_input(decode_grpid, "V_cache");

                auto &output_k_cache = layer.layer.get_output(_attr.prefill_grpid, "K_cache_out");
                auto &output_v_cache = layer.layer.get_output(_attr.prefill_grpid, "V_cache_out");

                int kv_offset = (_attr.precompute_len + p * _attr.prefill_token_num) * _attr.kv_cache_size;

                memcpy((unsigned short *)input_decoder_k_cache.pVirAddr + kv_offset,
                        (void *)output_k_cache.pVirAddr,
                            sizeof(unsigned short) * input_num_token * _attr.kv_cache_size);

                memcpy((unsigned short *)input_decoder_v_cache.pVirAddr + kv_offset,
                            (void *)output_v_cache.pVirAddr,
                            sizeof(unsigned short) * input_num_token * _attr.kv_cache_size
                            );

                for(int gid=_attr.prefill_grpid+1; gid<prefill_split_num+1; gid++){
                    auto &input_prefill_k_cache = layer.layer.get_input(gid, "K_cache");
                    memcpy((unsigned short *)input_prefill_k_cache.pVirAddr + kv_offset,
                                (void *)output_k_cache.pVirAddr,
                                sizeof(unsigned short) * input_num_token * _attr.kv_cache_size
                                );
                }

                for(int gid=_attr.prefill_grpid+1; gid<prefill_split_num+1; gid++){
                    auto &input_prefill_v_cache = layer.layer.get_input(gid, "V_cache");
                    memcpy((unsigned short *)input_prefill_v_cache.pVirAddr + kv_offset,
                                (void *)output_v_cache.pVirAddr,
                                sizeof(unsigned short) * input_num_token * _attr.kv_cache_size
                                );
                }

                auto &output = layer.layer.get_output(_attr.prefill_grpid, "output");
                memcpy(embed_tmp.data(), (void *)output.pVirAddr, embed_tmp.size() * sizeof(unsigned short) );

                if (_attr.b_dynamic_load_axmodel_layer)
                {
                    layer.layer.deinit();
                }

            }
            if (p == (prefill_split_num - 1))
            {
                memcpy(embed.data(),
                       embed_tmp.data() + (input_embed_num - p * _attr.prefill_token_num - 1) * _attr.tokens_embed_size,
                       _attr.tokens_embed_size * sizeof(unsigned short));
            }
        }

        int next_token = -1;
        t_cqdm cqdm = create_cqdm(_attr.max_token_len, 32);
        int max_index;
        // CosyVoice3: Use eos_token_id for scores vector size (decoder output vocab)
        // The decoder outputs speech_token_size+4 logits, but we allocate based on speech_embed_num
        // to be safe. memcpy will only copy actual output bytes.
        std::vector<float> scores(_attr.speech_embed_num, 0.0f);
        // Pre-allocate logits buffer outside decode loop to avoid per-token heap allocation
        std::vector<float> logits;
        {

            // post process
            auto &input = llama_post.get_input(0);
            memcpy((void *)input.pVirAddr, embed.data(), embed.size() * sizeof(unsigned short));
            llama_post.inference();
            if (_attr.b_use_topk)
            {
                AXCL_SYS_MinvalidateCache(llama_post.get_output("indices").phyAddr, llama_post.get_output("indices").pVirAddr, llama_post.get_output("indices").nSize);
                max_index = *(int *)llama_post.get_output("indices").pVirAddr;
            }
            else
            {
                auto &output_post = llama_post.get_output("output_norm");
                unsigned short *post_out = (unsigned short *)output_post.pVirAddr;
                logits.resize(output_post.nSize/sizeof(unsigned short));
                for (int i = 0; i < output_post.nSize/sizeof(unsigned short); i++)
                {
                    unsigned int proc = post_out[i] << 16;
                    logits[i] = *reinterpret_cast<float *>(&proc);
                }

                std::fill(scores.begin(), scores.end(), 0.0f);
                if (_attr.b_use_cpu_decoder) {
                    cpu_decoder_forward(logits.data(), scores.data());
                    // Debug: dump first iteration logits
                    dump_cpu_decoder_logits(scores, 0);
                } else {
                    auto & input_decoder = llm_decoder.get_input(0);
                    memcpy(input_decoder.pVirAddr, logits.data(), logits.size()*sizeof(float));
                    llm_decoder.inference();
                    auto & output_decoder = llm_decoder.get_output(0);
                    float *post_decoder = (float *)output_decoder.pVirAddr;
                    memcpy(scores.data(), post_decoder, std::min((int)(scores.size() * sizeof(float)), (int)output_decoder.nSize));
                }

                // Temperature scaling
                if (_attr.temperature != 1.0f && _attr.temperature > 0.0f) {
                    float inv_temp = 1.0f / _attr.temperature;
                    for (int i = 0; i < (int)scores.size(); i++) {
                        scores[i] *= inv_temp;
                    }
                }

                // CosyVoice3: Use eos_token_id instead of speech_embed_num-3
                max_index = sampling::sampling_ids(scores, cached_token, _attr.eos_token_id, true, 100, 0.8f, _attr.top_k);
            }
            next_token = max_index;

            // CosyVoice3: Use eos_token_id for EOS check
            // Note: ignore_eos=true for first token, so sampling already avoids EOS
            // But also check here: first token should never be EOS (min_len > 0)
            if (max_index >= _attr.eos_token_id && min_len <= 0){
                llm_finished = true;
                buffer_cv.notify_all();
                ALOGI("hit eos (token=%d, eos=%d), llm finished", max_index, _attr.eos_token_id);
                return -1;
            }
            // If first token was EOS despite ignore_eos, resample without it
            if (max_index >= _attr.eos_token_id) {
                ALOGI("first token was eos (%d), but min_len=%d, forcing re-sample", max_index, min_len);
                max_index = 0;
                next_token = 0;
            }

            token_ids.push_back(max_index);
            cached_token.push_back(max_index);
            {
                std::lock_guard<std::mutex> lock(buffer_mutex);
                token_buffer.push_back(max_index);
            }
            buffer_cv.notify_one();
            ALOGI("ttft: %.2f ms", ttft_timer.cost());
        }
        t_cost.start();

        bool b_hit_eos = false;

        for (unsigned int indices = max_pos_id+1; indices - max_pos_id < max_len; indices++)
        {
            if (b_stop)
            {
                break;
            }

            speech_embed_selector.getByIndex(next_token, embed.data());
            memcpy((void *)llama_layers[0].layer.get_input(decode_grpid, "input").pVirAddr, embed.data(), llama_layers[0].layer.get_input(decode_grpid, "input").nSize);

            for (int m = 0; m < _attr.axmodel_num; m++)
            {
                if (b_stop)
                {
                    break;
                }

                auto &layer = llama_layers[m];

                if (_attr.b_dynamic_load_axmodel_layer)
                {
                    int ret;
                    if (_attr.b_use_mmap_load_layer)
                    {
                        ret = layer.layer.init((char *)layer.layer_buffer.data(), layer.layer_buffer.size(), _attr.dev_id);
                    }
                    else
                    {
                        ret = layer.layer.init(layer.layer_buffer_vec.data(), layer.layer_buffer_vec.size(), _attr.dev_id);
                    }
                    if (ret != 0)
                    {
                        ALOGE("init axmodel(%s) failed", layer.filename.c_str());
                    }
                }

                auto &input_k_cache = layer.layer.get_input(decode_grpid, "K_cache");
                auto &input_v_cache = layer.layer.get_input(decode_grpid, "V_cache");

                auto &input_indices = layer.layer.get_input(decode_grpid, "indices");
                memcpy((void *)input_indices.pVirAddr, &indices, sizeof(indices));

                auto &input_mask = layer.layer.get_input(decode_grpid, "mask");
                memcpy((void *)input_mask.pVirAddr, mask.data(), mask.size() * sizeof(unsigned short));

                layer.layer.inference(decode_grpid);

                auto &output_k_cache = layer.layer.get_output(decode_grpid, "K_cache_out");
                memcpy((unsigned short *)input_k_cache.pVirAddr + indices * _attr.kv_cache_size, (void *)output_k_cache.pVirAddr, output_k_cache.nSize);

                auto &output_v_cache = layer.layer.get_output(decode_grpid, "V_cache_out");
                memcpy((unsigned short *)input_v_cache.pVirAddr + indices * _attr.kv_cache_size, (void *)output_v_cache.pVirAddr, output_v_cache.nSize);

                if (m == _attr.axmodel_num - 1)
                {
                    memcpy((void *)llama_post.get_input(0).pVirAddr,
                           (void *)layer.layer.get_output(decode_grpid, "output").pVirAddr, llama_post.get_input(0).nSize);
                }
                else if (m < _attr.axmodel_num - 1)
                {
                    memcpy((void *)llama_layers[m + 1].layer.get_input(decode_grpid, "input").pVirAddr,
                           (void *)layer.layer.get_output(decode_grpid, "output").pVirAddr, layer.layer.get_input(decode_grpid, "input").nSize);
                }
            }

            mask[indices] = 0;
            {
                llama_post.inference();

                auto &output_post = llama_post.get_output("output_norm");
                unsigned short *post_out = (unsigned short *)output_post.pVirAddr;

                int logits_count = output_post.nSize/sizeof(unsigned short);
                logits.resize(logits_count);  // no-op after first iteration (already sized)
                for (int i = 0; i < logits_count; i++)
                {
                    unsigned int proc = post_out[i] << 16;
                    logits[i] = *reinterpret_cast<float *>(&proc);
                }

                std::fill(scores.begin(), scores.end(), 0.0f);
                if (_attr.b_use_cpu_decoder) {
                    cpu_decoder_forward(logits.data(), scores.data());
                    int decoded_count_dbg = indices - max_pos_id;
                    dump_cpu_decoder_logits(scores, decoded_count_dbg);
                } else {
                    auto & input_decoder = llm_decoder.get_input(0);
                    memcpy(input_decoder.pVirAddr, logits.data(), logits.size()*sizeof(float));
                    llm_decoder.inference();
                    auto & output_decoder = llm_decoder.get_output(0);
                    float *post_decoder = (float *)output_decoder.pVirAddr;
                    memcpy(scores.data(), post_decoder, std::min((int)(scores.size() * sizeof(float)), (int)output_decoder.nSize));
                }

                // Temperature scaling
                if (_attr.temperature != 1.0f && _attr.temperature > 0.0f) {
                    float inv_temp = 1.0f / _attr.temperature;
                    for (int i = 0; i < (int)scores.size(); i++) {
                        scores[i] *= inv_temp;
                    }
                }

                bool ignore_eos = false;
                int decoded_count = indices - max_pos_id;  // number of tokens decoded so far
                int expected_len = min_len * 3;  // ~text_len * 6
                if (_attr.b_use_cpu_decoder) {
                    // CPU decoder: ignore EOS until 80% of expected length
                    // bf16 Qwen2 layers produce unpredictably strong spurious EOS (gap > 10)
                    // Only ignore_eos=true can reliably block these
                    if (decoded_count < expected_len * 8 / 10) {
                        ignore_eos = true;
                    }
                } else {
                    if (decoded_count < min_len) {
                        ignore_eos = true;
                    }
                }

                // EOS adjustment: different strategies for NPU vs CPU decoder
                {
                    int expected_len = min_len * 3;  // ~text_len * 6
                    if (_attr.b_use_cpu_decoder) {
                        // CPU decoder: EOS schedule
                        // 0..80%: ignore_eos=true (handled above), suppress -5 as backup
                        // 80%..100%: ramp -5 → 0
                        // 100%..120%: ramp 0 → +5
                        // 120%+: +10 (force stop)
                        float eos_adj = 0.0f;
                        int neutral_point = expected_len;
                        int boost_point = expected_len + expected_len * 2 / 10;  // 1.2x
                        int ramp_start = expected_len * 8 / 10;
                        if (decoded_count < ramp_start) {
                            eos_adj = -5.0f;
                        } else if (decoded_count < neutral_point) {
                            float p = (float)(decoded_count - ramp_start) / (float)(neutral_point - ramp_start + 1);
                            eos_adj = -5.0f * (1.0f - p);  // ramp -5 → 0
                        } else if (decoded_count < boost_point) {
                            float p = (float)(decoded_count - neutral_point) / (float)(boost_point - neutral_point + 1);
                            eos_adj = p * 5.0f;  // ramp 0 → +5
                        } else {
                            eos_adj = 10.0f;  // force stop
                        }
                        if (eos_adj != 0.0f) {
                            for (int i = _attr.eos_token_id; i < (int)scores.size(); i++) {
                                scores[i] += eos_adj;
                            }
                        }
                    } else {
                        // NPU decoder: U16 quantization clips EOS logits, need boost
                        int ramp_start = expected_len;
                        int ramp_end = expected_len * 2;
                        if (decoded_count > ramp_start && !ignore_eos) {
                            float progress = (float)(decoded_count - ramp_start) / (float)(ramp_end - ramp_start + 1);
                            progress = std::min(progress, 1.0f);
                            float eos_boost = progress * 3.0f;
                            for (int i = _attr.eos_token_id; i < (int)scores.size(); i++) {
                                scores[i] += eos_boost;
                            }
                        }
                    }
                }

                // CosyVoice3: Use eos_token_id instead of speech_embed_num-3
                max_index = sampling::sampling_ids(scores, cached_token, _attr.eos_token_id, ignore_eos, 100, 0.8f, _attr.top_k);
                next_token = max_index;

                // CosyVoice3: Any token >= eos_token_id means stop
                if (max_index >= _attr.eos_token_id)
                {
                    b_hit_eos = true;
                    llm_finished = true;
                    buffer_cv.notify_all();
                    ALOGI("hit eos (token=%d >= %d) at decoded_count=%d, llm finished", max_index, _attr.eos_token_id, decoded_count);
                    break;
                }

                // Hard cap: force stop at 1.5x expected_len for CPU decoder
                if (_attr.b_use_cpu_decoder && decoded_count >= expected_len * 3 / 2) {
                    b_hit_eos = true;
                    llm_finished = true;
                    buffer_cv.notify_all();
                    ALOGI("hard cap at %d tokens (expected=%d), forcing stop", decoded_count, expected_len);
                    break;
                }

                if(max_index < _attr.eos_token_id)
                {
                    token_ids.push_back(max_index);
                    cached_token.push_back(max_index);
                    {
                        std::lock_guard<std::mutex> lock(buffer_mutex);
                        token_buffer.push_back(max_index);
                    }
                    buffer_cv.notify_one();

                }
            }


            if (b_hit_eos)
            {
                llm_finished = true;
                buffer_cv.notify_all();
                ALOGI("hit eos, llm finished");
                break;
            }
        }

        llm_finished = true;
        buffer_cv.notify_all();
        ALOGI("llm finished");

        printf("\n\n");
        fflush(stdout);
        float t_cost_ms = t_cost.cost();
        ALOGI("total decode tokens:%d", cached_token.size());
        ALOGN("hit eos, decode avg %.2f token/s\n", cached_token.size() / (t_cost_ms / 1000));

        // Dump speech tokens to file for comparison with original model
        {
            FILE *f = fopen("npu_speech_tokens.txt", "w");
            if (f) {
                for (size_t i = 0; i < token_ids.size(); i++) {
                    fprintf(f, "%d%s", token_ids[i], (i + 1 < token_ids.size()) ? " " : "\n");
                }
                fclose(f);
                ALOGI("Dumped %zu speech tokens to npu_speech_tokens.txt", token_ids.size());
            }
        }

        for (size_t i = 0; i < _attr.axmodel_num; i++)
        {
            for (size_t j = 0; j < llama_layers[i].layer.get_num_input_groups(); j++)
            {
                memset((void *)llama_layers[i].layer.get_input(j, "K_cache").pVirAddr, 0, llama_layers[i].layer.get_input(j, "K_cache").nSize);
                memset((void *)llama_layers[i].layer.get_input(j, "V_cache").pVirAddr, 0, llama_layers[i].layer.get_input(j, "V_cache").nSize);
            }
        }

        return cached_token.size();
    }
};
