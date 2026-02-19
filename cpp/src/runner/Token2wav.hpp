#pragma once
#include <string>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <cstddef>
#include <stdexcept>
#include "bfloat16.hpp"
// Tokenizer not needed in Token2wav — removed to avoid ax-llm/CV2 API conflict
#include "LLMEmbedSelector.hpp"
#include "ax_model_runner/ax_model_runner_ax650.hpp"
#include "utils/utils.hpp"
#include "utils/slice_3d.h"
#include "utils/concat_3d.h"
#include "ax_cmm_utils.hpp"
#include "cqdm.h"
#include "timer.hpp"
#include "opencv2/opencv.hpp"
// AXCL host mode: no direct ax_sys_api
// NOTE: MNN removed - CosyVoice3 runs HiFT Part1 on NPU (axmodel) instead of CPU (MNN)
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#endif
// For ONNX HiFT socket client
#include <sys/socket.h>
#include <sys/un.h>

class Token2Wav
{
public:
    // CosyVoice3 constants (changed from CosyVoice2)
    int flow_embed_num = 6561;     // same as CV2
    int flow_embed_size = 80;      // CV2: 512, CV3: 80 (flow.input_embedding dim)
    int token_mel_ratio = 2;       // same
    int token_hop_len = 25;        // same
    int max_infer_chunk_num = 3;   // same
    int mel_cache_len = 8;         // same
    int source_cache_len = mel_cache_len * 480;  // same
    int pre_lookahead_len = 3;     // same
    float inference_cfg_rate = 0.7; // same

private:
    ax_runner_ax650 flow_encoder_28;
    ax_runner_ax650 flow_encoder_53;
    ax_runner_ax650 flow_encoder_78;
    ax_runner_ax650 flow_encoder_50_final;

    ax_runner_ax650 flow_estimator_200;
    ax_runner_ax650 flow_estimator_250;
    ax_runner_ax650 flow_estimator_300;

    ax_runner_ax650 hift_p2_50_first;
    ax_runner_ax650 hift_p2_58;

    // CosyVoice3: HiFT Part1 on NPU (axmodel) instead of CPU (MNN)
    ax_runner_ax650 hift_p1_50_first;
    ax_runner_ax650 hift_p1_58;

    std::vector<float> rand_noise;
    std::vector<float> t_span;

    LLaMaEmbedSelector flow_embed_selector;

    std::unordered_map<std::string, std::vector<float>> hift_cache_dict;
    std::vector<float> speech_window; // np.hamming(2 * 8 * 480)

    // ONNX HiFT: socket path for external ONNX HiFT server (empty = use NPU)
    std::string onnx_hift_socket;
    // Full-mel mode: accumulate mel, run HiFT once at finalize (no crossfade)
    bool onnx_hift_fullmel = false;
    std::vector<float> onnx_mel_accum;
    int onnx_mel_next_global_offset = 0;  // tracks global token_offset for finalize adjustment

    int init_noise(std::string model_dir)
    {
        return readtxt(model_dir+"/rand_noise_1_80_300.txt", rand_noise);
    }

    int init_speech_window(std::string model_dir)
    {
        return readtxt(model_dir+"/speech_window_2x8x480.txt", speech_window);
    }

    int init_tspan(int n_timesteps)
    {
        if(n_timesteps <4)
        {
            return -1;
        }

        n_timesteps = n_timesteps;
        t_span = linspace(0.0, 1.0, n_timesteps + 1);
        return 0;
    }

public:
    bool Init(std::string model_dir, int n_timesteps, int devid = 0)
    {
        int ret;

        ret = init_tspan(n_timesteps);
        if(ret != 0){
            ALOGE("init_tspan failed, n_timesteps:%d", n_timesteps);
            return false;
        }

        ret = init_noise(model_dir);
        if(ret != 0){
            ALOGE("init rand noise(%s) failed", "rand_noise_1_80_300.txt");
            return false;
        }

        ret = init_speech_window(model_dir);
        if(ret != 0){
            ALOGE("init speech_window(%s) failed", "speech_window_2x8x480.txt");
            return false;
        }

        // CosyVoice3: bfloat16 embedding file, 80-dim
        if (!flow_embed_selector.Init((model_dir+"/flow.input_embedding.bfloat16.bin").c_str(), flow_embed_num, flow_embed_size, false))
        {
            ALOGE("flow_embed_selector.Init(%s, %d, %d) failed", (model_dir+"/flow.input_embedding.bfloat16.bin").c_str(), flow_embed_num, flow_embed_size);
            return false;
        }

        // Helper lambda to init axmodel with auto-sync enabled (PCIe host↔device DMA)
        auto init_model = [devid](ax_runner_ax650 &model, const std::string &path) -> int {
            int ret = model.init(path.c_str(), devid);
            if (ret != 0) {
                ALOGE("init axmodel(%s) failed", path.c_str());
                return ret;
            }
            // CRITICAL: Enable auto DMA sync for PCIe host mode
            // Without this, input data stays in host memory (pVirAddr) and never reaches
            // the NPU device memory (phyAddr), and NPU output stays in device memory
            model.set_auto_sync_before_inference(true);
            model.set_auto_sync_after_inference(true);
            return 0;
        };

        if (init_model(flow_encoder_28, model_dir+"/flow_encoder_28.axmodel") != 0) return false;
        if (init_model(flow_encoder_53, model_dir+"/flow_encoder_53.axmodel") != 0) return false;
        if (init_model(flow_encoder_78, model_dir+"/flow_encoder_78.axmodel") != 0) return false;
        if (init_model(flow_encoder_50_final, model_dir+"/flow_encoder_50_final.axmodel") != 0) return false;
        if (init_model(flow_estimator_200, model_dir+"/flow_estimator_200.axmodel") != 0) return false;
        if (init_model(flow_estimator_250, model_dir+"/flow_estimator_250.axmodel") != 0) return false;
        if (init_model(flow_estimator_300, model_dir+"/flow_estimator_300.axmodel") != 0) return false;
        if (init_model(hift_p2_50_first, model_dir+"/hift_p2_50_first.axmodel") != 0) return false;
        if (init_model(hift_p2_58, model_dir+"/hift_p2_58.axmodel") != 0) return false;
        // CosyVoice3: HiFT Part1 as axmodel (not MNN)
        if (init_model(hift_p1_50_first, model_dir+"/hift_p1_50_first.axmodel") != 0) return false;
        if (init_model(hift_p1_58, model_dir+"/hift_p1_58.axmodel") != 0) return false;

        ALOGI("Token2Wav init ok (CosyVoice3)");
        return true;
    }

    void set_onnx_hift_socket(const std::string &path, bool fullmel = false) {
        onnx_hift_socket = path;
        onnx_hift_fullmel = fullmel;
        if (!path.empty()) {
            ALOGI("ONNX HiFT mode: %s (NPU HiFT disabled, fullmel=%d)", path.c_str(), fullmel);
        }
    }

    void Deinit()
    {
        flow_encoder_28.deinit();
        flow_encoder_53.deinit();
        flow_encoder_78.deinit();
        flow_encoder_50_final.deinit();
        flow_estimator_200.deinit();
        flow_estimator_250.deinit();
        flow_estimator_300.deinit();
        hift_p2_50_first.deinit();
        hift_p2_58.deinit();
        hift_p1_50_first.deinit();
        hift_p1_58.deinit();
        flow_embed_selector.Deinit();
    }

    int SpeechToken2Embeds(std::vector<int> & token_ids,  std::vector<float> &token_embeds)
    {
        if(token_embeds.empty() || token_embeds.size() != token_ids.size()* flow_embed_size)
        {
            token_embeds.resize(token_ids.size()* flow_embed_size);
        }
        std::vector<unsigned short> speech_embeds_one(flow_embed_size);
        for (size_t i = 0; i < token_ids.size(); i++)
        {
            flow_embed_selector.getByIndex(token_ids[i], speech_embeds_one.data());
            for (int j = 0; j < flow_embed_size; j++)
                {
                    unsigned int proc = speech_embeds_one[j] << 16;
                    token_embeds[i * flow_embed_size + j] = *reinterpret_cast<float *>(&proc);
                }
        }
        return token_embeds.size();
    }

    int infer_flow_encoder(
        std::vector<float> & token_embeds, std::vector<float> & prompt_feat, std::vector<float> & spk_embeds, int token_len, bool finalize,
        std::vector<float> & mu, std::vector<float> & spks, std::vector<float> & cond
        )
    {
        timer t_enc; t_enc.start();
        ax_runner_ax650 * model;
        if(!finalize)
        {
            if(token_len == 28)
            {
                model = &flow_encoder_28;
            }else if(token_len == 53)
            {
                model = &flow_encoder_53;
            }else if(token_len == 78)
            {
                model = &flow_encoder_78;
            }else{
                return -1;
            }
        }else if(token_len == 50){
            model = &flow_encoder_50_final;
        }else{
            return -1;
        }

        void * p = model->get_input("token_embedding").pVirAddr;
        memcpy(p, token_embeds.data(), token_embeds.size() * sizeof(float));
        p = model->get_input("prompt_feat").pVirAddr;
        memcpy(p, prompt_feat.data(), prompt_feat.size() * sizeof(float));
        p = model->get_input("embedding").pVirAddr;
        memcpy(p, spk_embeds.data(), spk_embeds.size() * sizeof(float));

        model->inference();

        auto &output_mu = model->get_output("mu");
        if(mu.empty())
        {
            mu.resize(output_mu.nSize / sizeof(float));
        }
        memcpy(mu.data(), output_mu.pVirAddr, output_mu.nSize);

        auto &output_spks = model->get_output("spks");
        if(spks.empty())
        {
            spks.resize(output_spks.nSize / sizeof(float));
        }
        memcpy(spks.data(), output_spks.pVirAddr, output_spks.nSize);

        auto &output_cond = model->get_output("cond");
        if(cond.empty())
        {
            cond.resize(output_cond.nSize / sizeof(float));
        }
        memcpy(cond.data(), output_cond.pVirAddr, output_cond.nSize);

        ALOGI("[profile] flow_encoder (token_len=%d, final=%d): %.1f ms", token_len, finalize, t_enc.cost());
        return 0;
    }

    int infer_flow_estimator(
        std::vector<float> & x, std::vector<float> & mask, std::vector<float> & t,
        std::vector<float> & mu, std::vector<float> & spks, std::vector<float> & cond,
        std::vector<float> & dphi_dt
        )
    {
        ax_runner_ax650 * model;
        int len = x.size()/(2*80);
        if(len == 200){
            model = &flow_estimator_200;
        }else if(len == 250){
            model = &flow_estimator_250;
        }else if(len == 300){
            model = &flow_estimator_300;
        }else{
            return -1;
        }

        void * p = model->get_input("x").pVirAddr;
        memcpy(p, x.data(), x.size() * sizeof(float));
        p = model->get_input("mask").pVirAddr;
        memcpy(p, mask.data(), mask.size() * sizeof(float));
        p = model->get_input("t").pVirAddr;
        memcpy(p, t.data(), t.size() * sizeof(float));
        p = model->get_input("mu").pVirAddr;
        memcpy(p, mu.data(), mu.size() * sizeof(float));
        p = model->get_input("spks").pVirAddr;
        memcpy(p, spks.data(), spks.size() * sizeof(float));
        p = model->get_input("cond").pVirAddr;
        memcpy(p, cond.data(), cond.size() * sizeof(float));

        model->inference();

        auto &output_dphi_dt = model->get_output("y");
        if(dphi_dt.empty() || dphi_dt.size() != output_dphi_dt.nSize / sizeof(float))
        {
            dphi_dt.resize(output_dphi_dt.nSize / sizeof(float));
        }
        memcpy(dphi_dt.data(), output_dphi_dt.pVirAddr, output_dphi_dt.nSize);

        return 0;
    }

    // CosyVoice3: Simplified infer_hift without source cache
    // Part1 (F0 + source) and Part2 (decode) both run on NPU via axmodel
    // No hift_cache_source input, no source output (CausalConv handles continuity)
    int infer_hift(std::vector<float> &mel, bool is_first,
                    std::vector<float> & tts_speech)
    {
        timer t_hift; t_hift.start();
        ax_runner_ax650 * model_p1;
        ax_runner_ax650 * model_p2;
        int len = mel.size()/(80);

        if(len == 50 && is_first)
        {
            model_p1 = &hift_p1_50_first;
            model_p2 = &hift_p2_50_first;
        }else if(len == 58 && !is_first)
        {
            model_p1 = &hift_p1_58;
            model_p2 = &hift_p2_58;
        }else
        {
            ALOGE("invalid hift size: %d, is_first: %d", len, is_first);
            return -1;
        }

        // Part1: mel -> source signal (s) via axmodel
        timer t_p1; t_p1.start();
        auto &input_mel_p1 = model_p1->get_input("mel");
        memcpy(input_mel_p1.pVirAddr, mel.data(), mel.size() * sizeof(float));

        // Debug: mel stats
        {
            float mel_min = mel[0], mel_max = mel[0], mel_sum = 0;
            for (auto v : mel) { mel_min = std::min(mel_min, v); mel_max = std::max(mel_max, v); mel_sum += v; }
            ALOGI("[debug] hift_p1 input mel: len=%d, min=%.3f, max=%.3f, mean=%.3f, input_nSize=%u",
                  len, mel_min, mel_max, mel_sum / mel.size(), (unsigned)input_mel_p1.nSize);
        }

        model_p1->inference();

        auto &output_s = model_p1->get_output("s");
        ALOGI("[debug] hift_p1 output s: nSize=%u, expected=%d", (unsigned)output_s.nSize, (int)(len * 480 * sizeof(float)));

        // Debug: source signal stats
        {
            float *s_data = (float*)output_s.pVirAddr;
            int s_len = output_s.nSize / sizeof(float);
            float s_min = s_data[0], s_max = s_data[0], s_sum = 0;
            for (int i = 0; i < s_len; i++) { s_min = std::min(s_min, s_data[i]); s_max = std::max(s_max, s_data[i]); s_sum += s_data[i]; }
            ALOGI("[debug] hift_p1 output s stats: len=%d, min=%.4f, max=%.4f, mean=%.6f", s_len, s_min, s_max, s_sum / s_len);
        }

        float t_p1_ms = t_p1.cost();
        // Part2: mel + s -> audio via axmodel
        timer t_p2; t_p2.start();
        auto &input_mel_p2 = model_p2->get_input("mel");
        memcpy(input_mel_p2.pVirAddr, mel.data(), mel.size() * sizeof(float));

        auto &input_s_p2 = model_p2->get_input("s");
        size_t s_copy_size = std::min((size_t)output_s.nSize, (size_t)(len * 480 * sizeof(float)));
        memcpy(input_s_p2.pVirAddr, output_s.pVirAddr, s_copy_size);
        ALOGI("[debug] hift_p2 input: mel_nSize=%u, s_nSize=%u, s_copied=%zu",
              (unsigned)input_mel_p2.nSize, (unsigned)input_s_p2.nSize, s_copy_size);

        // CosyVoice3: No hift_cache_source input (CausalConv1d handles context internally)

        model_p2->inference();

        auto &output_speech = model_p2->get_output("audio");
        if(tts_speech.empty() || tts_speech.size() != output_speech.nSize / sizeof(float))
        {
            tts_speech.resize(output_speech.nSize / sizeof(float));
        }
        memcpy(tts_speech.data(), output_speech.pVirAddr, output_speech.nSize);

        // Debug: audio output stats
        {
            float a_min = tts_speech[0], a_max = tts_speech[0], a_sum = 0;
            for (auto v : tts_speech) { a_min = std::min(a_min, v); a_max = std::max(a_max, v); a_sum += v; }
            ALOGI("[debug] hift_p2 output audio: nSize=%u, samples=%zu, min=%.4f, max=%.4f, mean=%.6f",
                  (unsigned)output_speech.nSize, tts_speech.size(), a_min, a_max, a_sum / tts_speech.size());
        }

        ALOGI("[profile] hift (mel_len=%d, first=%d): p1=%.1f ms, p2=%.1f ms, total=%.1f ms", len, is_first, t_p1_ms, t_p2.cost(), t_hift.cost());
        return 0;
    }

    // ONNX HiFT via external Python server (socket)
    int infer_hift_onnx(std::vector<float> &mel, bool is_first,
                         std::vector<float> &tts_speech)
    {
        timer t_hift; t_hift.start();
        int mel_len = mel.size() / 80;

        // Connect to ONNX HiFT server
        int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
        if (fd < 0) {
            ALOGE("infer_hift_onnx: socket() failed: %s", strerror(errno));
            return -1;
        }

        struct sockaddr_un addr;
        memset(&addr, 0, sizeof(addr));
        addr.sun_family = AF_UNIX;
        strncpy(addr.sun_path, onnx_hift_socket.c_str(), sizeof(addr.sun_path) - 1);

        if (::connect(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
            ALOGE("infer_hift_onnx: connect(%s) failed: %s", onnx_hift_socket.c_str(), strerror(errno));
            ::close(fd);
            return -1;
        }

        // Send: [mel_len (int32)] [is_first (int32)] [mel_data (float32 * mel_len * 80)]
        int32_t hdr[2] = {mel_len, is_first ? 1 : 0};
        if (::send(fd, hdr, 8, 0) != 8) {
            ALOGE("infer_hift_onnx: send header failed");
            ::close(fd);
            return -1;
        }
        size_t mel_bytes = mel.size() * sizeof(float);
        size_t sent = 0;
        while (sent < mel_bytes) {
            ssize_t n = ::send(fd, (char*)mel.data() + sent, mel_bytes - sent, 0);
            if (n <= 0) {
                ALOGE("infer_hift_onnx: send mel failed at %zu/%zu", sent, mel_bytes);
                ::close(fd);
                return -1;
            }
            sent += n;
        }

        // Receive: [audio_len (int32)] [audio_data (float32 * audio_len)]
        int32_t audio_len = 0;
        ssize_t r = 0;
        size_t total_recv = 0;
        while (total_recv < 4) {
            r = ::recv(fd, (char*)&audio_len + total_recv, 4 - total_recv, 0);
            if (r <= 0) { ::close(fd); return -1; }
            total_recv += r;
        }

        if (audio_len < 0) {
            ALOGE("infer_hift_onnx: server returned error");
            ::close(fd);
            return -1;
        }

        tts_speech.resize(audio_len);
        size_t audio_bytes = audio_len * sizeof(float);
        total_recv = 0;
        while (total_recv < audio_bytes) {
            r = ::recv(fd, (char*)tts_speech.data() + total_recv, audio_bytes - total_recv, 0);
            if (r <= 0) {
                ALOGE("infer_hift_onnx: recv audio failed at %zu/%zu", total_recv, audio_bytes);
                ::close(fd);
                return -1;
            }
            total_recv += r;
        }

        ::close(fd);
        ALOGI("[profile] hift_onnx (mel_len=%d, first=%d): %.1f ms, audio=%d samples",
              mel_len, is_first, t_hift.cost(), audio_len);
        return 0;
    }

    int infer_flow_decoder_solve_euler(
        std::vector<float> & x,  std::vector<float> & mu, std::vector<float> & spks, std::vector<float> & cond, std::vector<float> & mask,
        std::vector<float> & mel
    )
    {
        timer t_ode_total; t_ode_total.start();
        int len = mu.size()/80;

        float t = t_span[0];
        float dt = t_span[1] - t_span[0];

        std::vector<float> x_in(2*80*len, 0);
        std::vector<float> mask_in(2*1*len, 0);
        std::vector<float> mu_in(2*80*len,0);
        std::vector<float> t_in(2,0);
        std::vector<float> spks_in(2*80, 0);
        std::vector<float> cond_in(2*80*len, 0);
        std::vector<float> dphi_dt;  // pre-allocate outside loop
        for(int step=1; step<t_span.size(); step++)
        {
            timer t_step; t_step.start();
            memcpy(x_in.data(), x.data(), x.size() * sizeof(float));
            memcpy(x_in.data()+x.size(),  x.data(), x.size() * sizeof(float));

            memcpy(mask_in.data(), mask.data(), mask.size() * sizeof(float));
            memcpy(mask_in.data()+mask.size(), mask.data(), mask.size() * sizeof(float));

            memcpy(mu_in.data(), mu.data(), mu.size() * sizeof(float));

            t_in[0] = t;
            t_in[1] = t;

            memcpy(spks_in.data(), spks.data(), spks.size() * sizeof(float));
            memcpy(cond_in.data(), cond.data(), cond.size() * sizeof(float));

            int ret = infer_flow_estimator(x_in, mask_in, t_in, mu_in, spks_in, cond_in, dphi_dt);
            if(ret != 0)
            {
                return ret;
            }

            // CFG interpolation + Euler step (NEON-optimized on ARM)
            {
                int total = 80 * len;
                float cfg_pos = 1.0f + inference_cfg_rate;
                float cfg_neg = inference_cfg_rate;
#if defined(__ARM_NEON) || defined(__aarch64__)
                float32x4_t v_cfg_pos = vdupq_n_f32(cfg_pos);
                float32x4_t v_cfg_neg = vdupq_n_f32(cfg_neg);
                float32x4_t v_dt = vdupq_n_f32(dt);
                int i = 0;
                for (; i + 3 < total; i += 4) {
                    float32x4_t d0 = vld1q_f32(&dphi_dt[i]);
                    float32x4_t d1 = vld1q_f32(&dphi_dt[total + i]);
                    float32x4_t d = vsubq_f32(vmulq_f32(v_cfg_pos, d0), vmulq_f32(v_cfg_neg, d1));
                    float32x4_t xi = vld1q_f32(&x[i]);
                    vst1q_f32(&x[i], vfmaq_f32(xi, v_dt, d));
                }
                for (; i < total; i++) {
                    float d = cfg_pos * dphi_dt[i] - cfg_neg * dphi_dt[total + i];
                    x[i] += dt * d;
                }
#else
                for (int i = 0; i < total; i++) {
                    float d = cfg_pos * dphi_dt[i] - cfg_neg * dphi_dt[total + i];
                    x[i] += dt * d;
                }
#endif
            }

            t = t + dt;

            ALOGI("[profile] ODE step %d/%d (len=%d): %.1f ms", step, (int)t_span.size()-1, len, t_step.cost());

            if(step < t_span.size()-1)
            {
                dt = t_span[step+1] - t;
            }
            else{
                if(mel.empty() || mel.size()!=x.size())
                {
                    mel.resize(x.size());
                }
                memcpy(mel.data(), x.data(), x.size() * sizeof(float));
            }

        }

        ALOGI("[profile] ODE total (%d steps, len=%d): %.1f ms", (int)t_span.size()-1, len, t_ode_total.cost());
        return 0;
    }

    int infer_flow_decoder(
        std::vector<float> & mu, std::vector<float> & spks, std::vector<float> & cond, std::vector<float> & mask,
        std::vector<float> & mel
    )
    {
        std::vector<float> z;
        z.insert(z.end(), rand_noise.begin(), rand_noise.begin() + mu.size());

        int ret = infer_flow_decoder_solve_euler(z, mu, spks, cond, mask, mel);
        return ret;
    }

    std::vector<float> infer_flow(
        std::vector<float> & token_embeds, std::vector<float> & prompt_feat, std::vector<float> & spk_embeds, int token_len, bool finalize
    )
    {
        timer t_flow; t_flow.start();
        int ret;
        int len;
        std::vector<float> mu;
        std::vector<float> spks;
        std::vector<float> cond;

        ret = infer_flow_encoder(token_embeds, prompt_feat, spk_embeds, token_len, finalize, mu, spks, cond);
        if(ret != 0)
        {
            return std::vector<float>{};
        }

        len = mu.size()/80;

        std::vector<float> mask(len, 1.0);
        std::vector<float> all_mel;

        ret = infer_flow_decoder(mu, spks, cond, mask, all_mel);
        if(ret != 0)
        {
            return std::vector<float>{};
        }

        int len_mel1 = prompt_feat.size()/80;
        int len_mel2 = all_mel.size()/80 - len_mel1;

        std::vector<float> mel(len_mel2 * 80, 0);
        auto result = slice_3d_last_dim_from<float>(all_mel, 1, 80, all_mel.size()/80, len_mel1);

        // NO mel correction — NPU mel already correct per calibration analysis
        // NPU-mel + ONNX-hift = ZCR=1073 (87% voiced) — mel is fine
        // Issue is in NPU HiFT quantization, not mel values

        ALOGI("[profile] infer_flow (token_len=%d, final=%d, calibrated correction): %.1f ms", token_len, finalize, t_flow.cost());
        return result;
    }

    void fade_in_out(std::vector<float>& fade_in_mel_data,
                 const std::vector<float>& fade_out_mel_data,
                 const std::vector<float>& window) {

        // Hann window COLA crossfade — best results for chunked HiFT
        const size_t WINDOW_SIZE = 2 * 8 * 480; // 7680
        const size_t MEL_OVERLAP_LEN = WINDOW_SIZE / 2; // 3840
        size_t dim1_in = fade_in_mel_data.size();
        size_t dim1_out = fade_out_mel_data.size();

        if (window.size() != WINDOW_SIZE) {
            throw std::invalid_argument("window size mismatch");
        }
        if (dim1_in < MEL_OVERLAP_LEN || dim1_out < MEL_OVERLAP_LEN) {
            throw std::invalid_argument("data too short for crossfade");
        }

        for (size_t i = 0; i < MEL_OVERLAP_LEN; ++i) {
            const size_t out_idx = dim1_out - MEL_OVERLAP_LEN + i;
            fade_in_mel_data[i] = fade_in_mel_data[i] * window[i] +
                                  fade_out_mel_data[out_idx] * window[MEL_OVERLAP_LEN + i];
        }
    }

    void reset()
    {
        std::unordered_map<std::string, std::vector<float>>().swap(hift_cache_dict);
        onnx_mel_accum.clear();
        onnx_mel_next_global_offset = 0;
    }

    std::vector<float> infer(std::vector<int> & text_speech_token, std::vector<float> & prompt_speech_embeds, std::vector<float> & prompt_feat,
                std::vector<float> & spk_embeds, int token_offset, bool finalize)
    {
        timer t_infer; t_infer.start();
        int ret = 0;
        std::vector<float> speech_embeds( text_speech_token.size()*flow_embed_size + prompt_speech_embeds.size(), 0.0f);
        std::vector<unsigned short> speech_embeds_one(flow_embed_size, 0);

        memcpy(speech_embeds.data(), prompt_speech_embeds.data(), prompt_speech_embeds.size() * sizeof(float));

        for (size_t i = 0; i < text_speech_token.size(); i++)
        {
            flow_embed_selector.getByIndex(text_speech_token[i], speech_embeds_one.data());

            for (int j = 0; j < flow_embed_size; j++)
                {
                    unsigned int proc = speech_embeds_one[j] << 16;
                    speech_embeds[prompt_speech_embeds.size() + i * flow_embed_size + j] = *reinterpret_cast<float *>(&proc);
                }
        }

        std::vector<float> mel;

        mel = infer_flow(speech_embeds, prompt_feat, spk_embeds, text_speech_token.size(), finalize);

        std::vector<float> tts_mel;
        int neg_offset=0, start;
        if(finalize)
        {
            neg_offset = token_offset * token_mel_ratio - mel.size()/80;
            start = - token_hop_len * token_mel_ratio;
        }
        else{
            start = std::min( int(token_offset / token_hop_len), max_infer_chunk_num-1) * token_hop_len * token_mel_ratio;
        }

        tts_mel = slice_3d_last_dim_from<float>(mel, 1, 80, mel.size()/80, start);

        std::vector<float> tts_speech;

        // ============ Full-mel mode: accumulate mel, run HiFT once at finalize ============
        if (onnx_hift_fullmel) {
            int mel_frames = mel.size() / 80;
            int accum_frames = onnx_mel_accum.size() / 80;
            int exact_start;

            if (!finalize) {
                // Non-finalize: use same sliding window formula as non-fullmel mode.
                exact_start = std::min(std::min(int(token_offset / token_hop_len), max_infer_chunk_num-1) * token_hop_len * token_mel_ratio, mel_frames);
                // Track global offset for finalize adjustment
                onnx_mel_next_global_offset = token_offset + token_hop_len;
            } else {
                // Finalize: token_offset is LOCAL within a token window.
                // The finalize window starts at global position:
                //   finalize_global_start = onnx_mel_next_global_offset - token_offset
                // We need mel frames starting from where accum actually ends:
                //   accum covers positions 0 through (accum_frames / token_mel_ratio - 1)
                //   In finalize mel, that corresponds to frame:
                //     (accum_end_pos - finalize_global_start) * token_mel_ratio
                int finalize_global_start = onnx_mel_next_global_offset - token_offset;
                int accum_end_pos = accum_frames / token_mel_ratio;
                int overlap_positions = std::max(0, accum_end_pos - finalize_global_start);
                exact_start = std::min(overlap_positions * token_mel_ratio, mel_frames);
                ALOGI("Finalize mel adjustment: global_start=%d, accum_end=%d, overlap=%d, exact_start=%d (was %d)",
                      finalize_global_start, accum_end_pos, overlap_positions, exact_start, token_offset * token_mel_ratio);
            }

            // Append new mel frames with crossfade at boundaries [80, T] layout
            if (exact_start < mel_frames) {
                const int XFADE_LEN = 10;  // crossfade length in mel frames
                int blend_start = std::max(0, exact_start - XFADE_LEN);
                int blend_len = exact_start - blend_start;  // actual overlap frames

                auto new_mel = slice_3d_last_dim_from<float>(mel, 1, 80, mel_frames, blend_start);
                int new_T = new_mel.size() / 80;

                if (onnx_mel_accum.empty()) {
                    onnx_mel_accum = std::move(new_mel);
                } else {
                    int old_T = accum_frames;

                    if (blend_len > 0 && old_T >= blend_len) {
                        // Crossfade: blend last blend_len of accum with first blend_len of new_mel
                        int fresh_T = new_T - blend_len;
                        int total_T = old_T + fresh_T;
                        std::vector<float> merged(80 * total_T);
                        for (int ch = 0; ch < 80; ch++) {
                            // Copy accum up to blend zone
                            memcpy(&merged[ch * total_T],
                                   &onnx_mel_accum[ch * old_T],
                                   (old_T - blend_len) * sizeof(float));
                            // Linear crossfade in blend zone
                            for (int f = 0; f < blend_len; f++) {
                                float alpha = (float)(f + 1) / (blend_len + 1);
                                float v_old = onnx_mel_accum[ch * old_T + (old_T - blend_len + f)];
                                float v_new = new_mel[ch * new_T + f];
                                merged[ch * total_T + (old_T - blend_len + f)] = v_old * (1.0f - alpha) + v_new * alpha;
                            }
                            // Copy fresh (non-overlap) portion of new_mel
                            if (fresh_T > 0) {
                                memcpy(&merged[ch * total_T + old_T],
                                       &new_mel[ch * new_T + blend_len],
                                       fresh_T * sizeof(float));
                            }
                        }
                        onnx_mel_accum = std::move(merged);
                    } else {
                        // No overlap possible (first chunk), hard concat
                        int total_T = old_T + new_T;
                        std::vector<float> merged(80 * total_T);
                        for (int ch = 0; ch < 80; ch++) {
                            memcpy(&merged[ch * total_T],
                                   &onnx_mel_accum[ch * old_T],
                                   old_T * sizeof(float));
                            memcpy(&merged[ch * total_T + old_T],
                                   &new_mel[ch * new_T],
                                   new_T * sizeof(float));
                        }
                        onnx_mel_accum = std::move(merged);
                    }
                }
            }

            accum_frames = onnx_mel_accum.size() / 80;

            if (!finalize) {
                ALOGI("[profile] Token2Wav::infer chunk (tokens=%d, offset=%d, exact_start=%d, final=0): %.1f ms, mel_accum=%d frames",
                      (int)text_speech_token.size(), token_offset, exact_start, t_infer.cost(), accum_frames);
                return tts_speech;  // empty, wait for more mel
            }

            // Finalize: send all accumulated mel to ONNX server
            ALOGI("Full-mel HiFT: %d mel frames → ONNX", accum_frames);
            std::vector<float> speech;
            ret = infer_hift_onnx(onnx_mel_accum, true, speech);
            onnx_mel_accum.clear();
            onnx_mel_next_global_offset = 0;

            if (ret != 0) {
                ALOGE("infer_hift_onnx (fullmel) failed");
                return std::vector<float>{};
            }

            tts_speech = std::move(speech);

            ALOGI("[profile] Token2Wav::infer chunk (tokens=%d, offset=%d, final=%d): %.1f ms, audio=%d (%.2fs)",
                  (int)text_speech_token.size(), token_offset, finalize, t_infer.cost(),
                  (int)tts_speech.size(), (float)tts_speech.size() / 24000.0f);
            return tts_speech;
        }

        // ============ Chunked mode (NPU or legacy ONNX) ============
        bool is_first = hift_cache_dict.empty();
        std::vector<float> tts_mel1;
        std::vector<float> speech;

        if (!is_first)
        {
            auto hift_cache_mel = hift_cache_dict["mel"];
            tts_mel1 = concat_3d_dim2<float>(hift_cache_mel, 1, 80, hift_cache_mel.size()/80, tts_mel, 1, 80, tts_mel.size()/80);
        }
        else{
            tts_mel1 = tts_mel;
        }

        // Dump mel for ONNX comparison when MEL_DUMP_DIR env is set
        {
            static int mel_chunk_idx = 0;
            const char* dump_dir = getenv("MEL_DUMP_DIR");
            if (dump_dir) {
                char fname[256];
                snprintf(fname, sizeof(fname), "%s/mel_chunk_%d.bin", dump_dir, mel_chunk_idx);
                FILE* f = fopen(fname, "wb");
                if (f) {
                    int mel_len = tts_mel1.size() / 80;
                    fwrite(&mel_len, sizeof(int), 1, f);
                    fwrite(tts_mel1.data(), sizeof(float), tts_mel1.size(), f);
                    fwrite(&is_first, sizeof(bool), 1, f);
                    fwrite(&finalize, sizeof(bool), 1, f);
                    fwrite(&neg_offset, sizeof(int), 1, f);
                    fclose(f);
                    ALOGI("[mel_dump] Saved chunk %d: mel_len=%d, is_first=%d, finalize=%d, neg_offset=%d -> %s",
                          mel_chunk_idx, mel_len, is_first, finalize, neg_offset, fname);
                }
                mel_chunk_idx++;
            }
        }

        if (!onnx_hift_socket.empty()) {
            ret = infer_hift_onnx(tts_mel1, is_first, speech);
        } else {
            ret = infer_hift(tts_mel1, is_first, speech);
        }

        if(ret != 0){
            ALOGE("infer_hift failed");
            return std::vector<float>{};
        }

        if(!finalize)
        {
            // Audio crossfade with previous chunk
            if(!is_first)
            {
                fade_in_out(speech, hift_cache_dict["speech"], speech_window);
            }

            // Cache mel overlap for next chunk
            hift_cache_dict["mel"] = slice_3d_last_dim_from<float>(tts_mel1, 1, 80, tts_mel1.size()/80, -mel_cache_len);

            int offset = speech.size();
            if(speech.size() > source_cache_len)
            {
                offset = source_cache_len;
            }

            // Cache speech tail for crossfade
            hift_cache_dict["speech"].assign(speech.end()-offset, speech.end());
            tts_speech.assign(speech.begin(), speech.end()-offset);

        }
        else{

            if(speech.size() < source_cache_len){
                tts_speech.assign(speech.begin(), speech.end());
            }
            else if (- neg_offset*480 >= source_cache_len)
            {
                tts_speech.assign(speech.end() + neg_offset*480, speech.end());

                if(!is_first)
                {
                    fade_in_out(tts_speech, hift_cache_dict["speech"], speech_window);
                }
            }
            else{
                tts_speech.assign(speech.end()-source_cache_len, speech.end());

                if(!is_first)
                {
                    fade_in_out(tts_speech, hift_cache_dict["speech"], speech_window);
                }

                int offset = speech.size() + neg_offset*480 - (speech.size() - source_cache_len);
                tts_speech.assign(tts_speech.begin() + offset, tts_speech.end());
            }

        }

        // Clip audio to [-0.99, 0.99]
        for (auto &v : tts_speech) {
            v = std::max(-0.99f, std::min(0.99f, v));
        }

        ALOGI("[profile] Token2Wav::infer chunk (tokens=%d, offset=%d, final=%d): %.1f ms, audio_samples=%d",
              (int)text_speech_token.size(), token_offset, finalize, t_infer.cost(), (int)tts_speech.size());
        return tts_speech;
    }
};
