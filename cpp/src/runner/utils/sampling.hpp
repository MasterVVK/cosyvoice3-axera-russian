// sampling.hpp - CosyVoice3 version
// Modified from CosyVoice2: EOS check changed from exact match to range check
// CosyVoice3 has 200 stop tokens (eos_token_id..eos_token_id+199)
#ifndef SAMPLING_H
#define SAMPLING_H

#include <vector>
#include <random>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <iostream>
#include <stdexcept>

namespace sampling {

    std::vector<float> softmax_stable(const std::vector<float>& logits) {
        if (logits.empty()) return {};

        float max_val = *std::max_element(logits.begin(), logits.end());

        std::vector<float> exp_values(logits.size());
        float sum_exp = 0.0f;
        for (size_t i = 0; i < logits.size(); ++i) {
            exp_values[i] = std::exp(logits[i] - max_val);
            sum_exp += exp_values[i];
        }

        if (sum_exp > 0.0f) {
            for (float& val : exp_values) {
                val /= sum_exp;
            }
        } else {
            float uniform_prob = 1.0f / static_cast<float>(exp_values.size());
            for (float& val : exp_values) {
                val = uniform_prob;
            }
        }
        return exp_values;
    }

    std::vector<size_t> sort_indices_desc(const std::vector<float>& v) {
        std::vector<size_t> idx(v.size());
        std::iota(idx.begin(), idx.end(), 0);

        std::stable_sort(idx.begin(), idx.end(),
                [&v](size_t i1, size_t i2) {return v[i1] > v[i2];});

        return idx;
    }

    int sample_multinomial(const std::vector<float>& probabilities, std::mt19937& gen) {
        if (probabilities.empty()) {
            throw std::invalid_argument("Cannot sample from an empty probability distribution.");
        }

        std::discrete_distribution<int> dist(probabilities.begin(), probabilities.end());
        return dist(gen);
    }

    int nucleus_sampling(const std::vector<float>& weighted_scores, float top_p = 0.8f, int top_k = 25) {
        if (weighted_scores.empty()) {
            throw std::invalid_argument("weighted_scores cannot be empty.");
        }

        std::vector<float> probs = softmax_stable(weighted_scores);

        std::vector<size_t> sorted_indices = sort_indices_desc(probs);

        std::vector<float> filtered_probs;
        std::vector<size_t> filtered_indices;
        float cum_prob = 0.0f;

        int actual_top_k = std::min(top_k, static_cast<int>(sorted_indices.size()));

        for (int i = 0; i < actual_top_k; ++i) {
            size_t idx = sorted_indices[i];
            float prob = probs[idx];
            if (cum_prob < top_p && static_cast<int>(filtered_probs.size()) < top_k) {
                cum_prob += prob;
                filtered_probs.push_back(prob);
                filtered_indices.push_back(idx);
            } else {
                break;
            }
        }

        if (filtered_probs.empty()) {
            filtered_probs.push_back(1.0f);
            filtered_indices.push_back(sorted_indices[0]);
        }

        float sum_filtered = std::accumulate(filtered_probs.begin(), filtered_probs.end(), 0.0f);
        if (sum_filtered > 0.0f) {
            for (float& prob : filtered_probs) {
                prob /= sum_filtered;
            }
        }

        static std::random_device rd;
        static std::mt19937 gen(rd());
        int sampled_index_in_filtered = sample_multinomial(filtered_probs, gen);

        return static_cast<int>(filtered_indices[sampled_index_in_filtered]);
    }

    int random_sampling(const std::vector<float>& weighted_scores) {
        if (weighted_scores.empty()) {
            throw std::invalid_argument("weighted_scores cannot be empty.");
        }
        std::vector<float> probs = softmax_stable(weighted_scores);

        static std::random_device rd;
        static std::mt19937 gen(rd());
        return sample_multinomial(probs, gen);
    }

    // Repetition-Aware Sampling (RAS)
    int ras_sampling(const std::vector<float>& weighted_scores,
                    const std::vector<int>& decoded_tokens,
                    int speech_token_size,
                    float top_p = 0.8f, int top_k = 25,
                    int win_size = 10, float tau_r = 0.1f) {

        int top_id = nucleus_sampling(weighted_scores, top_p, top_k);

        int rep_num = 0;
        int window_start = std::max(0, static_cast<int>(decoded_tokens.size()) - win_size);
        for (size_t i = window_start; i < decoded_tokens.size(); ++i) {
            if (decoded_tokens[i] == top_id) {
                rep_num++;
            }
        }

        if (rep_num >= static_cast<int>(win_size * tau_r)) {
            top_id = random_sampling(weighted_scores);
        }

        return top_id;
    }

    // Main sampling function with EOS handling
    // CosyVoice3: speech_token_size = eos_token_id (6561)
    //   Any token >= speech_token_size is a stop token
    //   Changed from CV2's exact match (!=) to range check (<)
    int sampling_ids(const std::vector<float>& weighted_scores,
                    const std::vector<int>& decoded_tokens,
                    int speech_token_size,
                    bool ignore_eos = true,
                    int max_trials = 100,
                    float top_p = 0.8f,
                    int top_k = 25) {

        static std::random_device rd;
        static std::mt19937 gen(rd());

        int num_trials = 0;
        int top_id = -1;

        while (true) {
            top_id = ras_sampling(weighted_scores, decoded_tokens, speech_token_size,
                                  top_p, top_k);

            // CosyVoice3: Check if token is in valid range (< eos_token_id)
            // CV2 used: top_id != speech_token_size (exact match)
            // CV3 uses: top_id < speech_token_size (range check, any token >= eos is stop)
            if (!ignore_eos || (speech_token_size < 0 || top_id < speech_token_size)) {
                break;
            }

            num_trials++;
            if (num_trials > max_trials) {
                // Fallback: find the best non-EOS token deterministically
                float best_val = -1e30f;
                int best_id = 0;
                for (int i = 0; i < speech_token_size && i < (int)weighted_scores.size(); i++) {
                    if (weighted_scores[i] > best_val) {
                        best_val = weighted_scores[i];
                        best_id = i;
                    }
                }
                top_id = best_id;
                break;
            }
        }
        return top_id;
    }

} // namespace sampling

#endif // SAMPLING_H
