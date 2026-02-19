#include <iostream>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <vector>
#include <atomic>
#include <chrono>
#include <random>
#include <sstream>
#include <opencv2/opencv.hpp>
#include "signal.h"
#include "runner/LLM.hpp"
#include "runner/Token2wav.hpp"
#include "runner/utils/slice_3d.h"
#include "runner/utils/wav.hpp"
#include "runner/utils/timer.hpp"
#include "cmdline.hpp"
#include "runner/utils/files.hpp"
#include "axcl_manager.h"

// For Unix domain socket (external token mode + daemon mode)
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/select.h>
#include <sys/stat.h>
#include <unistd.h>


static LLM lLaMa;
static Token2Wav lToken2Wav;

// --- Shared State ---
TokenBuffer g_token_buffer;
std::mutex g_buffer_mutex;
std::condition_variable g_buffer_cv;
std::atomic<bool> g_llm_finished{false};
std::atomic<bool> g_stop{false};
// --- Constants ---
const size_t MAX_BUFFER_SIZE = 100;

// --- External Token Mode ---
std::string g_external_tokens_socket;  // empty = normal mode, set = external token mode
std::vector<int> g_prompt_text_tokens;   // raw token IDs for RKLLM request
std::vector<int> g_prompt_speech_tokens; // raw token IDs for RKLLM request

// --- Daemon Mode ---
std::string g_daemon_socket;  // empty = no daemon, set = listen for requests on this socket


static int g_daemon_fd = -1;  // daemon listen socket fd for cleanup

void __sigExit(int iSigNo)
{
    if (g_external_tokens_socket.empty()) {
        lLaMa.Stop();
    }
    g_stop = true;
    if (g_daemon_fd >= 0) {
        close(g_daemon_fd);
        g_daemon_fd = -1;
    }
    return;
}

// --- External Token Reader (reads from RKLLM Token Server via Unix socket) ---
void external_token_reader(const std::string& socket_path, const std::string& request_json) {
    int sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock_fd < 0) {
        fprintf(stderr, "[ExtTokenReader] socket() failed: %s\n", strerror(errno));
        g_llm_finished = true;
        g_buffer_cv.notify_all();
        return;
    }

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path.c_str(), sizeof(addr.sun_path) - 1);

    if (connect(sock_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "[ExtTokenReader] connect(%s) failed: %s\n", socket_path.c_str(), strerror(errno));
        close(sock_fd);
        g_llm_finished = true;
        g_buffer_cv.notify_all();
        return;
    }

    // Send request: [4 bytes msg_len] [msg_len bytes JSON]
    uint32_t msg_len = (uint32_t)request_json.size();
    if (write(sock_fd, &msg_len, 4) != 4 ||
        write(sock_fd, request_json.c_str(), msg_len) != (ssize_t)msg_len) {
        fprintf(stderr, "[ExtTokenReader] Failed to send request\n");
        close(sock_fd);
        g_llm_finished = true;
        g_buffer_cv.notify_all();
        return;
    }

    printf("[ExtTokenReader] Connected to %s, reading tokens...\n", socket_path.c_str());

    // Read tokens: [4 bytes int32] per token, -1 = done, -2 = error
    int token_count = 0;
    while (!g_stop.load()) {
        int32_t token;
        ssize_t n = read(sock_fd, &token, 4);
        if (n != 4) {
            fprintf(stderr, "[ExtTokenReader] read error (got %zd bytes)\n", n);
            break;
        }

        if (token == -1) {  // SENTINEL_DONE
            break;
        }
        if (token == -2) {  // SENTINEL_ERROR
            fprintf(stderr, "[ExtTokenReader] Server reported error\n");
            break;
        }

        // Push token into shared buffer
        {
            std::lock_guard<std::mutex> lock(g_buffer_mutex);
            g_token_buffer.push_back(token);
        }
        g_buffer_cv.notify_one();
        token_count++;
    }

    close(sock_fd);
    printf("[ExtTokenReader] Received %d tokens\n", token_count);
    g_llm_finished = true;
    g_buffer_cv.notify_all();
}

void simulate_llm() {
    std::vector<int> tokens;
    readtxt("../../model_convert/llm_out_tokens.txt", tokens);

    std::cout << "[LLM Thread] Starting to generate tokens...\n";

    for (int& token : tokens)
    {
        {
            std::lock_guard<std::mutex> lock(g_buffer_mutex);
            g_token_buffer.push_back(token);

            std::cout << "[LLM Thread] Generated token " << g_token_buffer.back()
                      << " (Buffer size: " << g_token_buffer.size() << ")\n";
        }

        g_buffer_cv.notify_one();
    }

    g_llm_finished = true;
    std::cout << "[LLM Thread] Finished generating tokens.\n";

    g_buffer_cv.notify_all();
}

void reset()
{
    g_llm_finished = false;
    g_token_buffer.erase(g_token_buffer.begin(), g_token_buffer.end());
    lToken2Wav.reset();
}

int tts(
    // for llm
    std::string & text,
    std::vector<unsigned short> prompt_text_embeds,
    std::vector<unsigned short> prompt_speech_embeds,
    // for flow
    std::vector<float> prompt_feat,
    std::vector<float> prompt_speech_embeds_flow,
    std::vector<float> spk_embeds
)
{
    std::vector <float> output;
    timer time_total;
    time_total.start();
    try {
        std::thread llm_thread;

        if (g_external_tokens_socket.empty()) {
            // Normal mode: LLM runs on AX650N
            auto llm_thread_func = [&text, &prompt_text_embeds, &prompt_speech_embeds]() {
                lLaMa.Run(text, prompt_text_embeds, prompt_speech_embeds, g_token_buffer, g_buffer_mutex, g_buffer_cv, g_llm_finished);
            };
            llm_thread = std::thread(llm_thread_func);
        } else {
            // External mode: LLM runs on RK3588 RKLLM, tokens via Unix socket
            // Build JSON request: {"text":"...", "prompt_speech_tokens":[...], "prompt_text_tokens":[...]}
            std::ostringstream json;
            json << "{\"text\":\"";
            // Escape text for JSON
            for (char c : text) {
                switch (c) {
                    case '"': json << "\\\""; break;
                    case '\\': json << "\\\\"; break;
                    case '\n': json << "\\n"; break;
                    case '\r': json << "\\r"; break;
                    case '\t': json << "\\t"; break;
                    default: json << c;
                }
            }
            json << "\"";
            if (!g_prompt_speech_tokens.empty()) {
                json << ",\"prompt_speech_tokens\":[";
                for (size_t i = 0; i < g_prompt_speech_tokens.size(); i++) {
                    if (i > 0) json << ",";
                    json << g_prompt_speech_tokens[i];
                }
                json << "]";
            }
            if (!g_prompt_text_tokens.empty()) {
                json << ",\"prompt_text_tokens\":[";
                for (size_t i = 0; i < g_prompt_text_tokens.size(); i++) {
                    if (i > 0) json << ",";
                    json << g_prompt_text_tokens[i];
                }
                json << "]";
            }
            json << "}";

            std::string request_json = json.str();
            std::string sock_path = g_external_tokens_socket;
            llm_thread = std::thread([sock_path, request_json]() {
                external_token_reader(sock_path, request_json);
            });
        }

        int token_offset = 0;
        int prompt_token_len = prompt_speech_embeds_flow.size() / lToken2Wav.flow_embed_size;
        if(prompt_token_len < 75)
        {
            ALOGE("Error, prompt speech token len %d < 75", prompt_token_len);
            return -1;
        }
        // CosyVoice3: Only support 75 prompt tokens for now
        int prompt_token_align_len = 75;

        std::vector<float> prompt_speech_embeds_flow1;
        // CosyVoice3: Use flow_embed_size (80) instead of hardcoded 512
        prompt_speech_embeds_flow1.insert(prompt_speech_embeds_flow1.begin(), prompt_speech_embeds_flow.begin(), prompt_speech_embeds_flow.begin()+prompt_token_align_len * lToken2Wav.flow_embed_size);

        std::vector<float> prompt_feat1;
        prompt_feat1.insert(prompt_feat1.begin(), prompt_feat.begin(), prompt_feat.begin()+prompt_token_align_len*2*80);

        int promot_token_pad = 0;
        int this_token_hop_len;
        int i=0;
        while (true) {
            this_token_hop_len = (token_offset == 0)? lToken2Wav.token_hop_len + promot_token_pad : lToken2Wav.token_hop_len;

            std::unique_lock<std::mutex> lock(g_buffer_mutex);

            g_buffer_cv.wait(lock, [&] {
                return (g_token_buffer.size() - token_offset >= this_token_hop_len + lToken2Wav.pre_lookahead_len) || \
                        g_llm_finished.load() ||\
                        g_stop.load();
            });

            if(g_stop)
            {
                lock.unlock();
                break;
            }
            else if (g_token_buffer.size() - token_offset >= this_token_hop_len + lToken2Wav.pre_lookahead_len ) {

                std::vector<SpeechToken> token;
                int start = token_offset -  std::min( int(token_offset / lToken2Wav.token_hop_len), lToken2Wav.max_infer_chunk_num-1) * lToken2Wav.token_hop_len;
                int end = token_offset + this_token_hop_len + lToken2Wav.pre_lookahead_len;

                token.insert(token.end(), g_token_buffer.begin()+start, g_token_buffer.begin()+end);

                lock.unlock();

                std::cout << "[Main/Token2Wav Thread] Processing batch of " << token.size() << " tokens...\n";
                auto speech = lToken2Wav.infer(token, prompt_speech_embeds_flow1, prompt_feat1, spk_embeds, token_offset, false);
                token_offset += this_token_hop_len;

                output.insert(output.end(), speech.begin(), speech.end());
                std::string path = "output_"+std::to_string(i)+".wav";

                saveVectorAsWavFloat(speech, path, 24000, 1);
                i += 1;

            }

            else if (g_llm_finished.load() ) {
                std::cout << "[Main/Token2Wav Thread] Buffer is empty and LLM finished. Exiting.\n";
                lock.unlock();
                break;
            }
            else {
                lock.unlock();
            }

        }

        if (llm_thread.joinable()) {
            llm_thread.join();
        }

        if(g_stop)
        {
            g_token_buffer.erase(g_token_buffer.begin(), g_token_buffer.end());
            return 1;
        }

        std::vector<SpeechToken> token;
        int start = g_token_buffer.size() - std::min( int(g_token_buffer.size() / lToken2Wav.token_hop_len), lToken2Wav.max_infer_chunk_num-1) * lToken2Wav.token_hop_len;
        token.insert(token.end(), g_token_buffer.begin() + start, g_token_buffer.end());
        auto speech = lToken2Wav.infer(token, prompt_speech_embeds_flow1, prompt_feat1, spk_embeds, token_offset - start, true);
        output.insert(output.end(), speech.begin(), speech.end());
        std::string path = "output_"+std::to_string(i)+".wav";
        saveVectorAsWavFloat(speech, path, 24000, 1);
        saveVectorAsWavFloat(output, "output.wav", 24000, 1);

        float total_s = time_total.cost()/1000;
        float audio_s = output.size() / 24000.0f;
        float rtf = total_s / audio_s;
        ALOGI("tts total: %.3f s, audio: %.2f s, RTF: %.2fx", total_s, audio_s, rtf);
        ALOGI("LLM tokens: %d, Token2Wav chunks: %d", (int)g_token_buffer.size(), i);
        reset();
        std::cout << "\nVoice generation pipeline completed.\n";

    } catch (const std::exception& e) {
        std::cerr << "Error in pipeline: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}


// --- Daemon Mode: Accept TTS requests via Unix socket ---
// Protocol:
//   Client → Server: [4 bytes msg_len] [msg_len bytes: text string (UTF-8)]
//   Server → Client: [4 bytes result_len] [result_len bytes: JSON response]
//     Success: {"ok":true,"wav":"output.wav","rtf":1.23,"tokens":170,"audio_s":5.5}
//     Error:   {"ok":false,"error":"message"}
static bool recv_all(int fd, void* buf, size_t len) {
    size_t got = 0;
    while (got < len) {
        ssize_t n = read(fd, (char*)buf + got, len - got);
        if (n <= 0) return false;
        got += n;
    }
    return true;
}

static void send_response(int fd, const std::string& json) {
    uint32_t len = (uint32_t)json.size();
    write(fd, &len, 4);
    write(fd, json.c_str(), len);
}

int daemon_serve(
    const std::string& daemon_socket_path,
    std::vector<unsigned short>& prompt_text_embeds,
    std::vector<unsigned short>& prompt_speech_embeds,
    std::vector<float>& prompt_feat,
    std::vector<float>& prompt_speech_embeds_flow,
    std::vector<float>& spk_embeds)
{
    // Remove stale socket
    unlink(daemon_socket_path.c_str());

    int listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (listen_fd < 0) {
        fprintf(stderr, "[Daemon] socket() failed: %s\n", strerror(errno));
        return -1;
    }
    g_daemon_fd = listen_fd;

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, daemon_socket_path.c_str(), sizeof(addr.sun_path) - 1);

    if (bind(listen_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        fprintf(stderr, "[Daemon] bind(%s) failed: %s\n", daemon_socket_path.c_str(), strerror(errno));
        close(listen_fd);
        return -1;
    }
    chmod(daemon_socket_path.c_str(), 0666);

    if (listen(listen_fd, 2) < 0) {
        fprintf(stderr, "[Daemon] listen() failed: %s\n", strerror(errno));
        close(listen_fd);
        return -1;
    }

    printf("[Daemon] Listening on %s\n", daemon_socket_path.c_str());
    printf("[Daemon] Ready for TTS requests.\n");
    fflush(stdout);

    int request_count = 0;
    while (!g_stop.load()) {
        // Accept with timeout for clean shutdown
        fd_set fds;
        FD_ZERO(&fds);
        FD_SET(listen_fd, &fds);
        struct timeval tv = {1, 0};  // 1 second timeout
        int sel = select(listen_fd + 1, &fds, nullptr, nullptr, &tv);
        if (sel <= 0) continue;

        int conn_fd = accept(listen_fd, nullptr, nullptr);
        if (conn_fd < 0) {
            if (g_stop.load()) break;
            continue;
        }

        request_count++;
        printf("\n[Daemon] --- Request #%d ---\n", request_count);

        // Read message length
        uint32_t msg_len = 0;
        if (!recv_all(conn_fd, &msg_len, 4) || msg_len > 65536) {
            fprintf(stderr, "[Daemon] Invalid message length: %u\n", msg_len);
            send_response(conn_fd, "{\"ok\":false,\"error\":\"invalid message\"}");
            close(conn_fd);
            continue;
        }

        // Read text
        std::string text(msg_len, '\0');
        if (!recv_all(conn_fd, &text[0], msg_len)) {
            fprintf(stderr, "[Daemon] Failed to read message body\n");
            send_response(conn_fd, "{\"ok\":false,\"error\":\"read failed\"}");
            close(conn_fd);
            continue;
        }

        printf("[Daemon] Text: '%s'\n", text.substr(0, 60).c_str());

        // Reset state for this request
        g_stop = false;
        g_llm_finished = false;
        g_token_buffer.clear();

        // Run TTS (tts() prints stats and calls reset() internally)
        int ret = tts(text, prompt_text_embeds, prompt_speech_embeds,
                      prompt_feat, prompt_speech_embeds_flow, spk_embeds);

        if (ret == 0) {
            send_response(conn_fd, "{\"ok\":true,\"wav\":\"output.wav\"}");
        } else {
            send_response(conn_fd, "{\"ok\":false,\"error\":\"tts failed\"}");
        }

        close(conn_fd);
    }

    close(listen_fd);
    unlink(daemon_socket_path.c_str());
    g_daemon_fd = -1;
    printf("[Daemon] Stopped.\n");
    return 0;
}

int main(int argc, char *argv[])
{
    signal(SIGPIPE, SIG_IGN);
    signal(SIGINT, __sigExit);
    LLMAttrType attr;
    std::string text = "Привет, как дела? Сегодня хорошая погода для прогулки.";
    bool b_continue = true;

    cmdline::parser cmd;
    cmd.add<std::string>("text", 't', "text", true, text);
    cmd.add<std::string>("token2wav_axmodel_dir", 0, "token2wav axmodel path template", false, "");
    cmd.add<std::string>("template_filename_axmodel", 0, "axmodel path template", false, attr.template_filename_axmodel);
    cmd.add<std::string>("filename_post_axmodel", 0, "post axmodel path", false, attr.filename_post_axmodel);
    cmd.add<std::string>("filename_decoder_axmodel", 0, "decoder axmodel path", false, attr.filename_decoder_axmodel);
    cmd.add<std::string>("filename_tokenizer_model", 0, "tokenizer model path", false, attr.filename_tokenizer_model);
    cmd.add<std::string>("filename_tokens_embed", 0, "tokens embed path", false, attr.filename_tokens_embed);
    cmd.add<std::string>("filename_llm_embed", 0, "llm embed path", false, attr.filename_llm_embed);
    cmd.add<std::string>("filename_speech_embed", 0, "speech embed path", false, attr.filename_speech_embed);
    cmd.add<std::string>("prompt_files", 0, "prompt files dir", false, "prompt_files");

    cmd.add<bool>("bos", 0, "", false, attr.b_bos);
    cmd.add<bool>("eos", 0, "", false, attr.b_eos);
    cmd.add<int>("axmodel_num", 0, "num of axmodel(for template)", false, attr.axmodel_num);
    cmd.add<int>("n_timesteps", 'ts', "num of time steps", false, 7);
    cmd.add<int>("eos_token_id", 0, "EOS token ID (any token >= this means stop)", false, attr.eos_token_id);
    cmd.add<int>("speech_embed_num", 0, "speech embedding table size", false, attr.speech_embed_num);
    cmd.add<bool>("continue", 0, "continuous dialogue", false, b_continue);
    cmd.add<std::string>("devices", 0, "AXCL device IDs (comma-separated)", false, "0,");
    cmd.add<float>("temperature", 0, "LLM sampling temperature (<1.0 sharpens distribution)", false, 1.0f);
    cmd.add<int>("top_k", 0, "top-k for nucleus sampling", false, 25);
    cmd.add<std::string>("filename_decoder_weight", 0, "CPU decoder weight (float32, 6761*896*4 bytes) - bypasses NPU decoder", false, "");
    cmd.add<std::string>("external_tokens", 0, "Unix socket path for external RKLLM token server (dual-NPU mode)", false, "");
    cmd.add<std::string>("daemon", 0, "Daemon mode: listen for TTS requests on this Unix socket path", false, "");
    cmd.add<std::string>("onnx_hift", 0, "Unix socket path for ONNX HiFT server (replaces NPU HiFT with CPU ONNX)", false, "");
    cmd.add<bool>("onnx_fullmel", 0, "Full-mel mode: accumulate mel, run HiFT once (no crossfade artifacts)", false, false);

    cmd.parse_check(argc, argv);

    text = cmd.get<std::string>("text");

    attr.filename_tokenizer_model = cmd.get<std::string>("filename_tokenizer_model");
    attr.filename_tokens_embed = cmd.get<std::string>("filename_tokens_embed");
    attr.filename_llm_embed = cmd.get<std::string>("filename_llm_embed");
    attr.filename_speech_embed = cmd.get<std::string>("filename_speech_embed");
    attr.filename_post_axmodel = cmd.get<std::string>("filename_post_axmodel");
    attr.filename_decoder_axmodel = cmd.get<std::string>("filename_decoder_axmodel");
    attr.template_filename_axmodel = cmd.get<std::string>("template_filename_axmodel");

    attr.b_bos = cmd.get<bool>("bos");
    attr.b_eos = cmd.get<bool>("eos");
    attr.axmodel_num = cmd.get<int>("axmodel_num");
    attr.eos_token_id = cmd.get<int>("eos_token_id");
    attr.speech_embed_num = cmd.get<int>("speech_embed_num");
    std::string token2wav_axmodel_dir = cmd.get<std::string>("token2wav_axmodel_dir");
    int n_timesteps = cmd.get<int>("n_timesteps");
    std::string prompt_files = cmd.get<std::string>("prompt_files");

    b_continue = cmd.get<bool>("continue");
    attr.temperature = cmd.get<float>("temperature");
    attr.top_k = cmd.get<int>("top_k");
    attr.filename_decoder_weight = cmd.get<std::string>("filename_decoder_weight");
    g_external_tokens_socket = cmd.get<std::string>("external_tokens");
    g_daemon_socket = cmd.get<std::string>("daemon");
    std::string onnx_hift_socket = cmd.get<std::string>("onnx_hift");
    bool onnx_fullmel = cmd.get<bool>("onnx_fullmel");

    bool external_mode = !g_external_tokens_socket.empty();
    if (external_mode) {
        printf("=== DUAL-NPU MODE ===\n");
        printf("External token server: %s\n", g_external_tokens_socket.c_str());
        printf("LLM runs on RK3588 RKLLM, Token2Wav on AX650N\n\n");
    }

    // Parse AXCL device IDs
    auto devices_str = cmd.get<std::string>("devices");
    std::vector<int> devices;
    {
        std::stringstream ss(devices_str);
        std::string item;
        while (std::getline(ss, item, ',')) {
            if (!item.empty())
                devices.push_back(std::stoi(item));
        }
    }
    int devid = devices.empty() ? 0 : devices[0];
    attr.dev_id = devid;

    // Initialize AXCL runtime (PCIe connection to AX650N)
    printf("Initializing AXCL device %d...\n", devid);
    // Must call axclInit before any other AXCL API
    axclError axret = axclInit(nullptr);
    if (axret != 0)
    {
        fprintf(stderr, "axclInit() failed with error %d\n", axret);
        return -1;
    }
    axret = axcl_Init(devid);
    if (axret != 0)
    {
        fprintf(stderr, "axcl_Init(%d) failed with error %d\n", devid, axret);
        return -1;
    }

    // In external mode, skip LLM model loading (saves ~1.4GB NPU memory)
    if (!external_mode) {
        if (!lLaMa.Init(attr))
        {
            axcl_Exit(devid);
            return -1;
        }
    } else {
        printf("Skipping LLM init (external token mode)\n");
    }

    if (!lToken2Wav.Init(token2wav_axmodel_dir, n_timesteps, devid))
    {
        return -1;
    }
    if (!onnx_hift_socket.empty()) {
        lToken2Wav.set_onnx_hift_socket(onnx_hift_socket, onnx_fullmel);
    }
    ALOGI();
    // for llm
    std::vector<int> prompt_text_token;
    std::vector<unsigned short> prompt_text_embeds;
    std::vector<int> prompt_speech_token;
    std::vector<unsigned short> prompt_speech_embeds;

    // for flow
    std::vector<float> prompt_feat;
    std::vector<float> prompt_speech_embeds_flow;
    std::vector<float> spk_embeds;

    readtxt(prompt_files+"/prompt_text.txt", prompt_text_token);
    readtxt(prompt_files+"/llm_prompt_speech_token.txt", prompt_speech_token);
    readtxt(prompt_files+"/prompt_speech_feat.txt", prompt_feat);
    readtxt<float>(prompt_files+"/flow_embedding.txt", spk_embeds);

    if (!external_mode) {
        lLaMa.TextToken2Embeds(prompt_text_token, prompt_text_embeds);
        lLaMa.SpeechToken2Embeds(prompt_speech_token, prompt_speech_embeds);
    } else {
        // Store raw tokens for RKLLM server requests
        g_prompt_text_tokens = prompt_text_token;
        g_prompt_speech_tokens = prompt_speech_token;
    }
    lToken2Wav.SpeechToken2Embeds(prompt_speech_token, prompt_speech_embeds_flow);

    if (!g_daemon_socket.empty()) {
        // Daemon mode: listen for requests on Unix socket
        printf("=== DAEMON MODE ===\n");
        printf("Listening on: %s\n\n", g_daemon_socket.c_str());
        daemon_serve(g_daemon_socket,
                     prompt_text_embeds, prompt_speech_embeds,
                     prompt_feat, prompt_speech_embeds_flow, spk_embeds);
    } else {
        // Normal mode: process initial text and/or interactive stdin
        if(text.size()>0)
        {
            tts(
                // for llm
                text, prompt_text_embeds,prompt_speech_embeds,
                // for flow
                prompt_feat, prompt_speech_embeds_flow, spk_embeds
            );
        }

        if (b_continue)
        {
            printf("Type \"q\" to exit, Ctrl+c to stop current running\n");
        }

        while (b_continue)
        {
            if(g_stop)
            {
                break;
            }

            printf("text >> ");
            fflush(stdout);
            std::getline(std::cin, text);
            if (text == "q")
            {
                break;
            }
            if (text == "")
            {
                continue;
            }

            fflush(stdout);

            tts(
                // for llm
                text, prompt_text_embeds,prompt_speech_embeds,
                // for flow
                prompt_feat, prompt_speech_embeds_flow, spk_embeds
            );
        }
    }

    if (!external_mode) {
        lLaMa.Deinit();
    }
    lToken2Wav.Deinit();

    // Cleanup AXCL runtime
    axcl_Exit(devid);
    axclFinalize();

    return 0;
}
