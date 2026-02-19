// Redirect to CosyVoice2.Axera Tokenizer (different API from ax-llm-axcl)
// CV2: Init(path, bos, eos), Encode(text, ImageInfo)
// ax-llm: Init(path), Encode(text, last_reply, tokens, tokens_diff, b_img_prompt)
#pragma once

// Prevent ax-llm's Tokenizer.hpp from being included
#ifdef _TOKENIZER_HPP_
#error "ax-llm Tokenizer.hpp was included before CV2_Tokenizer.hpp"
#endif

#include <string>
#include <vector>
#include <memory>

enum TokenizerType
{
    TKT_LLaMa,
    TKT_Qwen,
    TKT_HTTP,
    TKT_Phi3,
    TKT_END
};

struct ImageInfo
{
    int imgsz = 448;
    int num_img = 1;
    bool img_prompt = false;
    int img_token_num = -1;
};

class BaseTokenizer
{
public:
    virtual bool Init(std::string model_path, bool b_bos = true, bool b_eos = false) = 0;
    virtual bool Encode(std::string input, std::vector<int> &output, ImageInfo img_info) = 0;
    virtual std::vector<int> Encode(std::string input, ImageInfo img_info) = 0;
    virtual std::string Decode(const std::vector<int> input) = 0;
    virtual int GetBosID() = 0;
    virtual int GetEosID() = 0;
    virtual int GetImgStartID() = 0;
    virtual int GetImgContextID() = 0;

    virtual bool isEnd(int id) { return id == GetEosID(); }
};

std::shared_ptr<BaseTokenizer> CreateTokenizer(TokenizerType type);
