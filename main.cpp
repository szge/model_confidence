#include "llama-cpp.h"
#include "llama.cpp/src/llama-vocab.h" // to get vocab size
#include <cstdio>
#include <string>
#include <vector>
#include <iostream>
#include <numeric>
#include <span>

llama_model* init_model()
{
    const std::string model_path = "./qwen2.5-7b/qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf";
    constexpr int num_gpu_layers = 99;
    // initialize the model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = num_gpu_layers;

    llama_model* model = llama_model_load_from_file(model_path.c_str(), model_params);
    return model;
}

llama_context* init_context(const int n_prompt, const int num_tokens_to_predict, llama_model* model)
{
    llama_context_params ctx_params = llama_context_default_params();
    // n_ctx is the context size
    ctx_params.n_ctx = n_prompt + num_tokens_to_predict - 1;
    // n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
    ctx_params.n_batch = n_prompt;
    // enable performance counters
    ctx_params.no_perf = false;

    llama_context* ctx = llama_init_from_model(model, ctx_params);
    return ctx;
}

llama_sampler* init_sampler()
{
    auto sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = false;
    llama_sampler* smpl = llama_sampler_chain_init(sampler_params);
    return smpl;
}

int print_prompt_tokens(const std::vector<llama_token>& prompt_tokens, const llama_vocab* vocab)
{
    for (const auto id : prompt_tokens) {
        char buf[128];
        const int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
            return 1;
        }
        std::string s(buf, n);
        printf("%s", s.c_str());
    }
    return 0;
}

void print_best_token(std::span<float> logits_view, const llama_vocab* vocab)
{
    const auto it_best = std::ranges::max_element(logits_view);
    const int best_id = std::distance(logits_view.begin(), it_best);
    // float best_logit = *it_best;
    char best_piece[64] = {0}; // null-terminated C-string
    llama_token_to_piece(
        vocab,
        (llama_token) best_id,
        best_piece,
        sizeof(best_piece),
        0,
        false
    );

    std::cout << "The current best token is: \'" << best_piece << "\'.\n\n";
}

float logit_to_prob(const float logit)
{
    const float odds = exp(logit);
    return odds / (1 + odds);
}

std::vector<float> get_top_n_logits(std::span<float> logits, const int n = 50)
{
    // returns the top n logits in descending order
    if (n <= 0) return {};
    if (n >= logits.size())
    {
        std::vector<float> all_logits(logits.begin(), logits.end());
        std::ranges::sort(all_logits, std::greater<float>());
        return all_logits;
    }
    // use a min heap so we have fast access to the lowest priority element
    std::priority_queue<float, std::vector<float>, std::greater<>> min_heap;
    for (int i = 0; i < n; i++)
    {
        min_heap.push(logits[i]);
    }

    for (int i = n; i < logits.size(); i++)
    {
        const float value = logits[i];
        if (value > min_heap.top())
        {
            min_heap.pop();
            min_heap.push(value);
        }
    }

    std::vector<float> top_n;
    top_n.reserve(n);
    while (!min_heap.empty())
    {
        top_n.push_back(min_heap.top());
        min_heap.pop();
    }

    // sort in descending order
    std::sort(top_n.begin(), top_n.end(), std::greater<float>());
    return top_n;
}

double get_model_confidence_score(const std::span<float> logits_view)
{
    // given the current logits, compute a confidence score.
    // the confidence score is based on the top N tokens.
    // if the model has 100% confidence in the top token, this will return a score of 1.
    // if the model has a uniform distribution, this will return a score of 0.
    // we use the top N (= 10 or 50 etc.) instead of the whole token vocab (which might be >100k)
    constexpr int score_window = 50;
    const std::vector<float> top_n_logits = get_top_n_logits(logits_view, score_window);
    // std::cout << "Top N logits:\n";
    // for (const auto logit : top_n_logits)
    // {
    //     std::cout << logit << " ";
    // }
    // std::cout << "\n\n";

    // convert logits to a probability distribution but just for the top n tokens
    std::vector<float> probs(score_window);
    probs.reserve(score_window);
    const float L_max = top_n_logits[0]; // use max logit for numerical stability

    double normalizer = 0.0; // double for precision
    double score = 0.0;
    for (int i = 0; i < score_window; i++)
    {
        const float shifted = top_n_logits[i] - L_max;
        const float e = exp(shifted);
        normalizer += e;
        probs[i] = e;
    }
    for (int i = 0; i < score_window; i++)
    {
        probs[i] /= static_cast<float>(normalizer);
        score += probs[i] / probs[0] / score_window;
    }

    return score;
}

int main() {
    const std::string prompt = "The 16th president of the United States was ";
    constexpr int num_tokens_to_predict = 3;
    std::vector<double> confidence_scores(num_tokens_to_predict);
    confidence_scores.reserve(num_tokens_to_predict);

    // load dynamic backends
    ggml_backend_load_all();

    llama_model* model = init_model();

    if (model == nullptr) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);

    // tokenize the prompt

    // find the number of tokens in the prompt
    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);

    // allocate space for the tokens and tokenize the prompt
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "%s: error: failed to tokenize the prompt\n", __func__);
        return 1;
    }

    // initialize the context
    llama_context* ctx = init_context(n_prompt, num_tokens_to_predict, model);

    if (ctx == nullptr) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // initialize the sampler
    llama_sampler* smpl = init_sampler();
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // print the prompt token-by-token
    const int print_prompt_status = print_prompt_tokens(prompt_tokens, vocab);
    if (print_prompt_status) return print_prompt_status;

    // prepare a batch for the prompt
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    // main loop
    const auto t_main_start = ggml_time_us();
    int n_decode = 0;
    llama_token new_token_id;

    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + num_tokens_to_predict; ) {
        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
        }

        n_pos += batch.n_tokens;

        // sample the next token
        {
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // is it an end of generation?
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            char buf[128];
            const int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return 1;
            }
            std::string s(buf, n);
            printf("%s", s.c_str());
            fflush(stdout);

            float* logits = llama_get_logits(ctx);
            const std::span logits_view(logits, vocab->n_tokens());
            // print_best_token(logits_view, vocab);
            const double confidence_score = get_model_confidence_score(logits_view);
            // std::cout << "\ntoken: \'" << s.c_str() << "\', score: " << confidence_score << "\n";
            // should be the number of tokens generated after the prompt; range from 0 to num_tokens_to_predict - 1
            const int curr_gen_idx = n_pos - n_prompt;
            // the main attraction
            confidence_scores[curr_gen_idx] = get_model_confidence_score(logits_view);

            // prepare the next batch with the sampled token
            batch = llama_batch_get_one(&new_token_id, 1);

            n_decode += 1;
        }
    }

    printf("\n");

    const auto t_main_end = ggml_time_us();

    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));

    fprintf(stderr, "\n");
    llama_perf_sampler_print(smpl);
    llama_perf_context_print(ctx);
    fprintf(stderr, "\n");

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    const double score_sum = std::accumulate(confidence_scores.begin(), confidence_scores.end(), 0.0);
    const double average_score = score_sum / static_cast<double>(confidence_scores.size());
    std::cout << "Average model confidence score: " << average_score;

    return 0;
}
