#include "llama-cpp.h"
#include "llama.cpp/src/llama-vocab.h" // to get vocab size
#include <cstdio>
#include <string>
#include <vector>
#include <iostream>
#include <numeric>
#include <span>

static void print_usage(int, char ** argv) {
    printf("\nexample usage:\n");
    printf("\n    %s -m model.gguf\n", argv[0]);
    printf("\n");
}

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

llama_context* init_context(llama_model* model)
{
    constexpr int n_ctx = 2048;
    // initialize the context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_batch = n_ctx;
    return llama_init_from_model(model, ctx_params);
}

llama_sampler* init_sampler()
{
    auto sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = false;
    llama_sampler* smpl = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(smpl, llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.8f));
    llama_sampler_chain_add(smpl, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));
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
    double score = 1.0;
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
        score -= probs[i] / probs[0] / score_window;
    }

    return score;
}

int main(int argc, char** argv) {
    std::string model_path = "./qwen2.5-7b/qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf";
    std::vector<double> confidence_scores {};

    // parse command line args
    for (int i = 1; i < argc; i++) {
        try {
            if (strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    model_path = argv[++i];
                } else {
                    print_usage(argc, argv);
                    return 1;
                }
            } else {
                print_usage(argc, argv);
                return 1;
            }
        } catch (std::exception & e) {
            fprintf(stderr, "error: %s\n", e.what());
            print_usage(argc, argv);
            return 1;
        }
    }
    if (model_path.empty()) {
        print_usage(argc, argv);
        return 1;
    }

    // only print errors
    llama_log_set([](enum ggml_log_level level, const char * text, void * /* user_data */) {
        if (level >= GGML_LOG_LEVEL_ERROR) {
            fprintf(stderr, "%s", text);
        }
    }, nullptr);

    // load dynamic backends
    ggml_backend_load_all();

    llama_model* model = init_model();

    if (model == nullptr) {
        fprintf(stderr , "%s: error: unable to load model\n" , __func__);
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);

    // initialize the context
    llama_context* ctx = init_context(model);
    if (!ctx) {
        fprintf(stderr , "%s: error: failed to create the llama_context\n" , __func__);
        return 1;
    }

    // initialize the sampler
    llama_sampler* smpl = init_sampler();

    // helper function to evaluate a prompt and generate a response
    auto generate = [&](const std::string & prompt) {
        std::string response;

        const bool is_first = llama_kv_self_seq_pos_max(ctx, 0) == 0;

        // tokenize the prompt
        const int n_prompt_tokens = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, is_first, true);
        std::vector<llama_token> prompt_tokens(n_prompt_tokens);
        if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), is_first, true) < 0) {
            GGML_ABORT("failed to tokenize the prompt\n");
        }

        // prepare a batch for the prompt
        llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
        llama_token new_token_id;
        while (true) {
            // check if we have enough space in the context to evaluate this batch
            int n_ctx = llama_n_ctx(ctx);
            int n_ctx_used = llama_kv_self_seq_pos_max(ctx, 0);
            if (n_ctx_used + batch.n_tokens > n_ctx) {
                printf("\033[0m\n");
                fprintf(stderr, "context size exceeded\n");
                exit(0);
            }

            if (llama_decode(ctx, batch)) {
                GGML_ABORT("failed to decode\n");
            }

            // sample the next token
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // is it an end of generation?
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            // convert the token to a string, print it and add it to the response
            char buf[256];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, true);
            if (n < 0) {
                GGML_ABORT("failed to convert token to piece\n");
            }
            std::string piece(buf, n);
            printf("%s", piece.c_str());
            fflush(stdout);
            response += piece;

            float* logits = llama_get_logits(ctx);
            const std::span logits_view(logits, vocab->n_tokens());
            // the main attraction
            confidence_scores.push_back(get_model_confidence_score(logits_view));

            // prepare the next batch with the sampled token
            batch = llama_batch_get_one(&new_token_id, 1);
        }

        return response;
    };

    std::vector<llama_chat_message> messages;
    std::vector<char> formatted(llama_n_ctx(ctx));
    int prev_len = 0;
    constexpr int max_turns = 1; // we only need 1 turn to get the model's response to the question
    for (int turns = 0; turns < max_turns; turns++) {
        const char * tmpl = llama_model_chat_template(model, /* name */ nullptr);

        // On the very first turn, inject a system prompt
        if (messages.empty()) {
            // we need strdup since we free later
            messages.push_back({"system", strdup("You are a helpful assistant that always replies concisely. Only reply with the direct answer to the question, without asking follow ups. Keep your answer as short as possible. Always attempt to respond.")});
            int new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
            if (new_len > (int)formatted.size()) {
                formatted.resize(new_len);
                new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
            }
            if (new_len < 0) {
                fprintf(stderr, "failed to apply the chat template\n");
                return 1;
            }
        }
        // get user input
        printf("\033[32m> \033[0m");
        std::string user;
        std::getline(std::cin, user);

        if (user.empty()) {
            break;
        }

        // add the user input to the message list and format it
        messages.push_back({"user", strdup(user.c_str())});
        int new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
        if (new_len > (int)formatted.size()) {
            formatted.resize(new_len);
            new_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), true, formatted.data(), formatted.size());
        }
        if (new_len < 0) {
            fprintf(stderr, "failed to apply the chat template\n");
            return 1;
        }

        // remove previous messages to obtain the prompt to generate the response
        std::string prompt(formatted.begin() + prev_len, formatted.begin() + new_len);

        // generate a response
        printf("\033[33m");
        std::string response = generate(prompt);
        printf("\n\033[0m");

        // add the response to the messages
        messages.push_back({"assistant", strdup(response.c_str())});
        prev_len = llama_chat_apply_template(tmpl, messages.data(), messages.size(), false, nullptr, 0);
        if (prev_len < 0) {
            fprintf(stderr, "failed to apply the chat template\n");
            return 1;
        }
    }

    // free resources
    for (auto & msg : messages) {
        free(const_cast<char *>(msg.content));
    }
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);

    const double score_sum = std::accumulate(confidence_scores.begin(), confidence_scores.end(), 0.0);
    const double average_score = score_sum / static_cast<double>(confidence_scores.size());
    std::cout << "Average model confidence score: " << average_score;

    return 0;
}
