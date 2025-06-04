# model_confidence
a minimal tool for generating a "confidence score" from a LLM's next token distribution

this all began with my friend's question, "why can't LLMs know when they're hallucinating?"

my thought was that you could try and get a sense of a model's confidence based on its token distribution.
intuition: tighter distribution = more confidence, wider distribution = less confidence

### how it works


TODO:
- add executable to makefile
- accept command-line arguments (model path, window, input string)
- benchmark (full logit sort vs top n)


### Instructions
1. Clone repo i.e. `git clone https://github.com/szge/model_confidence.git`
2. Download the model of your choice; I used `huggingface-cli` on Python to install qwen2.5-7b quantized
    ```shell
    huggingface-cli download Qwen/Qwen2.5-7B-Instruct-GGUF \
        --include "qwen2.5-7b-instruct-q8_0-000*.gguf" \
        --local-dir ./qwen2.5-7b/
    ```
3. Clone the llama.cpp GitHub repo in the project's root directory; it should exist under model_confidence/llama.cpp (I build it as a dynamic library)
    ```shell
    git clone https://github.com/ggml-org/llama.cpp.git
    cmake -B build
    cmake --build build --config Release
    ```
4. Run the CMake project in the repository root