# model_confidence
a minimal tool for generating a "confidence score" from a LLM's next token distribution

this all began with my friend's question, "why can't LLMs know when they're hallucinating?"

my thought was that you could try and get a sense of a model's confidence based on its token distribution.
intuition: tighter distribution = more confidence, wider distribution = less confidence

### example usage
```shell
jchiang@Jonathans-MacBook-Pro model_confidence % ./cmake-build-debug/model_confidence -m ./qwen2.5-7b/qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf
.......................................................................................
> 2 + 2 = 
4
Average model confidence score: 0.98
jchiang@Jonathans-MacBook-Pro model_confidence % ./cmake-build-debug/model_confidence -m ./qwen2.5-7b/qwen2.5-7b-instruct-q8_0-00001-of-00003.gguf
.......................................................................................
> 53982 * 8889 = 
478209022
Average model confidence score: 0.946323
```
so the model is more "confident" about the calculation `2 + 2 = 4` rather than the more complex calculation (which is the wrong result)

### how it works

how the score for a token is calculated:
1. look at the last layer logits and get the top N highest-likelihood tokens (using a priority queue), and sort in descending order
2. convert the top N logits to a probability distribution; the probability token $i$ is selected is $p_i = \frac{\exp{l_i}}{\sum_{j = 1}^{M} \exp{l_j}}$
3. normalize the resulting logit-probability graph to a unit square (i.e. multiply the index by 1/N and divide all probabilities by the highest one)
4. the score is one minus the area under the curve
   1. the intuition is that uniform distributions transform into the unit square, resulting in an area of 1.
   2. if the model is 100% confident about a token, then the area under the curve is very small.

the final score is the average confidence score among the generated tokens


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

### todo

- [x] collect average over tokens
- [x] add prompt templating
- [x] accept command-line arguments (model path, window, input string)
- [ ] add executable to makefile
- [ ] benchmark (full logit sort vs top n)
- [ ] display generated tokens with confidence score red/green highlight