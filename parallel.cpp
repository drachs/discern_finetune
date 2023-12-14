// A basic application simulating a server with multiple clients.
// The clients submit requests to the server and they are processed in parallel.

#include "common.h"
#include "llama.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>
#include <ctime>

// trim whitespace from the beginning and end of a string
static std::string trim(const std::string & str) {
    size_t start = 0;
    size_t end = str.size();

    while (start < end && isspace(str[start])) {
        start += 1;
    }

    while (end > start && isspace(str[end - 1])) {
        end -= 1;
    }

    return str.substr(start, end - start);
}

static std::string k_system =
R"(### Question:
You are a Student Web Activity Analyzer developed to support professionals, including Social Workers, School Psychologists, District Administrators, School Safety Specialists, and related roles. Your primary objective is to meticulously evaluate the online activity of K-12 students and identify specific indicators related to their interests and passions. For each identified indicator, provide a JSON object containing:

Presence: A value of 1 (if the indicator is present) or 0 (if not). Mark as 1 even if only part of the data aligns with an indicator.
Confidence: Provide a confidence level on a scale of 1-10 to indicate your level of certainty in the analysis.
Note: Include information on the logic used to decide that certain indicators were identified and a summary of the analyzed web activity.

Consider patterns in the data, not just individual searches.

Consider that the online activity originates from a school-issued device.

If ambiguous use your best judgment, reflect uncertainty in confidence score.

Here are the specific indicators for this task:

'sports-and-athletics'
'environmentalism-and-sustainability'
'gaming-and-e-sports'
'college-and-career'
'cooking-and-food'
'reading-and-literature'
'writing-and-creative-writing'
'science-and-technology'
'mathematics-and-statistics'
'creative-arts'
'animals-and-nature'
'history-and-social-studies'

### Search Data:
)";

static std::string k_example =
R"(

### Example Output:
{
  "sports-and-athletics": "1",
  "sports-and-athletics-confidence": "6",
  "environmentalism-and-sustainability": "0",
  "environmentalism-and-sustainability-confidence": "8",
  "gaming-and-e-sports": "0",
  "gaming-and-e-sports-confidence": "10",
  "college-and-career": "1",
  "college-and-career-confidence": "7",
  "cooking-and-food": "1",
  "cooking-and-food-confidence": "7",
  "reading-and-literature": "0",
  "reading-and-literature-confidence": "7",
  "writing-and-creative-writing": "1",
  "writing-and-creative-writing-confidence": "8",
  "science-and-technology": "0",
  "science-and-technology-confidence": "10",
  "mathematics-and-statistics": "1",
  "mathematics-and-statistics-confidence": "6",
  "creative-arts": "1",
  "creative-arts-confidence": "8",
  "animals-and-nature": "1",
  "animals-and-nature-confidence": "9",
  "history-and-social-studies": "0",
  "history-and-social-studies-confidence": "9",
  "note": "Detailed Summary Goes Here "
}

### Solution:
)";

static std::vector<std::string> k_prompts = {
    "['scratch', 'youtube', 'faded alan walker', 'faded alan walker piano', 'hart and soul', 'relaxing songs', 'relaxing piano music', 'fun work music', 'youtube', 'fun work music', 'youtube', 'fun work music']",
    "['whens christmas', 'google earth', 'what did isaac newtons mom do', 'why did isaac newton want to be a scientist', 'luffy', 'luffy', 'fortnite', 'yapo meaning', 'is ink okto eat', 'a dragon', 'how to spell beginning', 'bored human.com', 'saturn']",
    "['google docs', 'what does edge mean', 'what does edge mean slang', 'edpu', 'what to do when bubbles appear in your screen protector', 'indictment meaning', 'blooket', 'bubble guppies', 'blooket', 'what does rebel mean', 'pow', 'powerschool']",
    "['what is meant by stage right and stage left?', 'how do body positions change the actors relationship to the audience?', 'what is the purpose of blocking play?', 'what is meant by the ternm dress stage?', 'how does an actor avoid upstaging another actor?', 'how does an actor avoid upstaging another actor?']",
    "['july', 'english', 'christmas', 'bloxorz', 'bloxorz math games', 'icy purple head', 'for colors', 'fourcolors']",
    "['clogs', 'google cricket', 'abcya', 'chick race game', 'allegory', 'walker scobell']",
    //"['eldrow', 'in which direction does earth rotate', '.which 2 planets in our solar system do not rotate in the same direction as earth?', 'unneccessary quotation marks', 'unnecessary quotation marks', 'big slime', 'big slime gang', 'supreme bart', 'depressed bart', 'bart lean', 'bart lean', 'bread', 'french toast', 'pasta', 'bloons td not monkey', 'bloons td 1', 'bloons td 2', '3bloons td 2', 'bloons td 3', 'bloons td 4', 'bloons td 5', 'bloons td 5 dart monkey', 'bloons td 6', 'bloons td 7 trailer', 'bloons td adventure time crossover', 'is bloons td 6 free on phone', 'is bloons td 6 free on epic games', 'whats the theater map in cod zombies', 'all bo3 zombie maps', 'adam ant', 'grope meaning', 'custom wordle', 'how many weeks are in between november 15 and june 20', 'ben franklin', 'ben franlin facts', 'ben franlin birth date', 'ben franklin death date', 'ben franklin facts', 'ahns house', 'n104m', 'n104m ahns house', 'buff guy from paranorman', 'paranorman', 'horror', 'fortnite', 'as seen from earth , how long does it take them moon to rotate on its axis', 'what does the other side of the moon like like', 'what does the other side of the moon look like', 'what does the other side of the moon look like ____ maria and more _______', 'eldrow', 'colorfle', 'what are all the spellin bee words today', 'surbian turtle video', 'persian turtle', 'persian tutrle', 'infinite connections', 'infinite connections game', 'infinite spelling bee']",
    "['nicolaus copernicus', 'nicolaus copernicus', 'google slides', 'english to japanese', '7 continents', '5 great lakes', 'dictionary', 'googl slides', 'hair types', 'rod wave concert', 'jhene aiko concert', 'scientific method', 'who cotinued the swatsikas in linked book', 'fegely middle school 7th grade teachers', 'maya winky without makeup']",
    //"['stars in cassiopeia', 'what is cassiopeia the goddess of', 'what is cassiopeia ', 'cassiopeia', 'how do you disable chromevox', 'sleeping beauties', 'mini crossword', 'what is the average height for a 13 year old girl', 'what is the average height for an adult male', 'rupaul', 'sleeping beauties', 'sleeping beauties plot', 'wikipedia', 'stephen king', 'stephen king books', 'cujo novel', 'pltw', 'eileen name meaning', 'behind the name', 'heileen', 'intj personality', 'cassiopeia', 'scientology', 'connections', 'mini crossword', 'antisemitism', 'is antisemitism rising with the israel war', 'is antisemitism racism', 'antisemitic celebrities', 'swastika merged with star of david', 'famous books that have rape', 'does fried green tomatoes have rape', 'mini crossword', 'google translate', 'connections answers today', 'come ___ at the seams', 'weird person with green glasses', 'senator of maine', 'inherit the wind', 'inherit the wind plot', 'white chicks movie', 'chiropractor', 'chiropractor pop therapy', 'how much does a penny weigh', 'sean definition', 'rick astley', 'what does eleanor mean', 'and i hope one day sommers rides her train', 'and i hope one day selmers rides her train', 'penelope scott rät lyrics', 'who is selmers', 'who was the subway guy', 'what was the subway guy charged with', 'chris brown', 'john smith', 'kaur meaning', 'hkb seattle', 'ass turd the office', 'astrid the office gif', 'mini crossword', 'any _ is good _', 'any __ is good __', 'any ___ is good __', 'what is something i can do to be less embarrassed', 'what are the most embarrassing things to do', 'stylish outfits for women', 'stylish outfits for women coquette', 'how to draw outfits', 'ginny and georgia season 3', 'what eating disorder do i have', 'what is it called when you eat non food items', 'google translate', 'curi.live pin', 'lyrics to sitting in a tree kissing', 'connections', 'airway definition', 'lungs definition', 'si=algxslauqlutylbfhlz-fairlhbbfugzb9a7j7_il_cpw_bwpfjc7bi94b7k9vtn66tkozprm71ttspm-rhwkallald8mwi812yshyo6yeksa7mnhtoomo-krp5lhyhoskot9yqv0wu-6pdmmdhgwivlumqd_ncubk5khl8jagdr2-qg7lrhavfkvlsg0z7taqjuterhvgea', 'how to make a fortune teller out of paper', 'vivien was here', 'eleanor kinn', 'mini crossword', 'constant critic nyt mini', 'itson nyt', 'calculator', 'lyrics to rat penelope scott', 'booboo stewart', 'ginny and georgia season 2 twist', 'does norah get engaged in ginny and georgia', 'diesel la torraca', 'nikki roumel', 'matt ziering', '5/8 ', '5/8', 'discrete vs continuous', 'discrete vs continuous data examples', '50/3', 'what is considered short person', 'what height is considered short for a kid', '5 ft people', 'fraction calculator', 'youre killing me smalls', 'kinn\\'s the medical assistant', 'kinn\\'s the medical assistant who wrote', 'mary kinn medical assistant', 'what is a beauty mark', 'is the input the domain', 'is antisemitism rising with the israel war', 'what percentage of the us population is jewish', 'mini crossword', 'adieu meaning', 'evergreen trees', 'farewell that literally means to god', 'connections september 1', 'pltw', 'my cousin vinny', 'ratatosk', 'ratatosk 231.github.io/i/', 'zenni', 'is russell stover good chocolateid', 'is russell stover good chocolate', 'abby ginny and georgia', 'eating disorder abby ginny and georgia', 'samantha ginny and georgia', 'max ginny and georgia', 'bad pickup lines', 'bad pickup lines dirty', 'natalie portman at 13', 'text features', 'connections', 'connections hint', 'connections hint 17', 'mini crossword', 'opposite of ssw', 'father of the national parks nyt', 'food safety org', 'food safety org nyt', 'scenic stretch of california\\'s coast', 'scenic stretch of maine\\'s coast', 'something added to stairs when a baby starts walking', 'types of phones', 'types of phones retro', 'hope against hope', 'hope against hope nyt', 'eleanor of aquitaine', 'bella hadid', 'gigi hadid', 'scratch games', 'did justin cheat on selena', 'kristen stewart', 'spencer (film)', 'blond mullet', 'connections unlimited', 'wikipedia malice mizer', 'malice', 'malice heather walter', 'malice plot twist', 'malice book', 'malice book plot twist', 'bat mitzvah', 'bat mitzvah established date', 'google translate', 'bella poarch', 'bella poarch m to the b', 'm to the b', 'millie b m to the b lyrics', 'connections']",
};

struct client {
    ~client() {
        if (ctx_sampling) {
            llama_sampling_free(ctx_sampling);
        }
    }

    int32_t id = 0;

    llama_seq_id seq_id = -1;

    llama_token sampled;

    int64_t t_start_prompt;
    int64_t t_start_gen;

    int32_t n_prompt  = 0;
    int32_t n_decoded = 0;
    int32_t i_batch   = -1;

    std::string input;
    std::string prompt;
    std::string response;

    struct llama_sampling_context * ctx_sampling = nullptr;
};

static void print_date_time() {
    std::time_t current_time = std::time(nullptr);
    std::tm* local_time = std::localtime(&current_time);
    char buffer[80];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", local_time);

    printf("\n\033[35mrun parameters as at %s\033[0m\n", buffer);
}

// Define a split string function to ...
static std::vector<std::string> split_string(const std::string& input, char delimiter) {
    std::vector<std::string> tokens;
    std::istringstream stream(input);
    std::string token;
    while (std::getline(stream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

int main(int argc, char ** argv) {
    srand(1234);

    gpt_params params;

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    // number of simultaneous "clients" to simulate
    const int32_t n_clients = params.n_parallel;

    // requests to simulate
    const int32_t n_seq = params.n_sequences;

    // insert new requests as soon as the previous one is done
    const bool cont_batching = params.cont_batching;

    const bool dump_kv_cache = params.dump_kv_cache;

#ifndef LOG_DISABLE_LOGS
    log_set_target(log_filename_generator("parallel", "log"));
    LOG_TEE("Log start\n");
    log_dump_cmdline(argc, argv);
#endif // LOG_DISABLE_LOGS

    // init llama.cpp
    llama_backend_init(params.numa);

    llama_model * model = NULL;
    llama_context * ctx = NULL;

    // load the target model
    params.logits_all = true;
    std::tie(model, ctx) = llama_init_from_gpt_params(params);

    // load the prompts from an external file if there are any
    if (params.prompt.empty()) {
        printf("\n\033[32mNo new questions so proceed with build-in defaults.\033[0m\n");
    } else {
        // Output each line of the input params.prompts vector and copy to k_prompts
        int index = 0;
        printf("\n\033[32mNow printing the external prompt file %s\033[0m\n\n", params.prompt_file.c_str());

        std::vector<std::string> prompts = split_string(params.prompt, '\n');
        for (const auto& prompt : prompts) {
            k_prompts.resize(index + 1);
            k_prompts[index] = prompt;
            index++;
            printf("%3d prompt: %s\n", index, prompt.c_str());
        }
    }

    fprintf(stderr, "\n\n");
    fflush(stderr);

    const int n_ctx = llama_n_ctx(ctx);

    std::vector<client> clients(n_clients);
    for (size_t i = 0; i < clients.size(); ++i) {
        auto & client = clients[i];
        client.id = i;
        client.ctx_sampling = llama_sampling_init(params.sparams);
    }

    std::vector<llama_token> tokens_system;
    tokens_system = ::llama_tokenize(ctx, k_system, true);
    const int32_t n_tokens_system = tokens_system.size();

    llama_seq_id g_seq_id = 0;

    // the max batch size is as large as the context to handle cases where we get very long input prompt from multiple
    // users. regardless of the size, the main loop will chunk the batch into a maximum of params.n_batch tokens at a time
    llama_batch batch = llama_batch_init(n_ctx, 0, 1);

    int32_t n_total_prompt = 0;
    int32_t n_total_gen    = 0;
    int32_t n_cache_miss   = 0;

    struct llama_kv_cache_view kvc_view = llama_kv_cache_view_init(ctx, n_clients);

    const auto t_main_start = ggml_time_us();

    LOG_TEE("%s: Simulating parallel requests from clients:\n", __func__);
    LOG_TEE("%s: n_parallel = %d, n_sequences = %d, cont_batching = %d, system tokens = %d\n", __func__, n_clients, n_seq, cont_batching, n_tokens_system);
    LOG_TEE("\n");

    {
        LOG_TEE("%s: Evaluating the system prompt ...\n", __func__);

        for (int32_t i = 0; i < n_tokens_system; ++i) {
            llama_batch_add(batch, tokens_system[i], i, { 0 }, false);
        }

        if (llama_decode(ctx, batch) != 0) {
            LOG_TEE("%s: llama_decode() failed\n", __func__);
            return 1;
        }

        // assign the system KV cache to all parallel sequences
        for (int32_t i = 1; i < n_clients; ++i) {
            llama_kv_cache_seq_cp(ctx, 0, i, 0, n_tokens_system);
        }

        LOG_TEE("\n");
    }

    LOG_TEE("Processing requests ...\n\n");

    while (true) {
        if (dump_kv_cache) {
            llama_kv_cache_view_update(ctx, &kvc_view);
            dump_kv_cache_view_seqs(kvc_view, 40);
        }

        llama_batch_clear(batch);

        // decode any currently ongoing sequences
        for (auto & client : clients) {
            if (client.seq_id == -1) {
                continue;
            }

            client.i_batch = batch.n_tokens;

            llama_batch_add(batch, client.sampled, n_tokens_system + client.n_prompt + client.n_decoded, { client.id }, true);

            client.n_decoded += 1;
        }

        if (batch.n_tokens == 0) {
            // all sequences have ended - clear the entire KV cache
            for (int i = 0; i < n_clients; ++i) {
                llama_kv_cache_seq_rm(ctx, i, n_tokens_system, -1);
            }

            LOG_TEE("%s: clearing the KV cache\n", __func__);
        }

        // insert new sequences for decoding
        if (cont_batching || batch.n_tokens == 0) {
            for (auto & client : clients) {
                if (client.seq_id == -1 && g_seq_id < n_seq) {
                    client.seq_id = g_seq_id;

                    client.t_start_prompt = ggml_time_us();
                    client.t_start_gen    = 0;

                    client.input    = k_prompts[rand() % k_prompts.size()];
                    client.prompt   = client.input + k_example;
                    client.response = "";

                    llama_sampling_reset(client.ctx_sampling);

                    // do not prepend BOS because we have a system prompt!
                    std::vector<llama_token> tokens_prompt;
                    tokens_prompt = ::llama_tokenize(ctx, client.prompt, false);

                    for (size_t i = 0; i < tokens_prompt.size(); ++i) {
                        llama_batch_add(batch, tokens_prompt[i], i + n_tokens_system, { client.id }, false);
                    }

                    // extract the logits only for the last token
                    if (batch.n_tokens > 0) {
                        batch.logits[batch.n_tokens - 1] = true;
                    }

                    client.n_prompt  = tokens_prompt.size();
                    client.n_decoded = 0;
                    client.i_batch   = batch.n_tokens - 1;

                    LOG_TEE("\033[31mClient %3d, seq %4d, started decoding ...\033[0m\n", client.id, client.seq_id);

                    g_seq_id += 1;

                    // insert new requests one-by-one
                    //if (cont_batching) {
                    //    break;
                    //}
                }
            }
        }

        if (batch.n_tokens == 0) {
            break;
        }

        // process in chunks of params.n_batch
        int32_t n_batch = params.n_batch;

        for (int32_t i = 0; i < (int32_t) batch.n_tokens; i += n_batch) {
            // experiment: process in powers of 2
            //if (i + n_batch > (int32_t) batch.n_tokens && n_batch > 32) {
            //    n_batch /= 2;
            //    i -= n_batch;
            //    continue;
            //}

            const int32_t n_tokens = std::min(n_batch, (int32_t) (batch.n_tokens - i));

            llama_batch batch_view = {
                n_tokens,
                batch.token    + i,
                nullptr,
                batch.pos      + i,
                batch.n_seq_id + i,
                batch.seq_id   + i,
                batch.logits   + i,
                0, 0, 0, // unused
            };

            const int ret = llama_decode(ctx, batch_view);
            if (ret != 0) {
                if (n_batch == 1 || ret < 0) {
                    // if you get here, it means the KV cache is full - try increasing it via the context size
                    LOG_TEE("%s : failed to decode the batch, n_batch = %d, ret = %d\n", __func__, n_batch, ret);
                    return 1;
                }

                LOG("%s : failed to decode the batch, retrying with n_batch = %d\n", __func__, n_batch / 2);

                n_cache_miss += 1;

                // retry with half the batch size to try to find a free slot in the KV cache
                n_batch /= 2;
                i -= n_batch;

                continue;
            }

            LOG("%s : decoded batch of %d tokens\n", __func__, n_tokens);

            for (auto & client : clients) {
                if (client.i_batch < (int) i || client.i_batch >= (int) (i + n_tokens)) {
                    continue;
                }

                //printf("client %d, seq %d, token %d, pos %d, batch %d\n",
                //        client.id, client.seq_id, client.sampled, client.n_decoded, client.i_batch);

                const llama_token id = llama_sampling_sample(client.ctx_sampling, ctx, NULL, client.i_batch - i);

                llama_sampling_accept(client.ctx_sampling, ctx, id, true);

                if (client.n_decoded == 1) {
                    // start measuring generation time after the first token to make sure all concurrent clients
                    // have their prompt already processed
                    client.t_start_gen = ggml_time_us();
                }

                const std::string token_str = llama_token_to_piece(ctx, id);

                client.response += token_str;
                client.sampled = id;

                //printf("client %d, seq %d, token %d, pos %d, batch %d: %s\n",
                //        client.id, client.seq_id, id, client.n_decoded, client.i_batch, token_str.c_str());

                if (client.n_decoded > 2 &&
                        (id == llama_token_eos(model) ||
                         (params.n_predict > 0 && client.n_decoded + client.n_prompt >= params.n_predict) ||
                         client.response.find("User:") != std::string::npos ||
                         client.response.find('}') != std::string::npos)) {
                    // basic reverse prompt
                    const size_t pos = client.response.find("User:");
                    if (pos != std::string::npos) {
                        client.response = client.response.substr(0, pos);
                    }

                    // delete only the generated part of the sequence, i.e. keep the system prompt in the cache
                    llama_kv_cache_seq_rm(ctx, client.id, n_tokens_system, -1);

                    const auto t_main_end = ggml_time_us();

                    LOG_TEE("\033[31mClient %3d, seq %3d/%3d, prompt %4d t, response %4d t, time %5.2f s, speed %5.2f t/s, cache miss %d \033[0m \nInput:    %s\n\033[35mResponse: %s\033[0m\n\n",
                            client.id, client.seq_id, n_seq, client.n_prompt, client.n_decoded,
                            (t_main_end - client.t_start_prompt) / 1e6,
                            (double) (client.n_prompt + client.n_decoded) / (t_main_end - client.t_start_prompt) * 1e6,
                            n_cache_miss,
                            ::trim(client.input).c_str(),
                            ::trim(client.response).c_str());

                    n_total_prompt += client.n_prompt;
                    n_total_gen    += client.n_decoded;

                    client.seq_id = -1;
                }

                client.i_batch = -1;
            }
        }
    }

    const auto t_main_end = ggml_time_us();

    print_date_time();

    LOG_TEE("\n%s: n_parallel = %d, n_sequences = %d, cont_batching = %d, system tokens = %d\n", __func__, n_clients, n_seq, cont_batching, n_tokens_system);
    if (params.prompt_file.empty()) {
        params.prompt_file = "used built-in defaults";
    }
    LOG_TEE("External prompt file: \033[32m%s\033[0m\n", params.prompt_file.c_str());
    LOG_TEE("Model and path used:  \033[32m%s\033[0m\n\n", params.model.c_str());

    LOG_TEE("Total prompt tokens: %6d, speed: %5.2f t/s\n", n_total_prompt, (double) (n_total_prompt              ) / (t_main_end - t_main_start) * 1e6);
    LOG_TEE("Total gen tokens:    %6d, speed: %5.2f t/s\n", n_total_gen,    (double) (n_total_gen                 ) / (t_main_end - t_main_start) * 1e6);
    LOG_TEE("Total speed (AVG):   %6s  speed: %5.2f t/s\n", "",             (double) (n_total_prompt + n_total_gen) / (t_main_end - t_main_start) * 1e6);
    LOG_TEE("Cache misses:        %6d\n", n_cache_miss);

    LOG_TEE("\n");

    llama_print_timings(ctx);

    llama_batch_free(batch);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    fprintf(stderr, "\n\n");

    return 0;
}
