import sys
import os
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model import ExLlama, ExLlamaConfig, ExLlamaCache
from generator import ExLlamaGenerator
from flask import Flask, render_template, request, jsonify
from flask import Response, stream_with_context
from threading import Timer, Lock
import webbrowser
import json
import model_init
import argparse
from tokenizer import ExLlamaTokenizer
from waitress import serve

app = Flask(__name__)
generate_lock = Lock()

model: ExLlama
tokenizer: ExLlamaTokenizer
cache: ExLlamaCache
generator: ExLlamaGenerator

def set_generator_defaults(generator, data):
    generator.settings.temperature = data.get("temperature", 0.95)
    generator.settings.top_p = data.get("top_p", 0.75)
    generator.settings.min_p = data.get("min_p", 0.0)
    generator.settings.top_k = data.get("top_k", 0)
    generator.settings.typical = data.get("typical", 0.25)
    generator.settings.token_repetition_penalty_max = data.get("token_repetition_penalty_max", 1.15)
    generator.settings.token_repetition_penalty_sustain = data.get("token_repetition_penalty_sustain", 2048)
    generator.settings.token_repetition_penalty_decay = data.get("token_repetition_penalty_decay", 512)

def generate_text_completion(generator, prompt, max_response_tokens, chunk_size, stop_conditions):
    
    reused = generator.gen_begin_empty()

    prompt_tokens = tokenizer.encode(prompt)
    num_prompt_tokens = prompt_tokens.shape[-1]

    generator.gen_feed_tokens(prompt_tokens)

    generator.begin_beam_search()

    stop_condition = False
    held_text = ""

    res_line = ""
    num_res_tokens = 0

    for i in range(max_response_tokens):

        # Truncate the past if the next chunk might generate past max_seq_length

        if generator.sequence_actual is not None:
            if generator.sequence_actual.shape[
                -1] + chunk_size + generator.settings.beam_length + 1 > model.config.max_seq_len:
                generator.gen_prune_left(chunk_size)

        # Get the token and append to sequence

        gen_token = generator.beam_search()

        # If token is EOS, replace it with newline before continuing

        if gen_token.item() == tokenizer.eos_token_id:
            generator.replace_last_token(tokenizer.newline_token_id)

        # Decode current line to get new characters added (decoding a single token gives incorrect results
        # sometimes due to hoe SentencePiece works)

        prev_res_line = res_line
        num_res_tokens += 1
        res_line = tokenizer.decode(generator.sequence_actual[0, -num_res_tokens:])
        new_text = res_line[len(prev_res_line):]

        # Since SentencePiece is slightly ambiguous, the first token produced after a newline may not be the
        # same that is reproduced when we encode the text later, even though it encodes the same string

        if num_res_tokens == 1 and len(new_text) > 0:
            replace = tokenizer.encode(new_text)[0]
            if replace.shape[-1] == 1: generator.replace_last_token(replace)

        # Stop conditions

        if gen_token.item() == tokenizer.eos_token_id:
            stop_condition = True
            break

        # for stop_tokens, stop_string in stop_conditions:
        #     if res_line.lower().endswith(stop_string.lower()):
        #         generator.gen_rewind(
        #             stop_tokens.shape[-1] - (1 if stop_tokens[0, 0].item() == tokenizer.newline_token_id else 0))
        #         res_line = res_line[:-len(stop_string)]
        #         stop_condition = True
        #         break
        #         
        # if stop_condition: break

    generator.end_beam_search()

    return {
        'status': 'ok',
        'completion': res_line, 
        'num_prompt_tokens': num_prompt_tokens,
        'num_completion_tokens': num_res_tokens
    }
    

@app.route('/api/v1/completions', methods=['POST'])
def api_text_completion():
    print(f'Got request')
    global model, tokenizer, cache, generator
    data = request.get_json()


    break_on_newline = data.get("gen_endnewline", False)
    max_response_tokens = data.get("max_response_tokens", 2048)
    chunk_size = data.get("chunk_size", 32)

    set_generator_defaults(generator, data)

    stop_conditions = [ (torch.Tensor([[tokenizer.eos_token_id]]).long(), None) ]

    with generate_lock:
        completion = generate_text_completion(
            generator,
            data.get("prompt", "No prompt specified"),
            max_response_tokens,
            chunk_size,
            stop_conditions
        )
        result = Response(json.dumps(completion), mimetype='application/json')
        return result

@app.route('/api/v1/chat/completions', methods=['POST'])
def api_chat_completion():
    return Response("{'status': 'error', 'err_msg': 'Not implemented'}", mimetype='application/json')

# Load the model

parser = argparse.ArgumentParser(description="Simple web-based chatbot for ExLlama")
parser.add_argument("-host", "--host", type = str, help = "IP:PORT eg, 0.0.0.0:7862", default = "localhost:5000")
parser.add_argument("-sd", "--sessions_dir", type = str, help = "Location for storing user sessions, default: ~/exllama_sessions/", default = "~/exllama_sessions/")

model_init.add_args(parser)
args = parser.parse_args()
model_init.post_parse(args)
model_init.get_model_files(args)

model_init.print_options(args)
config = model_init.make_config(args)

print(f" -- Loading model...")
model = ExLlama(config)

print(f" -- Loading tokenizer...")
tokenizer = ExLlamaTokenizer(args.tokenizer)

print(f" -- Creating cache...")
cache = ExLlamaCache(model)

print(f" -- Creating generator...")
generator = ExLlamaGenerator(model, tokenizer, cache)

model_init.print_stats(model)

# Start the web server

machine = args.host
host, port = machine.split(":")

serve(app, host = host, port = port)
