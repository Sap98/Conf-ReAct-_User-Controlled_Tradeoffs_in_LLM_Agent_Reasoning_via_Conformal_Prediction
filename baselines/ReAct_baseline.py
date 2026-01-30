import re
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
from alfworld.agents.environment import get_environment
import torch
import json
import sys
import warnings
import time 
import statistics
from conformal_prediction.API_llm_inference import OllamaClient

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
model = "qwen3:8b"
# Initialize the Ollama client for API-based inference
client = OllamaClient(base_url="http://10.5.30.32:11434", model="qwen3:8b")


def llm_infer(prompt, max_tokens=215):
    """
    Inference using the Ollama API endpoint instead of local model.
    """
    response = client.generate(prompt=prompt)
    return response.strip()



def format_time(timestamp: int):
    readable_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
    return readable_time

start_time = time.time()
print("This is the starting time of the code: ", format_time(start_time))

# warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# # Authenticate if accessing a private repository
# login(token="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")

# # Load LLaMA 7B model (Change to your preferred model)
# model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # Apply 4-bit or 8-bit quantization
# quantization_config = BitsAndBytesConfig(
#     load_in_4bit=True,  # Set to False for 8-bit quantization
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True,  # Further reduces memory usage
# )

# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     quantization_config=quantization_config,
#     device_map="auto"  # Automatically assigns GPU/CPU
# )

def convert_time(duration: int):
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    return hours, minutes, seconds


# def llama3_infer(prompt, max_tokens=100):
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
#     stop_token = '\n'
#     output = model.generate(
#         **inputs,
#         max_new_tokens=max_tokens,
#         do_sample=True,  # Greedy decoding instead of temperature=0
#         eos_token_id=tokenizer.encode(stop_token)[-1],
#         pad_token_id=tokenizer.eos_token_id,  # Prevents warnings
#         temperature = 0.1,
#         top_k=20,
#         top_p=1
#     )

#     response = tokenizer.decode(output[0], skip_special_tokens=True)

#     # Manually enforce stop sequences
#     # for stop_token in stop:
#     #     response = response.split(stop_token)[0]

#     return extract_first_tag_block(response.strip()).strip()


def extract_first_tag_block(text):
    lines = text.strip().split('\n')
    tag_block = []
    in_tag = False

    for line in lines:
        if line.startswith('>'):
            if in_tag:
                # We already captured the first tag block, so break
                break
            else:
                # Start capturing the first tag block
                in_tag = True
                tag_block.append(line)
        elif in_tag:
            # Continuation of the tag block (non '>' line)
            tag_block.append(line)
    
    return '\n'.join(tag_block) if tag_block else None


# def extract_action(llm_response: str) -> str:
#     lines = llm_response.split("\n")
    
#     for line in lines:
#         # Find lines that start with '>' or '>>' but are not 'think' statements
#         match = re.match(r"> (?!think:)(.+)", line.strip())
#         if match:
#             return match.group(1).strip()  # Extract the action part
    
#     return None  # Return None if no valid action is found


with open('/home/saptarshi/transfer/alfworld/ReAct/base_config.yaml') as reader:
    config = yaml.safe_load(reader)
    
split = "eval_out_of_distribution"

env = get_environment(config["env"]["type"])(config, train_eval=split)
env = env.init_env(batch_size=1)

def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob


with open('/home/saptarshi/transfer/alfworld/ReAct/prompts/alfworld_3prompts.json', 'r') as f:
    d = json.load(f)


def alfworld_run(prompt, to_print=True, ob=''):
    init_prompt = prompt + ob + '\n>'
    prompt = ''
    if to_print:
        print(ob)
        sys.stdout.flush()
    for i in range(1, 5):
        action = llm_infer(init_prompt + prompt).strip()
        print("\n This is the action from llm: ",action)
        if action.startswith('> think:'):
            observation = 'OK.'
        else:
            observation, reward, done, info = env.step([action])
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        if to_print:
            print(f'Act {i}: {action}\nObs {i}: {observation}')
            sys.stdout.flush()
        prompt += f' {action}\n{observation}\n>'
        print(f"\nThe prompt for next iteration is: {prompt}\n")
        if done:
            return reward
    return 0

prefixes = {
    'pick_and_place': 'put',
    'pick_clean_then_place': 'clean',
    'pick_heat_then_place': 'heat',
    'pick_cool_then_place': 'cool',
    'look_at_obj': 'examine',
    'pick_two_obj': 'puttwo'
}
cnts = [0] * 6
rs = [0] * 6
games = {}
time_per_game = {
    'put':[],
    'clean':[],
    'heat':[],
    'cool':[],
    'examine':[],
    'put two':[]
}

for _ in range(1):
    current_game_start_time = time.time()
    task_type = ''
    reward = False
    print("\nThis is the start time for the current game: ", format_time(current_game_start_time))
    ob, info = env.reset() # Initialization of an instance
    #print(f"Before preprocessing, ob is: {ob}") Output: ('-= Welcome to TextWorld, ALFRED! =-\n\nYou are in the middle of a room. Looking quickly around you..........)
    ob = '\n'.join(ob[0].split('\n\n')[1:]) # Removes the first sentence from the observation which is considered to be redundant and creates a new string
    #print(f"After pre-processing, ob is: {ob}") Output: You are in the middle of a room. Looking quickly around you..........
    #break
    print("<<--------AlfWorld Loaded------>>\n")
    name = '/'.join(info['extra.gamefile'][0].split('/')[-3:-1]) # 'name' stores the extracted task name so that it can be classified among the 6 tasks of ALFWorld
    # print(f"NAME: ------->>>>{name}\n")
    print(f"This is the OBSERVATION after initialization of the ALFWorld environment:\n {ob}\n")
    for i, (k, v) in enumerate(prefixes.items()):
        if name.startswith(k):
            prompt = 'Interact with a household to solve a task. Here are two examples.\n' + d[f'react_{v}_0'] + d[f'react_{v}_2'] + '\nHere is the task.\n'
            print(f"This is the task_descrip and its mapping to one of the 6 tasks of ALFWorld: {(k, v)}\n")
            print(f"This is the prompt given to Llama-3.1-8B:\n {prompt}\n")
            task_type = v
            # print(f"Observation: {ob}")
            #break
            r = alfworld_run(prompt, ob=ob)
            reward = r
            print("\nThis is the task type: ", task_type)
            print("\nThis is the reward: ", reward)
            game_name = ob.split('Your task is to:')
            game_task = game_name[1].strip()
            path = info['extra.gamefile'][0]
            if game_task not in games:
                games[game_task] = [path, 'Won' if r else 'Lost']
            rs[i] += r
            cnts[i] += 1
            break
    current_game_end_time = time.time()
    current_game_time = current_game_end_time - current_game_start_time
    if reward:
        if task_type in time_per_game:
            time_per_game[task_type].append(current_game_time)
        else:
            time_per_game[task_type] = []
            time_per_game[task_type].append(current_game_time)
    print("\nThis is the time taken by the current game: {:02.0f}:{:02.0f}:{:02.0f}".format(*convert_time(current_game_time)))
    print(_+1, 'r', r, 'rs', rs, 'cnts', cnts, 'sum(rs)/sum(cnts)', sum(rs) / sum(cnts))
    print("Results are: \n")
    if cnts[0]:
        print(f"put_task: {(rs[0]/cnts[0])*100}%")
    if cnts[1]:
        print(f"clean_task: {(rs[1]/cnts[1])*100}%")
    if cnts[2]:
        print(f"heat_task: {(rs[2]/cnts[2])*100}%")
    if cnts[3]:
        print(f"cool_task: {(rs[3]/cnts[3])*100}%")
    if cnts[4]:
        print(f"examine_task: {(rs[4]/cnts[4])*100}%")
    if cnts[5]:
        print(f"put_two_task: {(rs[5]/cnts[5])*100}%")
    print("\nThe game dict is: ")
    print(games)
    print('------------\n')
    print('<-------------------------------------START OF NEW GAME--------------------------------------------------------->')
    print('<--------------------------------------------------------------------------------------------------------------->\n')

print(time_per_game)
print("\nThe average time taken by each type of task: ")
for key, value in time_per_game.items():
    if len(value) == 0:
        print(f"This task {key} has no game")
    else:
        print("Task: {} ; average time taken is: {:.2f} seconds and median time taken is: {:.2f}".format(key, sum(value)/len(value), statistics.median(value)))
        

# print("\nThis is the average time taken by each game: {:02.0f}:{:02.0f}:{:02.0f}".format(*convert_time(average_time_per_game)))
end_time = time.time()
total_time = end_time - start_time
print("\nThis is the total time taken to run the entire code: {:02.0f}:{:02.0f}:{:02.0f}".format(*convert_time(total_time)))





