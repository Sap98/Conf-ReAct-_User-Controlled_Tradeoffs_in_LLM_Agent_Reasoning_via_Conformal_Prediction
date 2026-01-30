# This file is created to consider both think and action tags and then play all the games.

from collections import Counter
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
from sentence_transformers import SentenceTransformer, util
import networkx as nx

temp = 0.8
top_p = 1
top_k = 20

print(f"Temperature :{temp},\nTop_p: {top_p},\nTop_k: {top_k}")

def format_time(timestamp: int):
    readable_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
    return readable_time

start_time = time.time()
print("This is the starting time of the code: ", format_time(start_time))

warnings.filterwarnings("ignore", category=UserWarning, module="transformers")

# Authenticate if accessing a private repository
login(token="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")

# Load LLaMA 7B model (Change to your preferred model)
model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Apply 4-bit or 8-bit quantization
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Set to False for 8-bit quantization
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,  # Further reduces memory usage
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map="auto"  # Automatically assigns GPU/CPU
)

def convert_time(duration: int):
    hours, remainder = divmod(duration, 3600)
    minutes, seconds = divmod(remainder, 60)
    return hours, minutes, seconds


def llama3_infer(prompt, max_tokens=100):
    prompt = prompt.strip()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    stop_token = '\n'
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_tokens,
        num_return_sequences=10,
        do_sample=True,  # Greedy decoding instead of temperature=0
        eos_token_id=tokenizer.encode(stop_token)[-1],
        pad_token_id=tokenizer.eos_token_id,  # Prevents warnings
        temperature = 0.8,
        top_k=20,
        top_p=1
    )
    end_time = time.time()

    multiple_responses = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    # Manually enforce stop sequences
    # for stop_token in stop:
    #     response = response.split(stop_token)[0]

    return multiple_responses, end_time - start_time


def extract_first_tag_block(multi_response):
    first_tags = []
    for response in multi_response:  # Loop through each response string
        lines = [line.strip() for line in response.split("\n") if line.strip()]
        if not lines:
            continue  # Skip empty responses
        
        first_line = lines[0]
        if first_line.startswith("think:"):
            first_tags.append(first_line.strip())
        elif first_line is not None:
            first_tags.append(first_line.strip())
        # else:
        #     first_tags.append(first_line)  # Fallback
    return first_tags


with open('/home/saptarshi/alfworld/ReAct/base_config.yaml') as reader:
    config = yaml.safe_load(reader)
    
split = "eval_out_of_distribution"

env = get_environment(config["env"]["type"])(config, train_eval=split)
env = env.init_env(batch_size=1)

def process_ob(ob):
    if ob.startswith('You arrive at loc '):
        ob = ob[ob.find('. ')+2:]    
    return ob


with open('/home/saptarshi/alfworld/ReAct/prompts/alfworld_3prompts.json', 'r') as f:
    d = json.load(f)


def classify_tag(tag_line):
    if tag_line.startswith('think'):
        return 'think'
    else:
        return 'action'

def majority_voting_on_tags(tag_types):
    counter = Counter(tag_types)
    return counter.most_common(1)[0]


def majority_voting_on_actions(multi_responses)-> str:
    actions_counts = Counter(multi_responses)
    print("\n The counts of each actions are: ", actions_counts)
    sorted_actions = actions_counts.most_common()
    for action, _ in sorted_actions:
        if action is not None:
            return action.strip()       
    return None


def semantic_graph_clustering(actions):
    for _, action in enumerate(actions):
        if 'think: ' in action:
            all_actions = [action.split('think: ')[1] for action in actions if 'think: ' in action]
    print("\nAll the actions are: ", all_actions)
    unique_think_tags = list(set(all_actions))
    print("\nUnique think tags are: ", unique_think_tags)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(unique_think_tags, convert_to_tensor=True)
    similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)
    print("\nSimilarity matrix is: ", similarity_matrix)

    G = nx.Graph()
    for idx, tag in enumerate(unique_think_tags):
        G.add_node(idx, text=tag)

    threshold = 0.8
    N = len(unique_think_tags)

    for i in range(N):
        for j in range(i+1, N):
            sim = similarity_matrix[i][j].item()
            if sim >= threshold:
                G.add_edge(i, j, weight=sim)

    clusters = list(nx.connected_components(G)) # finds all connected components in the undirected graph G using NetworkX
    largest_cluster = max(clusters, key=len)
    largest_think_tags = [G.nodes[n]['text'] for n in largest_cluster]
    return largest_think_tags[0]



def alfworld_run(prompt, to_print=True, ob=''):
    current_game_total_infer_time = 0
    init_prompt = prompt + ob + '\n>'
    prompt = ''
    if to_print:
        print(ob)
        sys.stdout.flush()
    for i in range(1, 30):
        multiple_responses, infer_time = llama3_infer(init_prompt + prompt)
        multi_response = []
        for _, responses in enumerate(multiple_responses):
            response = responses[len(init_prompt + prompt):].strip() if responses.startswith(init_prompt + prompt) else responses
            multi_response.append(response)
        # for index, responses in enumerate(multi_response):
        #     print(f"Response {index + 1}: {responses}")
        current_game_total_infer_time += infer_time

        all_actions = extract_first_tag_block(multi_response) 
        # print("\nAll actions: ", all_actions)
        tag_classes = [classify_tag(tag) for tag in all_actions]
        # print("\n Tag classes: ", tag_classes)
        winner, count = majority_voting_on_tags(tag_classes)
        # print(f"After majority voting on the tags: {winner}:{count}")
        if winner == 'action':
            actions_for_mv = []
            for _, actions in enumerate(all_actions):
                if not actions.startswith('think: '):
                    actions_for_mv.append(actions)
            action = majority_voting_on_actions(actions_for_mv)
            print("\nAction is: ", action)
        else:
            actions_for_mv = []
            for _, actions in enumerate(all_actions):
                if actions.startswith('think: '):
                    actions_for_mv.append(actions)
            action = semantic_graph_clustering(actions_for_mv)
            action = "think: " + action
            print("\nAction is: ", action)
        observation, reward, done, info = env.step([action])
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        if action.startswith('think:'):
            observation = 'OK.'
        if to_print:
            print(f'Act {i}: {action}\nObs {i}: {observation}')
            sys.stdout.flush()
        prompt += f' {action}\n{observation}\n>'
        print("\n This is the prompt for the next iteration: ", prompt)
        if done:
            return reward, current_game_total_infer_time
    return 0, current_game_total_infer_time

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

for _ in range(134):
    task_type = ''
    reward = False
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
            prompt = 'Interact with a household to solve a task. Here are two examples.\n' + d[f'react_{v}_0'] + d[f'react_{v}_1'] + '\nHere is the task.\n'
            print(f"This is the task_descrip and its mapping to one of the 6 tasks of ALFWorld: {(k, v)}\n")
            print(f"This is the prompt given to Llama-3.1-8B:\n {prompt}\n")
            task_type = v
            # print(f"Observation: {ob}")
            #break
            r,current_game_infer_time = alfworld_run(prompt, ob=ob)
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
    
    if reward:
        if task_type in time_per_game:
            time_per_game[task_type].append(current_game_infer_time)
        else:
            time_per_game[task_type] = []
            time_per_game[task_type].append(current_game_infer_time)
    print("\nThis is the time taken by the current game: {:02.0f}:{:02.0f}:{:02.0f}".format(*convert_time(current_game_infer_time)))
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





