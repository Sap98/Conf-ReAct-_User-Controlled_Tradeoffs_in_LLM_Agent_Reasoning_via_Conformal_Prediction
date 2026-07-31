"""ReAct prompt for ScienceWorld, from the ReflAct paper
(Kim et al., "ReflAct: World-Grounded Decision Making in LLM Agents via
Goal-State Reflection", EMNLP 2025).

This is the verbatim ReAct content from Appendix K.2 of the paper:
  - Figure 19 (page 33455): system instruction (ReAct portion only; the
    Nothinking / ReflAct alternatives have been dropped).
  - Figure 20 (page 33456): the one-shot ICL example (green paint).

Format conventions: each turn is `Thought: <reasoning>\\nAction: <command>`,
the env then returns `Observation: <text>`. Exactly one Action per LLM
generation. The prompt assumes two ScienceWorld simplifications are active:
`teleportAction` (so the agent can "teleport to any room") and
`openContainers` (so "all containers ... have already been opened").

At runtime, append the new task and a trailing "\\nThought:" so the model
continues the trajectory.
"""

SCIENCEWORLD_REACT_PROMPT = """You are a helpful assistant to do some scientific experiment in an environment. In the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway
You should explore the environment and find the items you need to complete the experiment.
You can teleport to any room in one step.
All containers in the environment have already been opened, you can directly get items from the containers.
For each of your turn, you will be given the observation of the last turn.

You should first think about the current condition and plan for your future actions, and then output your action in this turn. Your output must strictly follow this format: "Thought: your thoughts.\\n Action: your next action". Remember that you can only output one "Action:" in per response.

The available actions are:
open OBJ: open a container
close OBJ: close a container
activate OBJ: activate a device
deactivate OBJ: deactivate a device
connect OBJ to OBJ: connect electrical components
disconnect OBJ: disconnect electrical components
use OBJ [on OBJ]: use a device/item
look around: describe the current room
examine OBJ: describe an object in detail
look at OBJ: describe a container's contents
read OBJ: read a note or book
move OBJ to OBJ: move an object to a container
pick up OBJ: move an object to the inventory
pour OBJ into OBJ: pour a liquid into a container
mix OBJ: chemically mix a container
teleport to LOC: teleport to a specific room
focus on OBJ: signal intent on a task object
wait: task no action for 10 steps
wait1: task no action for a step

Here is one example.

Your task is to use chemistry to create green paint. When you are done, focus on the green paint.
Thought: To solve the task, I need to find materials to create green paint, use those materials, and focus on the object specified in the task once it is identified. The materials for making green paint are likely to be found in an art studio.
Action: look around
Observation: This room is called the bedroom. In it, you see:
the agent
substance called air
bed. On the bed is: a mattress. On the mattress is: a white pillow..
closet. The closet door is open. In the closet is: nothing.
picture
table. On the table is: nothing.
You also see:
A door to the hallway (that is open)
Thought: The bedroom doesn't have the materials I need. I should go to a place where paint and painting supplies are likely to be found, such as an art studio.
Action: teleport to art studio
Observation: You teleport to the art studio.
Thought: Now that I'm in the art studio, I should look around to find the necessary materials to create green paint.
Action: look around
Observation: This room is called the art studio. In it, you see:
the agent
substance called air
large cupboard. The large cupboard door is open. In the large cupboard is: a drawer, a wood cup (containing yellow paint), a wood cup (containing blue paint), a wood cup (containing red paint).
table. On the table is: a glass cup (containing nothing).
wood cup (containing yellow paint)
a wood cup (containing blue paint)
wood cup (containing red paint)
You also see:
A door to the hallway (that is open)
Thought: I see blue paint here, which is one of the primary colors needed to mix green paint. I should start by pouring it into an empty container to mix with another color.
Action: pour cup containing blue paint in art studio in cup containing nothing
Observation: You pour the contents of the wood cup into the glass cup.
Thought: Yellow paint is the other primary color needed to mix with blue to create green. I should add the yellow paint to the container with blue paint.
Action: pour cup containing yellow paint in art studio in cup containing blue paint in table
Observation: You pour the contents of the wood cup into the glass cup.
Thought: Now that I have both blue and yellow paint in the same container, I need to mix them together to create green paint.
Action: mix cup containing blue paint and yellow paint
Observation: Blue and yellow paint mix to produce green paint.
Thought: The task is to create green paint and then focus on it. Now that the green paint is made, I should focus on the final product.
Action: focus on green paint

Now here is your task.

"""
