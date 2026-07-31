"""ReflAct prompt for ScienceWorld, from the ReflAct paper
(Kim et al., "ReflAct: World-Grounded Decision Making in LLM Agents via
Goal-State Reflection", EMNLP 2025).

This is the verbatim ReflAct content from Appendix K.2 of the paper:
  - Figure 19 (page 33455): system instruction (ReflAct portion only; the
    Nothinking / ReAct alternatives have been dropped).
  - Figure 21 (page 33456): the one-shot ICL example (green paint).

ReflAct counterpart of scienceworld_react_prompt.py. Per the paper the ONLY
differences vs the ReAct prompt are:
  1. The instruction tells the agent to "reflect on the agent's state,
     including the location, inventory, and focused object, in relation to the
     task goal" (rather than ReAct's "think ... and plan for your future
     actions"), and the turn tag is "Reflection:" instead of "Thought:".
  2. In the one-shot example, the middle reasoning steps are rewritten as
     ReflAct goal-state reflections of the form "Currently, I am in the
     <location>, holding <inventory> and focused on <object>. <relation to the
     goal>." The first and last reflections are left identical to ReAct's
     thoughts. Actions and observations are untouched.

Format conventions: each turn is `Reflection: <goal-state reflection>\\nAction:
<command>`, the env then returns `Observation: <text>`. Exactly one Action per
LLM generation. The prompt assumes two ScienceWorld simplifications are active:
`teleportAction` (so the agent can "teleport to any room") and `openContainers`
(so "all containers ... have already been opened").

At runtime, append the new task and a trailing "\\nReflection:" so the model
continues the trajectory.
"""

SCIENCEWORLD_REFLACT_PROMPT = """You are a helpful assistant to do some scientific experiment in an environment. In the environment, there are several rooms: kitchen, foundry, workshop, bathroom, outside, living room, bedroom, greenhouse, art studio, hallway
You should explore the environment and find the items you need to complete the experiment.
You can teleport to any room in one step.
All containers in the environment have already been opened, you can directly get items from the containers.
For each of your turn, you will be given the observation of the last turn.

You should first reflect on the agent's state, including the location, inventory, and focused object, in relation to the task goal. Then, output the action for this turn. Your output must strictly follow this format: "Reflection: your reflection.\\n Action: your next action".

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
Reflection: To solve the task, I need to find materials to create green paint, use those materials, and focus on the object specified in the task once it is identified. The materials for making green paint are likely to be found in an art studio.
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
Reflection: Currently, I am in the bedroom, holding nothing and focused on nothing. There are no useful materials here to create the green paint I need to focus on.
Action: teleport to art studio
Observation: You teleport to the art studio.
Reflection: Currently, I am in the art studio, holding nothing and focused on nothing. I need to look for materials to create green paint.
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
Reflection: Currently, I am in the art studio, not holding anything. I've found the primary colors yellow, blue, and red, which can be used to make green paint, the substance I will need to focus on.
Action: pour cup containing blue paint in art studio in cup containing nothing
Observation: You pour the contents of the wood cup into the glass cup.
Reflection: I poured blue paint into the glass cup. I'm now holding it and will next add yellow paint.
Action: pour cup containing yellow paint in art studio in cup containing blue paint in table
Observation: You pour the contents of the wood cup into the glass cup.
Reflection: I've added yellow paint to the blue in the glass cup. I'm now holding a mix of blue and yellow paint, ready to be mixed into green.
Action: mix cup containing blue paint and yellow paint
Observation: Blue and yellow paint mix to produce green paint.
Reflection: The task is to create green paint and then focus on it. Now that the green paint is made, I should focus on the final product.
Action: focus on green paint

Now here is your task.

"""
